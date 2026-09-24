#!/usr/bin/env python3
"""Train the full MP + LC + HN method on official Chinese-CLIP ViT-H/14."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mmmrag.io import read_jsonl
from mmmrag.full_method import (
    hard_negative_ranking_loss,
    local_contrastive_loss,
    multi_positive_contrastive_loss,
    relation_aware_batches,
    relation_maps,
)
from mmmrag.lora import inject_lora, load_lora_adapter, save_lora_adapter
from scripts.run_visual_retrieval import hard_negative_metrics, multi_positive_metrics


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True); p.add_argument("--validation",required=True)
    p.add_argument("--hard-negatives",required=True); p.add_argument("--validation-hard-negatives",required=True); p.add_argument("--model",required=True)
    p.add_argument("--init-adapter"); p.add_argument("--output-dir",required=True)
    p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--learning-rate",type=float,default=1e-5); p.add_argument("--local-learning-rate",type=float,default=1e-4)
    p.add_argument("--lambda-lc",type=float,default=.1); p.add_argument("--lambda-hn",type=float,default=.1)
    p.add_argument("--margin",type=float,default=.1); p.add_argument("--tau-local",type=float,default=.07)
    p.add_argument("--joint-eta",type=float,default=.1)
    p.add_argument("--hn-score",choices=("local","joint","joint_detached"),default="joint_detached")
    p.add_argument("--global-loss",choices=("multi_positive","clip"),default="multi_positive")
    p.add_argument("--early-stop-score",choices=("global","joint"),default="global")
    p.add_argument("--patience",type=int,default=2); p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

def path_key(p: str) -> str: return Path(p).name

def main() -> None:
    a=args(); random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    device=torch.device("cuda"); root=Path.cwd(); out=Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise SystemExit(f"Output directory is not empty: {out}")
    out.mkdir(parents=True)
    from cn_clip import clip as cn_clip
    model_path=Path(a.model).resolve(); ckpt=next(model_path.glob("clip_cn_*.pt"))
    model,preprocess=cn_clip.load_from_name(str(ckpt),device=device,vision_model_name="ViT-H-14",text_model_name="RoBERTa-wwm-ext-large-chinese",input_resolution=224)
    if a.init_adapter:
        load_lora_adapter(model,Path(a.init_adapter).resolve())
    else:
        inject_lora(model,rank=8,alpha=16.,dropout=.05,vision_layers=4,text_layers=4)
    for p in model.parameters():
        if p.requires_grad: p.requires_grad=True
    # Local heads map patch/token states into a shared 512-D space.
    img_proj=nn.Linear(1280,512,bias=False).to(device); txt_proj=nn.Linear(1024,512,bias=False).to(device)
    train=read_jsonl(Path(a.train)); val=read_jsonl(Path(a.validation))
    hard=json.loads(Path(a.hard_negatives).read_text())
    validation_hard=json.loads(Path(a.validation_hard_negatives).read_text())
    hard_by_text, hard_by_image = relation_maps(hard)
    class DS(Dataset):
        def __init__(self,rows): self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i]
            with Image.open(r["image_path"]) as im: x=preprocess(im.convert("RGB"))
            return x,r["query"],r["positive_group_id"],path_key(r["image_path"]),r
    def collate(batch):
        x,t,g,k,r=zip(*batch); return {"images":torch.stack(x),"texts":cn_clip.tokenize(list(t),context_length=52),"texts_raw":list(t),"groups":list(g),"keys":list(k),"rows":list(r)}
    train_batches=relation_aware_batches(train,hard_by_text,a.batch_size,a.seed)
    loader=DataLoader(DS(train),batch_sampler=train_batches,num_workers=4,pin_memory=True,collate_fn=collate)
    vloader=DataLoader(DS(val),batch_size=a.batch_size,shuffle=False,num_workers=4,pin_memory=True,collate_fn=collate)
    params=[p for p in model.parameters() if p.requires_grad]+list(img_proj.parameters())+list(txt_proj.parameters())
    opt=torch.optim.AdamW([{"params":[p for p in model.parameters() if p.requires_grad],"lr":a.learning_rate},{"params":list(img_proj.parameters())+list(txt_proj.parameters()),"lr":a.local_learning_rate}],weight_decay=.01)
    def encode(batch, need_local=True):
        images=batch["images"].to(device); texts=batch["texts"].to(device)
        with torch.autocast("cuda",dtype=torch.float16):
            v=model.visual.conv1(images.type(model.dtype)).reshape(images.shape[0],1280,-1).permute(0,2,1)
            v=torch.cat([model.visual.class_embedding.to(v.dtype).expand(images.shape[0],1,-1),v],1)+model.visual.positional_embedding.to(v.dtype)
            v=model.visual.ln_pre(v).permute(1,0,2); v=model.visual.transformer(v).permute(1,0,2)
            bert=model.bert(texts,attention_mask=texts.ne(model.tokenizer.vocab["[PAD]"]).type(model.dtype))[0]
            im=model.visual.ln_post(v[:,0,:])
            if model.visual.proj is not None: im=im@model.visual.proj
            tx=bert[:,0,:]@model.text_projection
            im=F.normalize(im.float(),dim=-1); tx=F.normalize(tx.float(),dim=-1)
            vp=F.normalize(img_proj(v[:,1:,:].float()),dim=-1)
            wt=F.normalize(txt_proj(bert.float()),dim=-1)
        mask=texts.ne(model.tokenizer.vocab["[PAD]"])
        for token in ("[CLS]","[SEP]"): mask &= texts.ne(model.tokenizer.vocab[token])
        return im,tx,vp,wt,mask
    def local_scores(vp,wt,mask):
        chunks=[]
        for st in range(0, wt.shape[0], 4):
            qwt, qmask = wt[st:st+4], mask[st:st+4]
            sim=torch.einsum("qld,ipd->qilp",qwt,vp); tok=sim.amax(-1)
            m=qmask[:,None,:].float(); t2i=(tok*m).sum(-1)/m.sum(-1).clamp_min(1)
            sim=sim.masked_fill(~qmask[:,None,:,None],-1e4); i2t=sim.amax(-2).sum(-1)/m.sum(-1).clamp_min(1)
            chunks.append((t2i+i2t).mul(.5))
        return torch.cat(chunks)
    def loss_batch(batch):
        im,tx,vp,wt,mask=encode(batch); groups=batch["groups"]; B=len(groups)
        global_scores=tx@im.T
        glob=model.logit_scale.exp().float()*global_scores; ids={}; [ids.setdefault(g,[]).append(i) for i,g in enumerate(groups)]
        pos=torch.zeros((B,B),device=device,dtype=torch.bool)
        for q,g in enumerate(groups): pos[q,ids[g]]=True
        if a.global_loss == "multi_positive":
            lmp=multi_positive_contrastive_loss(glob,pos)
        else:
            targets=torch.arange(B,device=device)
            lmp=(F.cross_entropy(glob,targets)+F.cross_entropy(glob.T,targets))*.5
        local=local_scores(vp,wt,mask)
        llc=local_contrastive_loss(local,pos,a.tau_local)
        # HN is ranked with the deployed joint score, but its global branch is
        # detached so it cannot compete with the global multi-positive loss.
        joint=(global_scores+a.joint_eta*local)/(1+a.joint_eta)
        hn_scores = local if a.hn_score == "local" else joint
        if a.hn_score == "joint_detached":
            hn_scores=(global_scores.detach()+a.joint_eta*local)/(1+a.joint_eta)
        explicit_t2i=torch.zeros_like(pos)
        explicit_i2t=torch.zeros_like(pos)
        for i,(text,key) in enumerate(zip(batch["texts_raw"],batch["keys"])):
            for j,image_key in enumerate(batch["keys"]):
                explicit_t2i[i,j]=image_key in hard_by_text.get(text,set())
            for j,candidate_text in enumerate(batch["texts_raw"]):
                explicit_i2t[i,j]=candidate_text in hard_by_image.get(key,set())
        explicit_t2i &= ~pos
        explicit_i2t &= ~pos.t()
        lhn,hn_stats=hard_negative_ranking_loss(
            hn_scores,pos,explicit_t2i,explicit_i2t,a.margin
        )
        total=lmp+a.lambda_lc*llc+a.lambda_hn*lhn
        return total,{"L_MP":lmp.item(),"L_LC":llc.item(),"L_HN":lhn.item(),"total":total.item(),**hn_stats}
    # Validation records the requested metrics; early stopping remains MR-only.
    @torch.no_grad()
    def validate():
        ims=[]; txs=[]; gs=[]; rows=[]; vps=[]; wts=[]; masks=[]
        for b in vloader:
            im,tx,vp,wt,mask=encode(b); ims.append(im.cpu()); txs.append(tx.cpu()); vps.append(vp.cpu()); wts.append(wt.cpu()); masks.append(mask.cpu()); gs += b["groups"]
            rows.extend({"pair_id": r["pair_id"], "query": r["query"], "image_id": r["image_id"],
                         "image_path": r["image_path"], "positive_group_id": r["positive_group_id"]}
                        for r in b["rows"])
        tf=torch.cat(txs); imf=torch.cat(ims); global_s=tf@imf.T
        local_s=local_scores(torch.cat(vps).to(device),torch.cat(wts).to(device),torch.cat(masks).to(device)).cpu()
        s=(global_s+a.joint_eta*local_s)/(1+a.joint_eta); n=len(gs); text_ranks=[]; image_ranks=[]
        text_rankings=[]; text_gold=[]; text_results=[]
        for i,g in enumerate(gs):
            pos={j for j,x in enumerate(gs) if x==g}
            order=s[i].argsort(descending=True).tolist()
            text_ranks.append(next(k+1 for k,j in enumerate(order) if j in pos))
            ranked=[rows[j]["image_id"] for j in order]; gold={rows[j]["image_id"] for j in pos}
            text_rankings.append(ranked); text_gold.append(gold)
            text_results.append({"pair_id":rows[i]["pair_id"],"query":rows[i]["query"],"ranked_image_ids":ranked,
                                 "scores":[float(s[i,j]) for j in order]})
            order=s[:,i].argsort(descending=True).tolist()
            image_ranks.append(next(k+1 for k,j in enumerate(order) if j in pos))
        image_results=[]
        for i in range(n):
            order=s[:,i].argsort(descending=True).tolist()
            image_results.append({"image_id":rows[i]["image_id"],"ranked_pair_ids":[rows[j]["pair_id"] for j in order],
                                  "scores":[float(s[j,i]) for j in order]})
        mr=sum(r<=k for ranks in (text_ranks,image_ranks) for r in ranks for k in (1,5,10))/(6*n)
        mp_t=multi_positive_metrics(text_rankings,text_gold)
        mp_i=multi_positive_metrics(
            [[x for x in z["ranked_pair_ids"]] for z in image_results],
            [{rows[j]["pair_id"] for j in range(n) if rows[j]["positive_group_id"]==rows[i]["positive_group_id"]} for i in range(n)])
        mp={k:(mp_t[k]+mp_i[k])/2 for k in ("map@R","R-Precision","R@1")}
        hn=hard_negative_metrics(text_results,image_results,validation_hard,rows)
        global_text_ranks=[]; global_image_ranks=[]
        for i,g in enumerate(gs):
            pos={j for j,x in enumerate(gs) if x==g}
            global_text_ranks.append(next(k+1 for k,j in enumerate(global_s[i].argsort(descending=True).tolist()) if j in pos))
            global_image_ranks.append(next(k+1 for k,j in enumerate(global_s[:,i].argsort(descending=True).tolist()) if j in pos))
        global_mr=sum(r<=k for ranks in (global_text_ranks,global_image_ranks) for r in ranks for k in (1,5,10))/(6*n)
        return {"MR":mr,"global_MR":global_mr,"Multi-positive mAP@R":mp["map@R"],"Multi-positive R-Precision":mp["R-Precision"],
                "Multi-positive R@1":mp["R@1"],"Hard-negative R@1":hn["R@1"],"HN Accuracy":hn["HN_Accuracy"]}
    best=-1.; stale=0; hist=[]; start=time.time()
    for ep in range(1,a.epochs+1):
        model.train(); img_proj.train(); txt_proj.train(); sums={"L_MP":0,"L_LC":0,"L_HN":0,"total":0,"explicit_t2i_queries":0,"explicit_i2t_queries":0}
        for b in loader:
            opt.zero_grad(set_to_none=True); loss,parts=loss_batch(b); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
            for k,v in parts.items(): sums[k]+=v
        val_metrics=validate(); val=val_metrics["global_MR"] if a.early_stop_score=="global" else val_metrics["MR"]
        rec={"epoch":ep,"train":{k:v/len(loader) for k,v in sums.items()},"validation":val_metrics,"validation_MR_proxy":val}; hist.append(rec); (out/"metrics.jsonl").open("a").write(json.dumps(rec)+"\n"); print(json.dumps(rec),flush=True)
        if val>best+1e-4:
            best=val; stale=0
            cfg={"rank":8,"alpha":16.,"dropout":.05,"vision_layers":4,"text_layers":4,"method":f"{a.global_loss}+LC+HN","lambda_lc":a.lambda_lc,"lambda_hn":a.lambda_hn,"margin":a.margin,"tau_local":a.tau_local,"joint_eta":a.joint_eta,"early_stop_score":a.early_stop_score,"local_loss":"multi_positive_supcon","positive_group_weighting":"equal_group","hard_negative_score":a.hn_score,"hard_negative_positive_aggregation":"per_positive_mean"}
            save_lora_adapter(model,out/"checkpoints"/"best",cfg)
            torch.save({"image_projection":img_proj.state_dict(),"text_projection":txt_proj.state_dict(),"config":cfg},out/"checkpoints"/"best"/"local_heads.pt")
        else: stale+=1
        if stale>=a.patience: break
    save_lora_adapter(model,out/"checkpoints"/"last",{"rank":8,"alpha":16.,"dropout":.05,"vision_layers":4,"text_layers":4,"method":f"{a.global_loss}+LC+HN","lambda_lc":a.lambda_lc,"lambda_hn":a.lambda_hn,"margin":a.margin,"tau_local":a.tau_local,"joint_eta":a.joint_eta,"early_stop_score":a.early_stop_score,"local_loss":"multi_positive_supcon","positive_group_weighting":"equal_group","hard_negative_score":"joint","hard_negative_global_gradient":"stopped","hard_negative_positive_aggregation":"per_positive_mean"})
    torch.save({"image_projection":img_proj.state_dict(),"text_projection":txt_proj.state_dict()},out/"checkpoints"/"last"/"local_heads.pt")
    (out/"summary.json").write_text(json.dumps({"status":"completed","epochs_completed":len(hist),"best_validation_MR_proxy":best,"elapsed_seconds":time.time()-start},indent=2))
if __name__=="__main__": main()
