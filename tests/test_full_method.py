import torch
from mmmrag.full_method import (
    hard_negative_ranking_loss, local_contrastive_loss,
    multi_positive_contrastive_loss, relation_aware_batches, relation_maps
)

def test_multi_positive_loss_rewards_probability_mass_on_all_positives():
    mask=torch.tensor([[1,1,0],[1,1,0],[0,0,1]],dtype=torch.bool)
    good=torch.tensor([[3.,3.,0.],[3.,3.,0.],[0.,0.,3.]])
    bad=torch.zeros_like(good)
    assert multi_positive_contrastive_loss(good,mask)<multi_positive_contrastive_loss(bad,mask)

def test_hard_negative_prefers_explicit_then_online():
    scores=torch.tensor([[.8,.7,.1],[.2,.8,.6],[.5,.1,.8]])
    pos=torch.eye(3,dtype=torch.bool)
    explicit=torch.zeros_like(pos); explicit[0,2]=True
    loss,stats=hard_negative_ranking_loss(scores,pos,explicit,explicit.t(),.1)
    assert torch.isfinite(loss)
    assert stats=={"explicit_t2i_queries":1,"explicit_i2t_queries":1}

def test_local_loss_treats_off_diagonal_group_members_as_positives():
    mask=torch.tensor([[1,1,0],[1,1,0],[0,0,1]],dtype=torch.bool)
    good=torch.tensor([[.8,.8,.1],[.8,.8,.1],[.1,.1,.8]])
    bad=torch.tensor([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
    assert local_contrastive_loss(good,mask,.1)<local_contrastive_loss(bad,mask,.1)

def test_hard_negative_loss_covers_every_positive():
    scores=torch.tensor([[.9,.2,.8],[.9,.9,.1],[.1,.1,.9]])
    pos=torch.tensor([[1,1,0],[1,1,0],[0,0,1]],dtype=torch.bool)
    explicit=torch.zeros_like(pos); explicit[0,2]=True
    loss,_=hard_negative_ranking_loss(scores,pos,explicit,explicit.t(),.1)
    assert loss>0

def test_relation_maps_are_bidirectional():
    t2i,i2t=relation_maps([{"description":"a","negative_image_paths":["x/a.jpg","y/b.jpg"]}])
    assert t2i["a"]=={"a.jpg","b.jpg"}
    assert i2t["a.jpg"]=={"a"}

def test_relation_batches_do_not_leave_singleton():
    rows=[{"positive_group_id":str(i),"image_path":f"{i}.jpg","query":str(i)} for i in range(9)]
    batches=relation_aware_batches(rows,{},8,42)
    assert sorted(i for batch in batches for i in batch)==list(range(9))
    assert min(map(len,batches))>=2
