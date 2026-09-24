# Method

## Problem Formulation

给定图文检索数据集

\[
\mathcal{D}=\{(I_i,T_i,g_i)\}_{i=1}^{N},
\]

其中 $I_i$、$T_i$ 和 $g_i$ 分别表示图像、文本描述和正样本组标识。具有相同 $g_i$ 的图像或文本属于同一语义正样本组，因此一个查询可以对应多个有效正样本。本文采用 CN-CLIP ViT-H/14 作为视觉骨架，并采用 RoBERTa-wwm-ext-large 作为文本骨架。

对一个 batch

\[
\mathcal{B}=\{(I_i,T_i,g_i)\}_{i=1}^{B},
\]

视觉编码器 $f_I(\cdot)$ 和文本编码器 $f_T(\cdot)$ 输出全局特征

\[
\mathbf{v}_i=f_I(I_i),\qquad
\mathbf{t}_i=f_T(T_i).
\]

经 $\ell_2$ 归一化后，全局图文相似度定义为

\[
S^{g}(I_i,T_j)=
\frac{\mathbf{v}_i^{\top}\mathbf{t}_j}
{\|\mathbf{v}_i\|_2\|\mathbf{t}_j\|_2}.
\]

文本 $T_i$ 的正图像集合和图像 $I_i$ 的正文本集合分别为

\[
\mathcal{P}^{I}_i=\{j\mid g_j=g_i\},\qquad
\mathcal{P}^{T}_i=\{j\mid g_j=g_i\}.
\]

该定义在正样本组大小为 1 时退化为普通的一对一图文匹配。

## Local Discriminative Matching

全局特征主要描述整体语义，难以充分区分具有相似场景、实体或局部结构的样本。为获得细粒度匹配信号，本文保留 ViT 的 patch 特征和文本编码器的 token 特征，并将两种特征投影到同一 $d$ 维空间：

\[
\hat{\mathbf{v}}_i^m=\operatorname{Norm}(h_I(\mathbf{v}_i^m)),\qquad
\hat{\mathbf{w}}_j^n=\operatorname{Norm}(h_T(\mathbf{w}_j^n)).
\]

其中 $h_I$ 和 $h_T$ 为可学习线性投影层。视觉 patch 与文本 token 的局部相似度为

\[
A_{mn}^{(i,j)}=(\hat{\mathbf{v}}_i^m)^{\top}\hat{\mathbf{w}}_j^n.
\]

在 Text-to-Image 方向，每个文本 token 与最相似的图像 patch 进行匹配；在 Image-to-Text 方向，每个图像 patch 与最相似的文本 token 进行匹配。因此，双向局部匹配分数为

\[
S^{l}_{T\rightarrow I}(I_i,T_j)=
\frac{1}{L_j}\sum_{n=1}^{L_j}\max_m A_{mn}^{(i,j)},
\]

\[
S^{l}_{I\rightarrow T}(I_i,T_j)=
\frac{1}{M}\sum_{m=1}^{M}\max_n A_{mn}^{(i,j)},
\]

\[
S^{l}(I_i,T_j)=
\frac{1}{2}\left(S^{l}_{T\rightarrow I}(I_i,T_j)+
S^{l}_{I\rightarrow T}(I_i,T_j)\right).
\]

为使局部监督与正样本组定义一致，局部匹配使用多正样本 supervised contrastive 损失。对得分矩阵 $Z$ 和正样本掩码 $M$，单个查询的损失为

\[
\ell_i(Z,M)=-\frac{1}{|\mathcal{P}_i|}
\sum_{j\in\mathcal{P}_i}
\log\frac{\exp(Z_{ij}/\tau_l)}
{\sum_{k=1}^{B}\exp(Z_{ik}/\tau_l)}.
\]

为避免正样本数量较大的组占据过高权重，令 $w_i=1/|\mathcal{P}_i|$，则

\[
\mathcal{L}_{\mathrm{LC}}^{T\rightarrow I}=
\frac{\sum_i w_i\ell_i(S^l,M)}{\sum_i w_i},
\qquad
\mathcal{L}_{\mathrm{LC}}^{I\rightarrow T}=
\frac{\sum_i w_i\ell_i((S^l)^\top,M^\top)}{\sum_i w_i}.
\]

最终局部对比损失为

\[
\mathcal{L}_{\mathrm{LC}}=\frac{1}{2}
\left(\mathcal{L}_{\mathrm{LC}}^{T\rightarrow I}+
\mathcal{L}_{\mathrm{LC}}^{I\rightarrow T}\right).
\]

## Multi-positive Global Alignment

传统 CLIP 损失只将配对样本视为正样本，会把同一正样本组中的其他有效样本误作负样本。本文在全局特征空间中对整组正样本进行 supervised contrastive 学习。Text-to-Image 方向的单查询损失为

$\ell_i^{T\rightarrow I}=-\frac{1}{|\mathcal{P}^{I}_i|}
\sum_{j\in\mathcal{P}^{I}_i}
\log\frac{\exp(S^g(I_j,T_i)/\tau_g)}
{\sum_{k=1}^{B}\exp(S^g(I_k,T_i)/\tau_g)}$


Image-to-Text 方向对称地定义为

$
\ell_i^{I\rightarrow T}=-\frac{1}{|\mathcal{P}^{T}_i|}
\sum_{j\in\mathcal{P}^{T}_i}
\log\frac{\exp(S^g(I_i,T_j)/\tau_g)}
{\sum_{k=1}^{B}\exp(S^g(I_i,T_k)/\tau_g)}
$

使用 $w_i=1/|\mathcal{P}_i|$ 进行正样本组平衡后，得到

$
\mathcal{L}_{\mathrm{MP}}^{T\rightarrow I}=
\frac{\sum_i w_i\ell_i^{T\rightarrow I}}{\sum_i w_i},
\qquad
\mathcal{L}_{\mathrm{MP}}^{I\rightarrow T}=
\frac{\sum_i w_i\ell_i^{I\rightarrow T}}{\sum_i w_i}
$

最终多正样本全局对齐损失为

\[
\mathcal{L}_{\mathrm{MP}}=\frac{1}{2}
\left(\mathcal{L}_{\mathrm{MP}}^{T\rightarrow I}+
\mathcal{L}_{\mathrm{MP}}^{I\rightarrow T}\right).
\]

因此，训练目标中不再额外加入普通单正样本全局损失，以避免其重新将其他有效正样本视为负样本。

## Hard Negative Discrimination

难负样本与查询具有较高语义或视觉相似度，但不属于同一目标。为使训练约束与最终检索排序一致，本文首先定义联合匹配分数

\[
S^{\mathrm{joint}}(I_i,T_j)=
\frac{S^g(I_i,T_j)+\eta S^l(I_i,T_j)}{1+\eta}.
\]

难负样本损失使用联合得分，但停止全局分支的梯度：

\[
\widetilde{S}(I_i,T_j)=
\frac{\operatorname{sg}(S^g(I_i,T_j))+\eta S^l(I_i,T_j)}{1+\eta},
\]

其中 $\operatorname{sg}(\cdot)$ 表示 stop-gradient。该设计使 $\mathcal{L}_{\mathrm{HN}}$ 主要优化局部判别能力，同时避免与全局多正样本监督发生梯度竞争。

对于文本 $T_i$，若存在显式标注的难负图像集合 $\mathcal{H}^{I}_i$，则选择其中联合得分最高者；否则，从 batch 中所有非正图像中动态选择最高分样本：

\[
I_i^-=
\begin{cases}
\displaystyle\arg\max_{j\in\mathcal{H}^{I}_i}\widetilde{S}(I_j,T_i),
&\mathcal{H}^{I}_i\neq\varnothing,\\[6pt]
\displaystyle\arg\max_{j\notin\mathcal{P}^{I}_i}\widetilde{S}(I_j,T_i),
&\mathcal{H}^{I}_i=\varnothing.
\end{cases}
\]

Text-to-Image 难负样本损失对同一正样本组中的每个正图像分别施加排序约束，并进行逐正样本平均：

\[
\mathcal{L}_{\mathrm{HN}}^{T\rightarrow I}=
\frac{1}{B}\sum_{i=1}^{B}\frac{1}{|\mathcal{P}^{I}_i|}
\sum_{p\in\mathcal{P}^{I}_i}
\left[m-\widetilde{S}(I_p,T_i)+\widetilde{S}(I_i^-,T_i)\right]_+.
\]

其中 $[x]_+=\max(0,x)$，$m$ 为排序 margin。Image-to-Text 方向对称地定义难负文本 $T_i^-$，并得到

\[
\mathcal{L}_{\mathrm{HN}}^{I\rightarrow T}=
\frac{1}{B}\sum_{i=1}^{B}\frac{1}{|\mathcal{P}^{T}_i|}
\sum_{p\in\mathcal{P}^{T}_i}
\left[m-\widetilde{S}(I_i,T_p)+\widetilde{S}(I_i,T_i^-)\right]_+.
\]

最终双向难负样本损失为

\[
\mathcal{L}_{\mathrm{HN}}=\frac{1}{2}
\left(\mathcal{L}_{\mathrm{HN}}^{T\rightarrow I}+
\mathcal{L}_{\mathrm{HN}}^{I\rightarrow T}\right).
\]

## Overall Objective

综合全局多正样本对齐、局部判别匹配和难负样本排序，最终优化目标为

\[
\boxed{
\mathcal{L}=\mathcal{L}_{\mathrm{MP}}+
\lambda_{\mathrm{LC}}\mathcal{L}_{\mathrm{LC}}+
\lambda_{\mathrm{HN}}\mathcal{L}_{\mathrm{HN}}}
\]

本文实验采用 $\lambda_{\mathrm{LC}}=0.1$、$\lambda_{\mathrm{HN}}=0.25$、$\eta=0.1$、$m=0.1$。训练阶段使用 $\widetilde{S}$ 计算难负样本排序损失，验证和测试阶段使用完整联合得分 $S^{\mathrm{joint}}$ 进行早停和检索排序。
