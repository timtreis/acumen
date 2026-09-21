# mined from: https://github.com/HelloWorldLTY/spEMO/blob/e86af1e545f2ee7a7ff2e166e25ee35bf6ce5e2f/demo/demo.ipynb
# symbols: squidpy.datasets.visium_fluo_adata_crop, squidpy.datasets.visium_fluo_image_crop

# %%
import scanpy as sc
import squidpy as sq
import pickle

import sklearn.metrics

import scib_metrics

import numpy as np

# %%
adata_st = sq.datasets.visium_fluo_adata_crop()
img = sq.datasets.visium_fluo_image_crop()

# %%
import torch
emb_data = torch.load("../GPFM/visium_fluo_image_allspot_gpfm.pkl")
emb_data = np.array(emb_data)
adata_st.obsm['X_emb'] = emb_data

# %%
adata_st.obsm['X_emb']

# %%
# evaluation
def compute_cluter(adata, emb_name = 'X_pca'):
    sc.pp.neighbors(adata, use_rep=emb_name)
    nmi = []
    ari = []
    asw = []
    for i in np.linspace(0,2,21)[1:]:
        sc.tl.leiden(adata, resolution = i)
        nmi.append(sklearn.metrics.normalized_mutual_info_score(adata.obs.leiden, adata.obs.cluster))
        ari.append(sklearn.metrics.adjusted_rand_score(adata.obs.leiden, adata.obs.cluster))
    asw.append(scib_metrics.silhouette_label(adata.obsm[emb_name], adata.obs.cluster))
    
    print(max(nmi))
    print(max(ari))
    print(max(asw))
    return max(nmi), max(ari), max(asw)

# %%
np.mean(compute_cluter(adata_st))

# %%
sc.pp.neighbors(adata_st, use_rep='X_emb')
sc.tl.umap(adata_st, random_state=0)

# %%
sc.pl.umap(adata_st, color='cluster')

# %%

