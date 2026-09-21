# mined from: https://github.com/HelloWorldLTY/spEMO/blob/e86af1e545f2ee7a7ff2e166e25ee35bf6ce5e2f/spatial_domain_iden/analyze spagcn.ipynb
# symbols: squidpy.im.ImageContainer

# %%
import SpaGCN as spg

# Please ensure that you use the SpaGCN under this path, which includes our modification for enhancement.

import numpy as np
from PIL import Image
import requests
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")

# %%
import scanpy as sc
import squidpy as sq

# %%
import pickle

# %%
label_dict = {}
for idx in ['151507',
 '151508',
 '151509',
 '151510',
 '151669',
 '151670',
 '151671',
 '151672',
 '151673',
 '151674',
 '151675',
 '151676']:
    adata = sc.read_h5ad(f"../dplfc_data/{idx}_adata.h5ad")
    img = sq.im.ImageContainer(f"../dplfc_data/{idx}_full_image.tif")

    x_array = adata.obs["array_row"].tolist()
    y_array = adata.obs["array_col"].tolist()
    x_pixel = (adata.obsm["spatial"][:, 0]).tolist()
    y_pixel = adata.obsm["spatial"][:, 1].tolist()

    img = np.array(img.data.image)

    img = img[:,:,0,:]

    # Calculate adjacent matrix
    adj = spg.calculate_adj_matrix(
        x=x_pixel,
        y=y_pixel,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        image=img,
        beta=55,
        alpha=1,
        histology=True,
        image_feature = f"../GPFM/visium_{idx}_allspot_gpfm_112.pkl"
    )

    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=3)

    # find mitochondrial (MT) genes
    adata.var["MT_gene"] = [gene.startswith("MT-") for gene in adata.var_names]
    # remove MT genes (keeping their counts in the object)
    adata.obsm["MT"] = adata[:, adata.var["MT_gene"].values].X.toarray()
    adata = adata[:, ~adata.var["MT_gene"].values].copy()

    # Normalize and take log for UMI
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    p=0.5 
    #Find the l value given p
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    n_clusters=7
    #Set seed
    r_seed=t_seed=n_seed=100
    #Seaech for suitable resolution
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

    
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    
    model = spg.SpaGCN()
    model.set_l(l)

    model.train(adata, adj, res=res)

    y_pred, prob = model.predict()

    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=y_pred, dis=adj_2d, shape="hexagon")
    label_dict[idx] = refined_pred

# %%

import pickle
with open("./visium_spagcn_includegpfm_update_adj_multi.pkl", "wb") as f:
    pickle.dump(label_dict, f)
