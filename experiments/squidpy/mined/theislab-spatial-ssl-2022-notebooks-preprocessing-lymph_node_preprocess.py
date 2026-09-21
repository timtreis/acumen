# mined from: https://github.com/theislab/spatial-ssl-2022/blob/d29e600f3438103939256b2c72864aa21a871270/notebooks/preprocessing/lymph_node_preprocess.ipynb
# symbols: squidpy.datasets.visium, squidpy.im.ImageContainer

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
import torchvision.transforms.functional as fn
from torchvision.utils import save_image

from sslbio.paths import DATA_DIR
from joblib import Parallel, delayed

# %%
experiment_id = "V1_Human_Lymph_Node"
adata = sq.datasets.visium(experiment_id, include_hires_tiff=True)

# %%
sq.im.ImageContainer()

# %%
from pathlib import Path

image_path = Path("data/V1_Human_Lymph_Node/image.tif")

# %%
image = sq.im.ImageContainer(image_path)

# %%
technology = "visium"
DATA_DIR = DATA_DIR / technology 

# %%
from tqdm.auto import tqdm

spot_imgs = []
spot_obs = []

spot_scale = 10
for spot_img, obs in tqdm(image.generate_spot_crops(adata=adata, return_obs=True, spot_scale=spot_scale)): 
    spot_img = spot_img['image'].squeeze().values
    spot_img = fn.resize(torch.Tensor(spot_img).permute(2, 0, 1), 224)
    spot_imgs.append(spot_img)
    spot_obs.append(obs)

# %%
len(spot_imgs)

# %%
spot_imgs = torch.stack(spot_imgs)
spot_imgs.size()

# %%
fp =  DATA_DIR / f"{experiment_id}_{spot_scale}spots.pt"
torch.save(spot_imgs, fp)

# %%
fp = DATA_DIR / f"{experiment_id}_{spot_scale}spots.txt"

with open(fp, 'w') as output:
    for row in spot_obs:
        output.write(str(row) + '\n')

# %%
plt.imshow(spot_img.permute(1, 2, 0)/255)

# %%
adata

# %%
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.distplot(
    adata.obs["total_counts"],
    kde=False,
    ax=axs[0],
)
sns.distplot(
    adata.obs["total_counts"][adata.obs["total_counts"] < 10000],
    kde=False,
    bins=40,
    ax=axs[1],
)
sns.distplot(
    adata.obs["n_genes_by_counts"],
    kde=False,
    bins=60,
    ax=axs[2],
)
sns.distplot(
    adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000],
    kde=False,
    bins=60,
    ax=axs[3],
)

# %%
# sc.pp.filter_cells(adata, min_counts=1000)
sc.pp.filter_genes(adata, min_cells=10)

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# %%
adata_path = DATA_DIR /"visium"/"V1_Human_Lymph_Node.h5ad"

# %%
adata.write(adata_path)

# %%
# obs_path = DATA_DIR /"visium"/"V1_Human_Lymph_Node_spots.txt"

# %%
# pd.Series(adata.obs_names).to_csv(obs_path, header=False, index=False)

# %%

