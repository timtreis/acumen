# mined from: https://github.com/zenglab-pku/FGT-HD/blob/6646b2e38020eeed2ea0650cccd22596baafc4f7/script/3.Clustering/cellcharter_clustering.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import anndata as ad
from collections import defaultdict
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
import scvi
import cellcharter as cc
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc
import warnings
warnings.filterwarnings("ignore")
from pandas.api.types import CategoricalDtype
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

scvi.settings.seed = 12345

# %%
## parameters

n_latent = 10
max_epochs = 5
n_rings = 1
n_neighs = 8
nhood_layers = 3
best_fit = True
min_clusters = 2
max_clusters = 15
set_clusters = 10

# %%
sample_list = pd.read_csv("HD-OV 100.csv")
sample_list = sample_list[sample_list["type"]=="HGSOC"]
path = "../out"
figure_output = "../figures"
output_path = f'{figure_output}/clusters'
os.makedirs(output_path, exist_ok=True)

cell_colors = {
    "NK": "#CDCE6B",
    "T": "#FFFF99",
    "B": "#00CC99",
    "Neutrophil": "#3366CC",
    "Macrophage": "#990066",
    "DC": "#FF9999",
    "Mast": "#6699FF",
    "Endothelial": "#FF3366",
    "Fibroblast": "#008080",
    "Epithelial": "#0D5886",
    "Others": "lightgray"
}

# %%

adata = sc.read_h5ad(f"{path}/integrated_adata.h5ad")
adata.obs['in_tissue'] = adata.obs['in_tissue'].astype(float)
adata.obs['array_row'] = adata.obs['array_row'].astype(float)
adata.obs['array_col'] = adata.obs['array_col'].astype(float)
adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
adata.X = adata.layers['counts']

scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key='sample')
model = scvi.model.SCVI(adata, n_latent=n_latent)
model.train(max_epochs=max_epochs, early_stopping=True, enable_progress_bar=True)
adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)
sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=n_neighs)
cc.gr.aggregate_neighbors(adata, n_layers=nhood_layers, use_rep='X_scVI', out_key='X_cellcharter')

if best_fit:
    autok = cc.tl.ClusterAutoK(
        n_clusters=(min_clusters,max_clusters),
        max_runs=5,
        model_params=dict(
            random_state=12345,
            trainer_params=dict(accelerator='gpu', devices=1)
        )
    )
    autok.fit(adata, use_rep='X_cellcharter')
    cc.pl.autok_stability(autok)

    plt.savefig(f'{output_path}/spatial_cluster.pdf', format="pdf", bbox_inches="tight")
    plt.close()
    
else:
    autok = cc.tl.Cluster(
        int(set_clusters),
        random_state=12345,
        trainer_params=dict(accelerator='gpu', devices=1)
    )
    autok.fit(adata, use_rep='X_cellcharter')

adata.obs['cluster_cellcharter'] = autok.predict(adata, use_rep='X_cellcharter')
adata.write_h5ad(f"clustered_adata_8um.h5ad")

cc.pl.proportion(
    adata,
    group_key='cluster_cellcharter',
    label_key=f'annotations',
    palette=cell_colors,
    save=f"{output_path}/niche-cell_type_proportion.pdf"
)
plt.close()

# %%


# %%

