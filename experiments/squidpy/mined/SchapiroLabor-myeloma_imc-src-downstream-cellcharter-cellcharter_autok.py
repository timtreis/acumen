# mined from: https://github.com/SchapiroLabor/myeloma_imc/blob/866e378b0ee212787f8a7c0f4a3cc916306c2003/src/downstream/cellcharter/cellcharter_autok.ipynb
# symbols: squidpy.gr.spatial_neighbors, squidpy.pl.spatial_scatter

# %%
import squidpy as sq
import cellcharter as cc
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scarches as sca
from lightning.pytorch import seed_everything
seed_everything(42)
sc._settings.ScanpyConfig.n_jobs = -1

# %%
adata = ad.read_h5ad("/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/results/standard/adatas/cells_annotated_pp_osteocytes_cleaned.h5ad")
adata.X =  adata.layers['arcsinh']
sc.pp.scale(adata)
adata.X = adata.X.astype(np.float32).copy()
condition_key = 'patient_ID'
cell_type_key = 'Phenotype3'
conditions = adata.obs[condition_key].unique().tolist()

# %%
#Code for the initial trvae model
trvae_epochs = 500
surgery_epochs = 500

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
trvae = cc.tl.TRVAE(
    adata=adata,
    condition_key=condition_key,
    conditions=conditions,
    recon_loss='mse',
    use_mmd=False,
)
trvae.train(
    n_epochs=trvae_epochs,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs,
    enable_progress_bar=True,
    
)
trvae.save('/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/github/myeloma_standal/src/downstream/advanced_neighborhood/cellcharter/trvae_scaled', overwrite=True)

# %%
model = cc.tl.TRVAE.load(
    '/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/github/myeloma_standal/src/downstream/advanced_neighborhood/cellcharter/trvae_nonscaled', 
    adata, 
    map_location='cpu'
)

# %%
adata.obsm['X_trvae']= model.get_latent(adata.X, adata.obs['patient_ID'])

# %%
adata.obsm['spatial'] = np.array(adata.obs[['X_centroid', 'Y_centroid']])
adata.uns['spatial'] = {
    'X': adata.obs['X_centroid'].values,
    'Y': adata.obs['Y_centroid'].values
}
unique_image_ids = adata.obs['image_ID'].unique()
adata.uns['spatial'] = {image_id: {} for image_id in unique_image_ids}

# %%
sq.gr.spatial_neighbors(adata, library_key='image_ID', coord_type='generic', delaunay=True)

# %%
sq.pl.spatial_scatter(
    adata, 
    color='image_ID', 
    library_key='image_ID', 
    img=None, 
    title=['TS-373_IMC77_B_001.csv'],
    size=6,
    connectivity_key='spatial_connectivities',
    edges_width=0.3,
    legend_loc=None,
    library_id=['TS-373_IMC77_B_001.csv']
    )

# %%
sq.pl.spatial_scatter(
    adata, 
    color='image_ID', 
    library_key='image_ID', 
    img=None, 
    title=['TS-373_IMC26_B_001.csv'],
    size=6,
    connectivity_key='spatial_connectivities',
    edges_width=0.3,
    legend_loc=None,
    library_id=['TS-373_IMC26_B_001.csv']
    )

# %%
cc.gr.remove_long_links(adata, distance_percentile=95)

# %%
sq.pl.spatial_scatter(
    adata, 
    color='image_ID', 
    library_key='image_ID', 
    img=None, 
    title=['TS-373_IMC77_B_001.csv'],
    size=6,
    connectivity_key='spatial_connectivities',
    edges_width=0.3,
    legend_loc=None,
    library_id=['TS-373_IMC77_B_001.csv']
    )

# %%
sq.pl.spatial_scatter(
    adata, 
    color='image_ID', 
    library_key='image_ID', 
    img=None, 
    title=['TS-373_IMC26_B_001.csv'],
    size=6,
    connectivity_key='spatial_connectivities',
    edges_width=0.3,
    legend_loc=None,
    library_id=['TS-373_IMC26_B_001.csv']
    )

# %%
cc.gr.aggregate_neighbors(adata, n_layers=3)

# %%
model_params = {
        'random_state': 42,
        'trainer_params': {
            'accelerator':'cpu',
            'enable_progress_bar': False
        },
    }

# %%
autok = cc.tl.ClusterAutoK(n_clusters=(4,20), model_class=cc.tl.GaussianMixture, 
                           model_params=model_params, 
                           max_runs=5)

# %%
autok.fit(adata, use_rep='X_cellcharter')

# %%
ax = cc.pl.autok_stability(autok, return_ax=True)
ax.figure.set_size_inches(12, 6)  

# %%
autok.best_k

# %%
autok.save("/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/results/downstream/neighborhoods/cellcharter_manual/autok_dmr_scale_layer3")
