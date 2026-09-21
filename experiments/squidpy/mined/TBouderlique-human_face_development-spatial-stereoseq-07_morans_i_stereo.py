# mined from: https://github.com/TBouderlique/human_face_development/blob/ea1a5fcc81495d889c3d4252f62d658ce53df0b2/spatial/stereoseq/07_morans_i_stereo.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm import tqdm
import squidpy as sq
from scipy.stats import spearmanr
import os

sc.settings.verbosity = 1
sns.set(font_scale=1)
sc.settings.set_figure_params(dpi=150)
sns.set_style("ticks")

from matplotlib import cm
from matplotlib.colors import ListedColormap

cm_color = cm.get_cmap("Reds", 128)
cm_grey = cm.get_cmap("Greys", 128)

Reds = ListedColormap(np.vstack((
    cm_grey(np.linspace(0.2, 0.2, 1)),
    cm_color(np.linspace(0.1, 1, 128)),
)))

# %%
# Removed B2, due to no cell segmentation present, it is processed seperately below.  
my_path ="/home/felix/projects/facial/felix/data/reprocessed_data/"
datasets = ["A02989E1", "A02993E1",  "A02994D6",  "A02994E6", "C01939A4",  "C01939A5",  "C01939A6"]

# %%
adatas_cells = {}
adatas = {}

for dataset in datasets:

    adatas_cells[dataset] = sc.read_h5ad(f"/home/felix/projects/facial/felix/data/reprocessed_data/{dataset}/processed/cells.h5ad")
    adatas[dataset] = sc.read_h5ad(f"/home/felix/projects/facial/felix/data/reprocessed_data/{dataset}/processed/for_morans_50.h5ad")
        
    
    sq.gr.spatial_neighbors(adatas[dataset], spatial_key="X_spatial", n_neighs=6)
    sq.gr.spatial_autocorr(adatas[dataset], mode="moran", genes=adatas[dataset].var_names)

    adatas_cells[dataset].obsm["X_pca"] = adatas_cells[dataset].obsm["X_pca"][:, 1:]
    sc.pp.neighbors(adatas_cells[dataset], n_neighbors=20)
    sq.gr.spatial_autocorr(adatas_cells[dataset], mode="moran", genes=adatas_cells[dataset].var_names,
                           connectivity_key="connectivities")

    # For visualization
    sc.tl.umap(adatas_cells[dataset])
    sc.tl.leiden(adatas_cells[dataset])

# %%
dfs = {}

min_cells = 50
n_genes = 20
signatures = {}

for dataset in datasets:
    os.makedirs(f"spatial_signature_genes/{dataset}_{n_genes}_genes", exist_ok=True)
    dfs[dataset] = pd.DataFrame({
        "GEX": adatas_cells[dataset].uns["moranI"].loc[adatas[dataset].uns["moranI"].index].I,
        "Spatial": adatas[dataset].uns["moranI"].I,
    }, index=adatas[dataset].uns["moranI"].index)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=np.sqrt(dfs[dataset].GEX),
        y=dfs[dataset].Spatial - np.sqrt(dfs[dataset].GEX),
        edgecolor=None,
        s=5,
        color="grey",
        ax=ax,
    )

    ax.grid(alpha=0.3)
    ax.set_xlabel("Moran's I on GEX PCA graph")
    ax.set_ylabel("Moran's I on spatial graph")
    
    ax.set_title(dataset)
    
    whitelist_genes = adatas_cells[dataset].var_names[(adatas_cells[dataset].X > 0).sum(axis=0).A[0] > min_cells]
    genes = (dfs[dataset].loc[whitelist_genes].Spatial - np.sqrt(dfs[dataset].loc[whitelist_genes].GEX)).fillna(0).sort_values().index[-n_genes:]
    ranked_list = (dfs[dataset].loc[whitelist_genes].Spatial - np.sqrt(dfs[dataset].loc[whitelist_genes].GEX)).fillna(0).sort_values()
    ranked_list[-100:].to_csv(f"spatial_signature_genes/{dataset}_{n_genes}_genes/ranked_list.tsv", sep='\t') # take top 100 and save to csv
    signatures[dataset] = list(genes).copy()[::-1]
    

# %%
dfs = {}

min_cells = 50
n_genes = 20

for dataset in datasets:
    os.makedirs(f"spatial_signature_genes/{dataset}_{n_genes}_genes", exist_ok=True) 
    
    dfs[dataset] = pd.DataFrame({
        "GEX": adatas_cells[dataset].uns["moranI"].loc[adatas[dataset].uns["moranI"].index].I,
        "Spatial": adatas[dataset].uns["moranI"].I,
    }, index=adatas[dataset].uns["moranI"].index)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=dfs[dataset].GEX,
        y=dfs[dataset].Spatial,
        edgecolor=None,
        s=5,
        color="grey",
        ax=ax,
    )
    
    sns.scatterplot(
        x=dfs[dataset].loc[signatures[dataset]].GEX,
        y=dfs[dataset].loc[signatures[dataset]].Spatial,
        edgecolor=None,
        s=10,
        color="red",
        ax=ax,
    )

    ax.grid(alpha=0.3)
    ax.set_xlabel("Moran's I on GEX PCA graph")
    ax.set_ylabel("Moran's I on spatial graph")
    
    ax.set_title(dataset)
    
    plt.savefig(f"spatial_signature_genes/{dataset}_{n_genes}_genes/signature_genes.svg")

# %%
for dataset in datasets:
    os.makedirs(f"spatial_codes/{dataset}", exist_ok=True) 
    for gene in signatures[dataset]:
        fig, axes = plt.subplots(figsize=(8, 4), ncols=2)

        sc.pl.embedding(adatas_cells[dataset], basis="X_spatial", color=gene, frameon=False, s=5, cmap=Reds, ax=axes[0], show=False)
        sc.pl.umap(adatas_cells[dataset], color=gene, cmap=Reds, s=5, frameon=False, ax=axes[1], show=False)
        
        plt.savefig(f"spatial_codes/{dataset}/{gene}.pdf")

# %%


# %%
# One does not have the cells.h5ad  Do it seperately.
my_path ="/home/felix/projects/facial/felix/data/reprocessed_data/"
dataset = "C01939B2"

# %%

adatas_cells = sc.read_h5ad(f"/home/felix/projects/facial/felix/data/reprocessed_data/{dataset}/processed/for_morans_30.h5ad")
adata = sc.read_h5ad(f"/home/felix/projects/facial/felix/data/reprocessed_data/{dataset}/processed/for_morans_50.h5ad")

  
sq.gr.spatial_neighbors(adata, spatial_key="X_spatial", n_neighs=6)
sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)

adatas_cells.obsm["X_pca"] = adatas_cells.obsm["X_pca"][:, 1:]
sc.pp.neighbors(adatas_cells, n_neighbors=20)
sq.gr.spatial_autocorr(adatas_cells, mode="moran", genes=adatas_cells.var_names,
                           connectivity_key="connectivities")

# For visualization
sc.tl.umap(adatas_cells)
sc.tl.leiden(adatas_cells)

# %%
dfs = {}

min_cells = 50
n_genes = 20
signatures = {}
datasets = [dataset]
for dataset in datasets:
    os.makedirs(f"spatial_signature_genes/{dataset}_{n_genes}_genes", exist_ok=True)
    dfs[dataset] = pd.DataFrame({
        "GEX": adatas_cells.uns["moranI"].loc[adata.uns["moranI"].index].I,
        "Spatial": adata.uns["moranI"].I,
    }, index=adata.uns["moranI"].index)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=np.sqrt(dfs[dataset].GEX),
        y=dfs[dataset].Spatial - np.sqrt(dfs[dataset].GEX),
        edgecolor=None,
        s=5,
        color="grey",
        ax=ax,
    )

    ax.grid(alpha=0.3)
    ax.set_xlabel("Moran's I on GEX PCA graph")
    ax.set_ylabel("Moran's I on spatial graph")
    
    ax.set_title(dataset)
    
    whitelist_genes = adatas_cells.var_names[(adatas_cells.X > 0).sum(axis=0).A[0] > min_cells]
    genes = (dfs[dataset].loc[whitelist_genes].Spatial - np.sqrt(dfs[dataset].loc[whitelist_genes].GEX)).fillna(0).sort_values().index[-n_genes:]
    ranked_list = (dfs[dataset].loc[whitelist_genes].Spatial - np.sqrt(dfs[dataset].loc[whitelist_genes].GEX)).fillna(0).sort_values()
    ranked_list[-100:].to_csv(f"spatial_signature_genes/{dataset}_{n_genes}_genes/ranked_list.tsv", sep='\t') # take top 100 and save to csv
    signatures[dataset] = list(genes).copy()[::-1]
    

# %%
dfs = {}

min_cells = 50
n_genes = 20

for dataset in datasets:
    os.makedirs(f"spatial_signature_genes/{dataset}_{n_genes}_genes", exist_ok=True) 
    
    dfs[dataset] = pd.DataFrame({
        "GEX": adatas_cells.uns["moranI"].loc[adata.uns["moranI"].index].I,
        "Spatial": adata.uns["moranI"].I,
    }, index=adata.uns["moranI"].index)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=dfs[dataset].GEX,
        y=dfs[dataset].Spatial,
        edgecolor=None,
        s=5,
        color="grey",
        ax=ax,
    )
    
    sns.scatterplot(
        x=dfs[dataset].loc[signatures[dataset]].GEX,
        y=dfs[dataset].loc[signatures[dataset]].Spatial,
        edgecolor=None,
        s=10,
        color="red",
        ax=ax,
    )

    ax.grid(alpha=0.3)
    ax.set_xlabel("Moran's I on GEX PCA graph")
    ax.set_ylabel("Moran's I on spatial graph")
    
    ax.set_title(dataset)
    
    #plt.savefig(f"spatial_signature_genes/{dataset}_{n_genes}_genes/signature_genes.pdf")

# %%
for dataset in datasets:
    os.makedirs(f"spatial_codes/{dataset}", exist_ok=True) 
    for gene in signatures[dataset]:
        fig, axes = plt.subplots(figsize=(8, 4), ncols=2)

        sc.pl.embedding(adatas_cells, basis="X_spatial", color=gene, frameon=False, s=5, cmap=Reds, ax=axes[0], show=False)
        sc.pl.umap(adatas_cells, color=gene, cmap=Reds, s=5, frameon=False, ax=axes[1], show=False)
        
        #plt.savefig(f"spatial_codes/{dataset}/{gene}.pdf")

# %%


# %%

