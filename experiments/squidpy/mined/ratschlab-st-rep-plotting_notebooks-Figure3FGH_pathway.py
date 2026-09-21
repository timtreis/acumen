# mined from: https://github.com/ratschlab/st-rep/blob/3246895f76b5d12c65b6dc3828bde98e73929148/plotting_notebooks/Figure3FGH_pathway.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
from aestetik.utils.utils_transcriptomics import preprocess_adata
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import decoupler as dc
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np

# %%
# Plotting options
sc.settings.set_figure_params(dpi=300, frameon=False)
sc.set_figure_params(dpi=300)
sc.set_figure_params(figsize=(10, 10))
plt.rcParams.update({'font.size': 13})

# %%
dataset = "10x_TuPro_v2"
sample = "MACEGEJ-2-2"
adata = sc.read_h5ad(f"../{dataset}/out_benchmark/data/h5ad/{sample}.h5ad")
adata.var_names_make_unique()
adata.layers["log1p"] = np.log1p(adata.X)
adata = preprocess_adata(adata)
adata

# %%
mapping = {2:2, 3:3, 1:4, 4:1}

# %%
training_split = "MAJOFIJ-2-1_MAJOFIJ-2-2_test_MANOFYB-1-1_MANOFYB-1-2_MELIPIT-1-1_MELIPIT-1-2_MACEGEJ-1-1_MACEGEJ-1-2_MAKYGIW-1-1_MAKYGIW-1-2_MAHEFOG-1-1_MAHEFOG-1-2_MIDEKOG-1-1_MIDEKOG-1-2_MACEGEJ-2-1_MACEGEJ-2-2_MAJOFIJ-1-1_MAJOFIJ-1-2"

clusters = pd.read_csv(f"../{dataset}/out_benchmark/{training_split}/AESTETIK_evaluate/clusters/model-MACEGEJ-2-2-best_param.csv")
adata.obs["AESTETIK_cluster"] = pd.Categorical([str(mapping[c]) for c in clusters.best_param.values + 1], ordered=True)
adata.obs["AESTETIK_cluster"]

# %%
adata.obs[["AESTETIK_cluster", "ground_truth"]].value_counts()

# %%
sc.tl.rank_genes_groups(adata, 'AESTETIK_cluster', use_raw=False, layer="log1p", method='wilcoxon', n_genes=50)

# %%
sorted_logfoldchanges_arg = (-pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges']).values).argsort(axis=0)
sorted_logfoldchanges_arg.shape

# %%
marker_genes = pd.DataFrame(np.take_along_axis(pd.DataFrame(adata.uns['rank_genes_groups']['names']).values, sorted_logfoldchanges_arg, axis=0), columns=adata.obs["AESTETIK_cluster"].dtype.categories).head(15)
marker_genes

# %%
#marker_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
marker_genes_dict = marker_genes.to_dict(orient='list')
marker_genes_list = list(set(marker_genes.values.reshape(-1)))
marker_genes.columns

# %%
plt.rcParams.update({'font.size': 11})
sc.pl.matrixplot(adata, marker_genes_dict, groupby='AESTETIK_cluster', 
                 use_raw=False, cmap='bwr',
                 save=f"{sample}_marker_genes.png"
                )

# %%
progeny = dc.get_progeny(organism='human', top=500)
progeny

# %%
dc.run_mlm(
    mat=adata,
    net=progeny,
    source='source',
    target='target',
    weight='weight',
    verbose=True,
    use_raw=False,
    min_n = 5
)

# %%
scaler = StandardScaler()
adata.obsm["mlm_estimate_norm"] = pd.DataFrame(scaler.fit_transform(adata.obsm["mlm_estimate"].T).T, columns=adata.obsm["mlm_estimate"].columns, index=adata.obsm["mlm_estimate"].index)
acts = dc.get_acts(adata, obsm_key='mlm_estimate_norm')
acts

# %%
sc.pl.matrixplot(acts, var_names=acts.var_names, groupby='AESTETIK_cluster', dendrogram=False,
                 colorbar_title='Activity\nz-score', cmap='bwr', 
                 figsize=(10,2), title="Pathway",
                 save=f"{sample}_pathway_activity.png"
                )

# %%
plt.rcParams.update({'font.size': 11})
pad=20
pathway = "MAPK"
bounds = (adata.obsm["spatial"][:,0].min() - pad, 
              adata.obsm["spatial"][:,1].min() - pad,
               adata.obsm["spatial"][:,0].max() + pad,
               adata.obsm["spatial"][:,1].max() + pad)

sq.pl.spatial_scatter(
            acts,
            crop_coord=bounds,
            #title="Tumor distance",
            img_alpha=0.8,
            color=[pathway],
            size=4,
            use_raw=False,
            cmap='bwr',
            ncols=3, wspace=0, dpi=300, frameon=False, figsize=(6,5),
            vcenter=0,
            save=f"{sample}_{pathway}.png"
)

# %%
plt.rcParams.update({'font.size': 11})
gene = "TYRP1"
pad=20
bounds = (adata.obsm["spatial"][:,0].min() - pad, 
              adata.obsm["spatial"][:,1].min() - pad,
               adata.obsm["spatial"][:,0].max() + pad,
               adata.obsm["spatial"][:,1].max() + pad)

sq.pl.spatial_scatter(
            adata,
            crop_coord=bounds,
            #title="Tumor distance",
            img_alpha=0.8,
            color=[gene],
            size=4,
            use_raw=False,
            ncols=3, wspace=0, dpi=300, frameon=False, figsize=(6,5),
            cmap='bwr',
            save=f"{sample}_{gene}.png"
)

# %%

