# mined from: https://github.com/luciakostialova/thesis/blob/e2d5c33d5eab8ab973d9e349d4c40961557098b6/DE_genes.ipynb
# symbols: squidpy.pl.spatial_scatter, squidpy.read.visium

# %%
import scanpy as sc
import squidpy as sq
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from scipy import sparse
import json

# %%
adata = ad.read_h5ad("BRCA_preprocessed.h5ad", backed=None)  # load the preprocessed data
library_id='Visium_Human_Breast_Cancer'
adata

# %%
# load unprocessed data
BASE_PATH = Path('/auto/brno2/home/luciakostialova/Master_thesis/10xVisium/BRCA1/')
library_id='Visium_Human_Breast_Cancer'
data = sq.read.visium(BASE_PATH, counts_file='Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5')

# %%
ssgsea_df = pd.read_csv('NES.csv', index_col=0)
ssgsea_df

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# %%
def get_markers_dict(adata, n_genes=5, key="rank_genes_groups"):
    result = adata.uns[key]
    groups = result["names"].dtype.names
    
    markers = {}
    
    for group in groups:
        markers[group] = result["names"][group][:n_genes].tolist()
    
    return markers

# %%
adata_emt = ad.read_h5ad("multimodal_results_EMT_clustering_ind/s0_output/adata_s0_segmentation_gene_sets_weights_1_1.h5ad")

# %%
sq.pl.spatial_scatter(adata_emt, color='clusters_8', img_res_key='hires', size=1, alpha=1, dpi=200)

# %%
sc.tl.rank_genes_groups(adata_emt, groupby='clusters_8')

# %%
markers = get_markers_dict(adata_emt, n_genes=5, key="rank_genes_groups")
markers

# %%
adata_emt.obs["clusters_8"] = adata_emt.obs["clusters_8"].astype(str).astype("category")
sc.tl.dendrogram(adata_emt, groupby="clusters_8")

sc.pl.rank_genes_groups_heatmap(
    adata_emt,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
)

# %%
sc.pl.rank_genes_groups_heatmap(
    adata_emt,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(6, 10),    
    show=False                
)

plt.gca().set_xlabel("")
plt.savefig(
    "figures_DE/EMT_k8_s0_rankgenes_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_emt,
    groupby="clusters_8",
    n_genes=5,
    standard_scale="var"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_emt,
    groupby="clusters_8",
    n_genes=5,
    figsize=(14, 4),    
    show=False,
    standard_scale="var"
)

plt.savefig(
    "figures_DE/EMT_k8_s8_rankgenes_dotplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
# plot with original counts, unnormalised.
sq.pl.spatial_scatter(data, color=markers['2'], img_res_key='hires', size=1, alpha=0.8, dpi=200)

# %%
# plot with original counts, unnormalised.
sq.pl.spatial_scatter(data, color=markers['6'], img_res_key='hires', size=1, alpha=0.8, dpi=200)

# %%
adata_er = ad.read_h5ad("multimodal_results_ER_late_clustering_ind/s4_output/adata_s4_segmentation_gene_sets_weights_1_1.h5ad")

# %%
sq.pl.spatial_scatter(adata_er, color='clusters_8', img_res_key='hires', size=1, alpha=1, dpi=200)

# %%
sc.tl.rank_genes_groups(adata_er, groupby='clusters_8')

# %%
markers = get_markers_dict(adata_er, n_genes=5, key="rank_genes_groups")
markers

# %%
adata_er.obs["clusters_8"] = adata_er.obs["clusters_8"].astype(str).astype("category")
sc.tl.dendrogram(adata_er, groupby="clusters_8")

sc.pl.rank_genes_groups_heatmap(
    adata_er,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
)

# %%
sc.pl.rank_genes_groups_heatmap(
    adata_er,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(6, 10),    
    show=False                
)

plt.gca().set_xlabel("")
plt.savefig(
    "figures_DE/ER_late_k8_s4_rankgenes_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_er,
    groupby="clusters_8",
    n_genes=5,
    standard_scale="var"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_er,
    groupby="clusters_8",
    n_genes=5,
    figsize=(14, 4),    
    show=False,
    standard_scale="var"
)

plt.savefig(
    "figures_DE/ER_late_k8_s4_rankgenes_dotplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
# plot with original counts, unnormalised.
sq.pl.spatial_scatter(data, color=markers['1'], img_res_key='hires', size=1, alpha=0.8, dpi=200)

# %%
# plot with original counts, unnormalised.
sq.pl.spatial_scatter(data, color=markers['7'], img_res_key='hires', size=1, alpha=0.8, dpi=200)

# %%
adata_myc = ad.read_h5ad("multimodal_results_MYC_V1/s0_output/adata_s0_segmentation_gene_sets_weights_1_1.h5ad")

# %%
sq.pl.spatial_scatter(adata_myc, color='clusters_8', img_res_key='hires', size=1, alpha=1, dpi=200)

# %%
sc.tl.rank_genes_groups(adata_myc, groupby='clusters_8')

# %%
markers = get_markers_dict(adata_myc, n_genes=5, key="rank_genes_groups")
markers

# %%
adata_myc.obs["clusters_8"] = adata_myc.obs["clusters_8"].astype(str).astype("category")
sc.tl.dendrogram(adata_myc, groupby="clusters_8")

sc.pl.rank_genes_groups_heatmap(
    adata_myc,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
)

# %%
sc.pl.rank_genes_groups_heatmap(
    adata_myc,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(6, 10),    
    show=True                
)

plt.gca().set_xlabel("")
plt.savefig(
    "figures_DE/MYC_k8_s0_rankgenes_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_myc,
    groupby="clusters_8",
    n_genes=5,
    standard_scale="var"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_myc,
    groupby="clusters_8",
    n_genes=5,
    figsize=(14, 4),    
    show=False,
    standard_scale="var"
)

plt.savefig(
    "figures_DE/MYC_k8_s0_rankgenes_dotplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
# plot with original counts, unnormalised.
sq.pl.spatial_scatter(data, color=markers['4'], img_res_key='hires', size=1, alpha=0.8, dpi=200)

# %%
# plot with original counts, unnormalised.
sq.pl.spatial_scatter(data, color=markers['0'], img_res_key='hires', size=1, alpha=0.8, dpi=200)

# %%
adata_gexpr = ad.read_h5ad("multimodal_results_gexpr_correlation/s8_output/adata_s8_segmentation_gene_expression_weights_1_1.h5ad")

# %%
sq.pl.spatial_scatter(adata_gexpr, color='clusters_8', img_res_key='hires', size=1.2, alpha=1, dpi=200)

# %%
sc.tl.rank_genes_groups(adata_gexpr, groupby='clusters_8', method='wilcoxon')

# %%
markers = get_markers_dict(adata_gexpr, n_genes=5, key="rank_genes_groups")
markers

# %%
adata_gexpr.obs["clusters_8"] = adata_gexpr.obs["clusters_8"].astype(str).astype("category")
sc.tl.dendrogram(adata_gexpr, groupby="clusters_8")

sc.pl.rank_genes_groups_heatmap(
    adata_gexpr,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    swap_axes=True,
    cmap="viridis"
)

# %%
sc.pl.rank_genes_groups_heatmap(
    adata_gexpr,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(6, 10),    
    show=False                
)

plt.gca().set_xlabel("")
plt.savefig(
    "figures_DE/gexpr_k8_s8_rankgenes_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_gexpr,
    groupby="clusters_8",
    n_genes=5,
    standard_scale="var"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_gexpr,
    groupby="clusters_8",
    n_genes=5,
    figsize=(14, 4),    
    show=False,
    standard_scale="var"
)

plt.savefig(
    "figures_DE/gexpr_k8_s8_rankgenes_dotplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
adata_gexpr_2 = ad.read_h5ad("multimodal_results_gexpr_correlation/s4_output/adata_s4_segmentation_gene_expression_weights_1_1.h5ad")

# %%
sq.pl.spatial_scatter(adata_gexpr_2, color='clusters_8', img_res_key='hires', size=1.2, alpha=1, dpi=200)

# %%
sc.tl.rank_genes_groups(adata_gexpr_2, groupby='clusters_8', method='wilcoxon')

# %%
markers = get_markers_dict(adata_gexpr_2, n_genes=5, key="rank_genes_groups")
markers

# %%
adata_gexpr_2.obs["clusters_8"] = adata_gexpr_2.obs["clusters_8"].astype(str).astype("category")
sc.tl.dendrogram(adata_gexpr_2, groupby="clusters_8")

sc.pl.rank_genes_groups_heatmap(
    adata_gexpr_2,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    swap_axes=True,
    cmap="viridis"
)

# %%
sc.pl.rank_genes_groups_heatmap(
    adata_gexpr_2,
    groupby="clusters_8",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(6, 10),    
    show=False                
)

plt.gca().set_xlabel("")
plt.savefig(
    "figures_DE/gexpr_k8_s4_rankgenes_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_gexpr_2,
    groupby="clusters_8",
    n_genes=5,
    standard_scale="var"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_gexpr_2,
    groupby="clusters_8",
    n_genes=5,
    standard_scale="var",
    figsize=(14, 4),    
    show=False 
)

plt.savefig(
    "figures_DE/gexpr_k8_s4_rankgenes_dotplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
adata_gexpr_3 = ad.read_h5ad("multimodal_results_gexpr_correlation/s6_output/adata_s6_segmentation_gene_expression_weights_1_1.h5ad")

# %%
sq.pl.spatial_scatter(adata_gexpr_3, color='clusters_16', img_res_key='hires', size=1.2, alpha=1, dpi=200)

# %%
sc.tl.rank_genes_groups(adata_gexpr_3, groupby='clusters_16', method='wilcoxon')

# %%
markers = get_markers_dict(adata_gexpr_3, n_genes=5, key="rank_genes_groups")
markers

# %%
adata_gexpr_3.obs["clusters_16"] = adata_gexpr_3.obs["clusters_16"].astype(str).astype("category")
sc.tl.dendrogram(adata_gexpr_3, groupby="clusters_16")

sc.pl.rank_genes_groups_heatmap(
    adata_gexpr_3,
    groupby="clusters_16",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    swap_axes=True,
    cmap="viridis"
)

# %%
sc.pl.rank_genes_groups_heatmap(
    adata_gexpr_3,
    groupby="clusters_16",
    n_genes=5,              # top N genes per cluster
    standard_scale="var",   # important for visualization
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(6, 10),    
    show=False                
)

plt.gca().set_xlabel("")
plt.savefig(
    "figures_DE/gexpr_k16_s6_rankgenes_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_gexpr_3,
    groupby="clusters_16",
    n_genes=5,
    standard_scale="var"
)

# %%
sc.pl.rank_genes_groups_dotplot(
    adata_gexpr_3,
    groupby="clusters_16",
    n_genes=5,
    standard_scale="var",
    figsize=(14, 4),    
    show=False 
)

plt.savefig(
    "figures_DE/gexpr_k16_s6_rankgenes_dotplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
