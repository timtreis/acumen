# mined from: https://github.com/Harshavardhan-BV/PHCCO-SpatialTranscriptomics/blob/3afff0db70ed5b59461ac785037e243fe7d91d11/Spatial_HandsOn.py
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter, squidpy.read.visium

# %% [markdown]
# # Spatial Transcriptomics Hands-On (PHCCO Summer Training 2025)

# %%
import os
import pandas as pd
import scanpy as sc
import gseapy as gp
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt
sc.settings.figdir = './figures'
os.makedirs('./figures', exist_ok=True)
os.makedirs('./Output', exist_ok=True)
plt.rcParams['svg.hashsalt'] = ''
sns.set_context('talk')

# %% [markdown]
# ## The Data structure
# We will be using the [Breast Cancer Visium](https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-breast-cancer-cytassist-ffpe-2-standard) dataset from 10x genomics. 
# 
# First take note of the directory structure. Usually [spaceranger](https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/output-overview) gives the output in a certain format. We need a minimum of the [`filtered_feature_bc_matrix.h5`](https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5) and a [`spatial`](https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_spatial.tar.gz) directory containing `tissue_hires_image.png`, `scalefactors_json.json`, and `tissue_positions_list.csv` for loading into squidpy. There are ways to load data not in this format but it is recommended to rearrange the data in this format if possible.
# 

# %%
DS = './Data/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer'

# %%
adata = sq.read.visium(DS)

# %%
adata

# %% [markdown]
# Let us take a look at the data. It is pretty much the same as the scanpy object (AnnData) with adata.obs for the cell annotations and adata.var for the gene annotations and adata.X being the gene expression matrix. Note that we have raw counts.

# %%
adata.obs

# %%
adata.var

# %%
adata.X.todense()

# %% [markdown]
# But in addition to that we also have some spatial specific data. We got the coordinates of spots, images and some metadata such as scaling factors.

# %%
adata.obsm['spatial']

# %%
adata.uns['spatial']

# %%
sq.pl.spatial_scatter(adata)

# %% [markdown]
# The data provided has information about the immunofluorescence channel intensities also. Let us add that to our data structure.

# %%
metdat = pd.read_csv(f'{DS}/spatial/barcode_fluorescence_intensity.csv', index_col=0)

# %%
metdat

# %%
metdat.drop('in_tissue', axis=1, inplace=True)

# %%
adata.obs = adata.obs.join(metdat)

# %%
adata.obs

# %% [markdown]
# ## Preprocessing
# We can now start with the pre-processing.
# 
# First let us look at the data quality.

# %%
def QC_metrics(adata):
    # Make unique if duplicates
    adata.var_names_make_unique()
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )
    varibs = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    sc.pl.violin(
        adata,
        varibs,
        jitter=0.4,
        multi_panel=True,
    )

# %%
QC_metrics(adata)

# %% [markdown]
# Then perform log-normalization.

# %%
def normalize(adata, target_sum=10000):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata,target_sum=target_sum)
    adata.layers["norm"] = adata.X.copy()
    sc.pp.log1p(adata)

# %%
normalize(adata)

# %% [markdown]
# And then we can do UMAP dimensional reduction and leiden clustering. These steps are the same as single cell.
# We'll also plot these clusters on our spatial image. Remember that these clusters depend on the parameters used, do no treat them as the ground truth.

# %%
sc.pp.pca(adata)
sc.pl.pca_variance_ratio(adata)

# %%
def umap_cluster(adata, res=None, n_pcs=None, img_res_key='hires'):
    sc.pp.neighbors(adata, 
                    n_neighbors=30, 
                    n_pcs=n_pcs
                    )
    sc.tl.umap(adata, 
               min_dist=0.3, 
               spread=1.0
               )
    if 'leiden_colors' in adata.uns_keys():
        adata.uns.pop('leiden_colors')
    sc.tl.leiden(adata, flavor='igraph', n_iterations=2, resolution=res)
    sc.pl.umap(adata, color='leiden', save='_leiden.png')
    sq.pl.spatial_scatter(adata, color='leiden', save='spatial_leiden.png', img_res_key=img_res_key)

# %%
umap_cluster(adata,res=0.2, n_pcs=5)

# %% [markdown]
# ## Visualization
# 
# Let us visualize the total_counts per spot and compare it with the DAPI (Nuclei) and PCNA (Proliferating cells) intensities.

# %%
sq.pl.spatial_scatter(adata,color=['leiden','total_counts', 'DAPI_mean', 'PCNA', 'PCNA_mean'],ncols=3, save='count_markers.png')

# %% [markdown]
# Now let us look at try to determine the tumour regions and subtypes
# - ER+ve Basal markers: ERS1 (Estrogen receptor), GATA3, PGR and FOXA1 
# - Epithelial markers: CDH1 (E-Cadherin), CD24
# - Mesenchymal markers: CD44, VIM

# %%
subtype_gene = ['ESR1', 'GATA3', 'PGR', 'FOXA1', 'CDH1', 'CD24', 'CD44','VIM']
sq.pl.spatial_scatter(adata, color=['leiden'] + subtype_gene + ['Vimentin_mean'], ncols=5, save='tumour_markers.png')

# %%
sc.pl.violin(adata, subtype_gene[:4], groupby='leiden', save='_tumour_markers1.png')

# %%
sc.pl.violin(adata, subtype_gene[4:], groupby='leiden', save='_tumour_markers2.png')

# %% [markdown]
# From these it looks like clusters 1 and 2 are tumour regions (Note that these labels could different when you run). Among them, cluster 2 is the core and cluster 1 is towards the periphery. 
# 
# These are mostly epithelial in nature as expected but the peripheral regions also show some mesenchymal charecteristics. However, we cannot differentiate if this is due to cancer cells undergoing EMT or due to prevalance of (normal) mesenchymal celltypes.

# %% [markdown]
# Now let us look at some immune cell markers.
# - CD3D: Generic T cell
# - CD4: T-helper cells
# - CD8A: Cytotoxic T cells
# - CD68: Macrophage
# - FN1, COL1A2: Fibroblasts
# - PECAM1: Endothelial cells
# - NKG7: Natural Killer (NK) cells 
# - CD79A: B cells

# %%
immune_gene = ['CD3D', 'CD4','CD8A', 'CD68', 'FN1', 'COL1A2', 'PECAM1', 'NKG7', 'CD79A']
sq.pl.spatial_scatter(adata, color=['leiden'] + immune_gene, ncols=5, save='immune_markers.png')

# %%
sc.pl.violin(adata, immune_gene[:4], groupby='leiden', save='_immune_markers1.png')

# %%
sc.pl.violin(adata, immune_gene[4:], groupby='leiden', save='_immune_markers2.png')

# %% [markdown]
# Cluster 0 tends to be enriched for fibroblasts and stromal cells whereas Cluster 3 is enriched for immune cells. 

# %% [markdown]
# ## Marker Genes
# 
# We can obtain genes that are highly expressed specific to the different clusters.

# %%
sc.tl.rank_genes_groups(adata, groupby='leiden', method='logreg')

# %%
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, standard_scale='var', save='DEG.png')

# %% [markdown]
# We can put these genes through Gene Ontology to get an idea of what the regions could be.

# %%
top_genes = pd.DataFrame({
    group: adata.uns['rank_genes_groups']['names'][group][:100]
    for group in adata.uns['rank_genes_groups']['names'].dtype.names
})
top_genes.to_csv('Output/top100_genes.csv', index=False)

# %% [markdown]
# ## Spatial statistics

# %%
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(adata, cluster_key="leiden",save='nhood.png')

# %% [markdown]
# This says that cluster 2 tend to be together. Cluster 1 tend to be neighbour with cluster 2 followed by 0 and 1. This makes sense that they are the core vs periphery.

# %%
sq.gr.co_occurrence(adata, cluster_key="leiden")
sq.pl.co_occurrence(
    adata,
    cluster_key="leiden",
    clusters="3",
    figsize=(8, 4),
    save='co_occur.png'
)

# %% [markdown]
# Co-occurance isn't really helpful due to the lower resolution of our region clusters.
# 
# However, Moran's I is more useful as it helps us identify spatially variable genes.

# %%
sc.pp.highly_variable_genes(adata)
sq.gr.spatial_neighbors(adata)
genes = adata[:, adata.var.highly_variable].var_names.values[:1000]
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=genes,
    n_perms=100,
    n_jobs=1, #Multi-threading has issues with dask
)

# %%
adata.uns["moranI"].head(10)

# %%
adata.uns["moranI"].to_csv('./Output/MoransI.csv')

# %%
sq.pl.spatial_scatter(adata, color=['leiden']+ adata.uns["moranI"].head(11).index.tolist(), save='spatially_variable.png')

# %%
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=adata.obs.columns[adata.obs.columns.str.endswith('_mean')].to_list(),
    attr='obs',
    n_perms=100,
    n_jobs=1, #Multi-threading has issues with dask
)

# %%
adata.uns["moranI"].to_csv('./Output/IF_MoransI.csv')

# %%



