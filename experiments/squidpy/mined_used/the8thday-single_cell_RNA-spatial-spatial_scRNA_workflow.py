# mined from: https://github.com/the8thday/single_cell_RNA/blob/a5de60a7fec81a76db05e0ffa7faba2e2753a291/spatial/spatial_scRNA_workflow.ipynb
# symbols: squidpy.datasets.visium_hne_adata, squidpy.gr.centrality_scores, squidpy.gr.co_occurrence, squidpy.gr.ligrec, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.centrality_scores, squidpy.pl.co_occurrence, squidpy.pl.ligrec, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_scatter

# %%
# Install required packages (uncomment if needed)
# !pip install scanpy squidpy anndata leidenalg scikit-misc
# !pip install cell2location  # For cell type deconvolution
# !pip install stlearn  # Alternative spatial analysis

# %%
# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# scverse ecosystem
import scanpy as sc
import squidpy as sq
import anndata as ad

# Additional utilities
from scipy import sparse
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set plotting defaults
sc.settings.verbosity = 2  # verbosity: 0=errors, 1=warnings, 2=info, 3=hints
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
sc.logging.print_header()

print(f"scanpy version: {sc.__version__}")
print(f"squidpy version: {sq.__version__}")

# %%
# ==============================================================================
# OPTION 1: Load 10x Visium data
# ==============================================================================
# Replace with your actual data path
# visium_path = "/path/to/visium/output"
# adata = sq.read.visium(visium_path, counts_file="filtered_feature_bc_matrix.h5")

# ==============================================================================
# OPTION 2: Load from h5ad file (pre-processed AnnData)
# ==============================================================================
# adata = sc.read_h5ad("/path/to/data.h5ad")

# ==============================================================================
# OPTION 3: Load example dataset for demonstration
# ==============================================================================
# This loads example Visium data from squidpy
adata = sq.datasets.visium_hne_adata()

print(f"Loaded data shape: {adata.shape}")
print(f"Number of spots: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
adata

# %%
# Examine the data structure
print("=" * 50)
print("Observation (cell/spot) metadata columns:")
print(adata.obs.columns.tolist())
print("\n" + "=" * 50)
print("Variable (gene) metadata columns:")
print(adata.var.columns.tolist())
print("\n" + "=" * 50)
print("Spatial coordinates available:")
print("obsm keys:", list(adata.obsm.keys()))
print("uns keys:", list(adata.uns.keys()))

# %%
# Visualize the H&E image with spots (if available)
if 'spatial' in adata.uns:
    sq.pl.spatial_scatter(adata, color=None, size=1.5, figsize=(8, 8))
    plt.title("Spatial distribution of spots")
    plt.show()

# %%
# Calculate QC metrics
# -----------------------------------------------------------------------------

# Identify mitochondrial genes (typically start with 'MT-' or 'mt-')
adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))

# Identify ribosomal genes (typically start with 'RPS' or 'RPL')
adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL', 'Rps', 'Rpl'))

# Identify hemoglobin genes
adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]', case=False)

# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata, 
    qc_vars=['mt', 'ribo', 'hb'], 
    percent_top=None, 
    log1p=False, 
    inplace=True
)

print("QC metrics calculated:")
print(adata.obs[['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo']].describe())

# %%
# Visualize QC metrics
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Total counts distribution
sns.histplot(adata.obs['total_counts'], bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].set_xlabel('Total counts')
axes[0, 0].set_title('Distribution of total counts per spot')

# Number of genes distribution
sns.histplot(adata.obs['n_genes_by_counts'], bins=50, kde=True, ax=axes[0, 1])
axes[0, 1].set_xlabel('Number of genes')
axes[0, 1].set_title('Distribution of genes detected per spot')

# Mitochondrial percentage
sns.histplot(adata.obs['pct_counts_mt'], bins=50, kde=True, ax=axes[0, 2])
axes[0, 2].set_xlabel('% mitochondrial')
axes[0, 2].set_title('Distribution of mitochondrial %')

# Scatter: counts vs genes
axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], 
                   c=adata.obs['pct_counts_mt'], cmap='viridis', s=3, alpha=0.5)
axes[1, 0].set_xlabel('Total counts')
axes[1, 0].set_ylabel('Number of genes')
axes[1, 0].set_title('Counts vs Genes (colored by MT%)')

# Scatter: counts vs MT%
axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], s=3, alpha=0.5)
axes[1, 1].set_xlabel('Total counts')
axes[1, 1].set_ylabel('% mitochondrial')
axes[1, 1].set_title('Counts vs MT%')

# Ribosomal percentage
sns.histplot(adata.obs['pct_counts_ribo'], bins=50, kde=True, ax=axes[1, 2])
axes[1, 2].set_xlabel('% ribosomal')
axes[1, 2].set_title('Distribution of ribosomal %')

plt.tight_layout()
plt.show()

# %%
# Spatial visualization of QC metrics
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sq.pl.spatial_scatter(
    adata, 
    color='total_counts',
    size=1.3,
    cmap='viridis',
    ax=axes[0],
    title='Total counts'
)

sq.pl.spatial_scatter(
    adata, 
    color='n_genes_by_counts',
    size=1.3,
    cmap='viridis',
    ax=axes[1],
    title='Number of genes'
)

sq.pl.spatial_scatter(
    adata, 
    color='pct_counts_mt',
    size=1.3,
    cmap='Reds',
    ax=axes[2],
    title='MT%'
)

plt.tight_layout()
plt.show()

# %%
# Apply QC filtering
# -----------------------------------------------------------------------------
# IMPORTANT: Adjust these thresholds based on your data!

print(f"Before filtering: {adata.n_obs} spots, {adata.n_vars} genes")

# Define thresholds (customize based on QC plots above)
min_counts = 500
max_counts = 50000
min_genes = 200
max_mt_pct = 20

# Filter spots
sc.pp.filter_cells(adata, min_counts=min_counts)
sc.pp.filter_cells(adata, min_genes=min_genes)
adata = adata[adata.obs['total_counts'] < max_counts, :].copy()
adata = adata[adata.obs['pct_counts_mt'] < max_mt_pct, :].copy()

# Filter genes (remove genes expressed in very few spots)
sc.pp.filter_genes(adata, min_cells=10)

print(f"After filtering: {adata.n_obs} spots, {adata.n_vars} genes")

# %%
# Store raw counts for later use (e.g., differential expression)
adata.layers['counts'] = adata.X.copy()

# Normalization: TPM-like normalization to 10,000 counts per spot
sc.pp.normalize_total(adata, target_sum=1e4)

# Log transformation
sc.pp.log1p(adata)

# Store normalized data
adata.layers['normalized'] = adata.X.copy()

print("Normalization and log transformation completed.")

# %%
# Identify Highly Variable Genes (HVGs)
# -----------------------------------------------------------------------------
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,  # Number of HVGs to select
    flavor='seurat_v3',  # Method for HVG selection
    layer='counts',  # Use raw counts for HVG selection
    subset=False  # Keep all genes, just mark HVGs
)

print(f"Number of highly variable genes: {adata.var['highly_variable'].sum()}")

# Visualize HVG selection
sc.pl.highly_variable_genes(adata)

# %%
# Scale data (for PCA)
# -----------------------------------------------------------------------------
# Only scale HVGs for efficiency
adata_hvg = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_hvg, max_value=10)

print("Data scaling completed on HVGs.")

# %%
# PCA on HVGs
# -----------------------------------------------------------------------------
sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=50)

# Copy PCA results back to main adata
adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
adata.varm['PCs'] = np.zeros((adata.n_vars, 50))
adata.varm['PCs'][adata.var['highly_variable'], :] = adata_hvg.varm['PCs']
adata.uns['pca'] = adata_hvg.uns['pca']

# Visualize variance explained
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

print("PCA completed.")

# %%
# Determine optimal number of PCs (elbow method)
# -----------------------------------------------------------------------------
# Look at the variance ratio plot above and choose n_pcs at the "elbow"
n_pcs = 30  # Adjust based on variance plot

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)

print(f"Neighborhood graph computed with {n_pcs} PCs.")

# %%
# UMAP embedding
# -----------------------------------------------------------------------------
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

# Visualize UMAP
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc.pl.umap(adata, color='total_counts', ax=axes[0], show=False, title='UMAP - Total counts')
sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1], show=False, title='UMAP - N genes')

plt.tight_layout()
plt.show()

# %%
# Leiden clustering
# -----------------------------------------------------------------------------
# Resolution parameter controls granularity (higher = more clusters)
resolutions = [0.3, 0.5, 0.8, 1.0]

for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
    n_clusters = adata.obs[f'leiden_{res}'].nunique()
    print(f"Resolution {res}: {n_clusters} clusters")

# %%
# Visualize clustering at different resolutions
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for idx, res in enumerate(resolutions):
    ax = axes[idx // 2, idx % 2]
    sc.pl.umap(adata, color=f'leiden_{res}', ax=ax, show=False,
               title=f'Leiden resolution={res}', legend_loc='on data')

plt.tight_layout()
plt.show()

# %%
# Choose optimal resolution and set as main clustering
# -----------------------------------------------------------------------------
optimal_resolution = 0.5  # Adjust based on visualization above
adata.obs['leiden'] = adata.obs[f'leiden_{optimal_resolution}']

print(f"Final clustering: {adata.obs['leiden'].nunique()} clusters")

# %%
# Spatial visualization of clusters
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# UMAP view
sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, 
           title='Clusters (UMAP)', legend_loc='on data')

# Spatial view
sq.pl.spatial_scatter(adata, color='leiden', size=1.3, ax=axes[1],
                      title='Clusters (Spatial)')

plt.tight_layout()
plt.show()

# %%
# Define marker genes for common cell types (customize for your tissue)
# -----------------------------------------------------------------------------
marker_genes = {
    'T cells': ['CD3D', 'CD3E', 'CD4', 'CD8A', 'TRAC'],
    'B cells': ['CD19', 'CD79A', 'MS4A1', 'CD79B'],
    'NK cells': ['NKG7', 'GNLY', 'NCAM1', 'KLRD1'],
    'Monocytes': ['CD14', 'LYZ', 'FCGR3A', 'MS4A7'],
    'Macrophages': ['CD68', 'CD163', 'MARCO', 'MSR1'],
    'Dendritic cells': ['ITGAX', 'CD1C', 'CLEC9A', 'FCER1A'],
    'Fibroblasts': ['COL1A1', 'COL1A2', 'DCN', 'LUM'],
    'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'ENG'],
    'Epithelial': ['EPCAM', 'KRT8', 'KRT18', 'KRT19'],
}

# Filter to genes present in dataset
available_markers = {}
for cell_type, genes in marker_genes.items():
    present = [g for g in genes if g in adata.var_names]
    if present:
        available_markers[cell_type] = present
        print(f"{cell_type}: {len(present)}/{len(genes)} markers found")

# %%
# Visualize marker genes expression
# -----------------------------------------------------------------------------
# Flatten marker list for dotplot
flat_markers = [g for genes in available_markers.values() for g in genes]

if flat_markers:
    sc.pl.dotplot(adata, flat_markers, groupby='leiden', 
                  dendrogram=True, standard_scale='var',
                  title='Marker gene expression by cluster')
else:
    print("No marker genes found in dataset. Using top cluster markers instead.")

# %%
# Find marker genes for each cluster (data-driven approach)
# -----------------------------------------------------------------------------
sc.tl.rank_genes_groups(
    adata, 
    groupby='leiden', 
    method='wilcoxon',  # 't-test', 'wilcoxon', 'logreg'
    pts=True,  # Calculate percentage of cells expressing each gene
    key_added='rank_genes_leiden'
)

# Visualize top markers
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, key='rank_genes_leiden')

# %%
# Heatmap of top markers per cluster
# -----------------------------------------------------------------------------
sc.pl.rank_genes_groups_heatmap(
    adata, 
    n_genes=5, 
    key='rank_genes_leiden',
    groupby='leiden', 
    show_gene_labels=True,
    cmap='viridis'
)

# %%
# Manual annotation based on marker expression
# -----------------------------------------------------------------------------
# Create a mapping from cluster to cell type (customize based on your analysis)
cluster_annotation = {
    '0': 'Cell Type A',
    '1': 'Cell Type B',
    '2': 'Cell Type C',
    # Add more mappings...
}

# Apply annotation (only if you've defined the mapping)
# adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_annotation)

# For now, use clusters as cell types
adata.obs['cell_type'] = adata.obs['leiden'].astype(str)

print("Cell type annotation completed (placeholder).")
print("Please update cluster_annotation dict based on marker analysis.")

# %%
# Extract DE results as DataFrame
# -----------------------------------------------------------------------------
def get_de_results(adata, key='rank_genes_leiden', group=None):
    """Extract DE results for a specific group or all groups."""
    result = sc.get.rank_genes_groups_df(adata, group=group, key=key)
    return result

# Get DE results for all clusters
de_results = get_de_results(adata, key='rank_genes_leiden')
print(f"Total DE results: {len(de_results)} gene-cluster combinations")
de_results.head(20)

# %%
# Filter significant DE genes
# -----------------------------------------------------------------------------
pval_threshold = 0.05
logfc_threshold = 0.5

significant_de = de_results[
    (de_results['pvals_adj'] < pval_threshold) & 
    (de_results['logfoldchanges'].abs() > logfc_threshold)
]

print(f"Significant DE genes: {len(significant_de)}")
print(f"Unique genes: {significant_de['names'].nunique()}")

# %%
# Volcano plot for a specific cluster
# -----------------------------------------------------------------------------
def plot_volcano(de_df, cluster, pval_thresh=0.05, logfc_thresh=0.5):
    """Create volcano plot for a specific cluster."""
    df = de_df[de_df['group'] == str(cluster)].copy()
    df['-log10(pval)'] = -np.log10(df['pvals_adj'] + 1e-300)
    
    # Color points
    df['significant'] = 'Not significant'
    df.loc[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > logfc_thresh), 'significant'] = 'Up'
    df.loc[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] < -logfc_thresh), 'significant'] = 'Down'
    
    colors = {'Up': 'red', 'Down': 'blue', 'Not significant': 'gray'}
    
    plt.figure(figsize=(10, 8))
    for sig, color in colors.items():
        subset = df[df['significant'] == sig]
        plt.scatter(subset['logfoldchanges'], subset['-log10(pval)'], 
                   c=color, label=sig, alpha=0.6, s=10)
    
    plt.axhline(-np.log10(pval_thresh), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(logfc_thresh, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(-logfc_thresh, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10(Adjusted P-value)')
    plt.title(f'Volcano Plot - Cluster {cluster}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot for first cluster
plot_volcano(de_results, cluster='0')

# %%
# Build spatial neighborhood graph
# -----------------------------------------------------------------------------
# Method 1: Fixed radius
sq.gr.spatial_neighbors(adata, coord_type='generic', radius=None, n_neighs=6)

# Method 2: Delaunay triangulation (alternative)
# sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)

print("Spatial neighborhood graph computed.")
print(f"Connectivities shape: {adata.obsp['spatial_connectivities'].shape}")

# %%
# Compute spatial autocorrelation (Moran's I)
# -----------------------------------------------------------------------------
# This measures whether gene expression is spatially clustered
sq.gr.spatial_autocorr(
    adata,
    mode='moran',
    n_perms=100,
    n_jobs=-1  # Use all CPU cores
)

# View results
moran_results = adata.uns['moranI']
print("Top spatially variable genes (by Moran's I):")
print(moran_results.sort_values('I', ascending=False).head(20))

# %%
# Visualize top spatially variable genes
# -----------------------------------------------------------------------------
top_spatial_genes = moran_results.sort_values('I', ascending=False).head(6).index.tolist()

sq.pl.spatial_scatter(
    adata, 
    color=top_spatial_genes,
    size=1.3,
    cmap='viridis',
    ncols=3,
    figsize=(15, 10)
)

# %%
# Neighborhood enrichment analysis
# -----------------------------------------------------------------------------
# Tests whether certain cell type pairs are spatially enriched/depleted
sq.gr.nhood_enrichment(adata, cluster_key='leiden')

# Visualize
sq.pl.nhood_enrichment(
    adata, 
    cluster_key='leiden',
    method='average',
    figsize=(8, 8),
    title='Neighborhood Enrichment'
)

# %%
# Co-occurrence analysis
# -----------------------------------------------------------------------------
# Measures how often cell types occur together at different distances
sq.gr.co_occurrence(
    adata,
    cluster_key='leiden',
    spatial_key='spatial',
    n_splits=1
)

# Visualize
sq.pl.co_occurrence(
    adata,
    cluster_key='leiden',
    clusters=['0', '1'],  # Adjust cluster IDs
    figsize=(10, 5)
)

# %%
# Centrality scores
# -----------------------------------------------------------------------------
# Compute network centrality for each cluster
sq.gr.centrality_scores(adata, cluster_key='leiden')

# Visualize
sq.pl.centrality_scores(adata, cluster_key='leiden', figsize=(12, 4))

# %%
# Load ligand-receptor database
# -----------------------------------------------------------------------------
# Squidpy has built-in LR databases
lr_pairs = sq.gr.ligrec(
    adata,
    cluster_key='leiden',
    n_perms=100,
    copy=True,
    use_raw=False,
    transmitter_params={'categories': 'ligand'},
    receiver_params={'categories': 'receptor'}
)

print("Ligand-receptor analysis completed.")

# %%
# Visualize significant LR interactions
# -----------------------------------------------------------------------------
sq.pl.ligrec(
    lr_pairs,
    source_groups=['0', '1'],  # Adjust
    target_groups=['2', '3'],  # Adjust
    pvalue_threshold=0.05,
    figsize=(10, 8)
)

# %%
# Compute spatial domain features
# -----------------------------------------------------------------------------
# Use cluster composition in local neighborhoods as features
sq.gr.spatial_neighbors(adata, n_neighs=15, coord_type='generic')

# Compute cluster proportions in neighborhoods
def compute_neighborhood_composition(adata, cluster_key='leiden', n_neighbors=15):
    """Compute cell type composition in neighborhoods."""
    from scipy.sparse import csr_matrix
    
    # Get adjacency matrix
    adj = adata.obsp['spatial_connectivities']
    
    # One-hot encode clusters
    clusters = pd.get_dummies(adata.obs[cluster_key])
    
    # Compute neighborhood composition
    nhood_comp = adj @ clusters.values
    nhood_comp = nhood_comp / (nhood_comp.sum(axis=1, keepdims=True) + 1e-10)
    
    # Add to adata
    comp_df = pd.DataFrame(
        nhood_comp,
        index=adata.obs_names,
        columns=[f'nhood_{c}' for c in clusters.columns]
    )
    
    return comp_df

nhood_comp = compute_neighborhood_composition(adata)
print("Neighborhood composition computed.")
nhood_comp.head()

# %%
# Cluster neighborhoods to identify spatial niches
# -----------------------------------------------------------------------------
from sklearn.cluster import KMeans

# Perform k-means on neighborhood composition
n_niches = 5  # Adjust based on data
kmeans = KMeans(n_clusters=n_niches, random_state=42, n_init=10)
adata.obs['spatial_niche'] = kmeans.fit_predict(nhood_comp).astype(str)

print(f"Identified {n_niches} spatial niches.")
print(adata.obs['spatial_niche'].value_counts())

# %%
# Visualize spatial niches
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(adata, color='spatial_niche', ax=axes[0], show=False,
           title='Spatial Niches (UMAP)')

sq.pl.spatial_scatter(adata, color='spatial_niche', size=1.3, ax=axes[1],
                      title='Spatial Niches (Tissue)')

plt.tight_layout()
plt.show()

# %%
# Characterize niches by cell type composition
# -----------------------------------------------------------------------------
niche_composition = pd.crosstab(
    adata.obs['spatial_niche'],
    adata.obs['leiden'],
    normalize='index'
)

plt.figure(figsize=(10, 6))
sns.heatmap(niche_composition, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Cell Type Composition per Spatial Niche')
plt.xlabel('Cell Type (Leiden)')
plt.ylabel('Spatial Niche')
plt.tight_layout()
plt.show()

# %%
# Multi-panel spatial plot
# -----------------------------------------------------------------------------
genes_to_plot = top_spatial_genes[:4] if top_spatial_genes else ['total_counts']

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for idx, gene in enumerate(genes_to_plot):
    ax = axes[idx // 2, idx % 2]
    sq.pl.spatial_scatter(
        adata, 
        color=gene,
        size=1.5,
        cmap='magma',
        ax=ax,
        title=gene
    )

plt.tight_layout()
plt.savefig('spatial_gene_expression.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved: spatial_gene_expression.png")

# %%
# Summary visualization
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 12))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# UMAP with clusters
ax1 = fig.add_subplot(gs[0, 0])
sc.pl.umap(adata, color='leiden', ax=ax1, show=False, 
           title='Clusters (UMAP)', legend_loc='right margin')

# Spatial clusters
ax2 = fig.add_subplot(gs[0, 1])
sq.pl.spatial_scatter(adata, color='leiden', size=1.2, ax=ax2,
                      title='Clusters (Spatial)')

# Spatial niches
ax3 = fig.add_subplot(gs[0, 2])
sq.pl.spatial_scatter(adata, color='spatial_niche', size=1.2, ax=ax3,
                      title='Spatial Niches')

# QC metric
ax4 = fig.add_subplot(gs[1, 0])
sq.pl.spatial_scatter(adata, color='total_counts', size=1.2, ax=ax4,
                      cmap='viridis', title='Total Counts')

# Top gene 1
ax5 = fig.add_subplot(gs[1, 1])
if top_spatial_genes:
    sq.pl.spatial_scatter(adata, color=top_spatial_genes[0], size=1.2, ax=ax5,
                          cmap='magma', title=f'{top_spatial_genes[0]} Expression')

# Top gene 2
ax6 = fig.add_subplot(gs[1, 2])
if len(top_spatial_genes) > 1:
    sq.pl.spatial_scatter(adata, color=top_spatial_genes[1], size=1.2, ax=ax6,
                          cmap='magma', title=f'{top_spatial_genes[1]} Expression')

plt.savefig('summary_figure.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved: summary_figure.png")

# %%
# Save processed AnnData object
# -----------------------------------------------------------------------------
output_path = 'processed_spatial_data.h5ad'
adata.write_h5ad(output_path)
print(f"Data saved to: {output_path}")

# %%
# Export key results as CSV
# -----------------------------------------------------------------------------
# Cell metadata
adata.obs.to_csv('cell_metadata.csv')
print("Cell metadata saved: cell_metadata.csv")

# DE results
de_results.to_csv('differential_expression_results.csv', index=False)
print("DE results saved: differential_expression_results.csv")

# Spatial autocorrelation
moran_results.to_csv('spatial_autocorrelation_moran.csv')
print("Moran's I results saved: spatial_autocorrelation_moran.csv")

# Neighborhood composition
nhood_comp.to_csv('neighborhood_composition.csv')
print("Neighborhood composition saved: neighborhood_composition.csv")

# %%
# Session info
# -----------------------------------------------------------------------------
print("=" * 60)
print("Analysis Session Summary")
print("=" * 60)
print(f"\nFinal dataset dimensions: {adata.n_obs} spots x {adata.n_vars} genes")
print(f"Number of clusters: {adata.obs['leiden'].nunique()}")
print(f"Number of spatial niches: {adata.obs['spatial_niche'].nunique()}")
print(f"Number of HVGs: {adata.var['highly_variable'].sum()}")
print(f"\nLayers available: {list(adata.layers.keys())}")
print(f"Embeddings available: {list(adata.obsm.keys())}")
print("\n" + "=" * 60)
sc.logging.print_header()
