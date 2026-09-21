# mined from: https://github.com/ccb-hms/spatial-workshop-python/blob/8e815f8e7b9526a7fbf5da4a4afb03e2b2dbe576/notebooks/nb4_spatial_statistics_with_squidpy.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment

# %%
import spatialdata as sd
import spatialdata_plot as sdp
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from pathlib import Path

# For cleaner output
import warnings
warnings.filterwarnings("ignore")

# Define the path to our data directory
# Note: This path is relative to the repository's root directory
_DATA_DIR_PATH = Path("../data/")
_VISIUM_PATH = _DATA_DIR_PATH / "visium_glioblastoma_subset.zarr"
_XENIUM_PATH = _DATA_DIR_PATH / "xenium_lung_cancer_subset.zarr"

# Print versions for reproducibility
for p in [sd, sdp, sc, sq]:
    print(f"{p.__name__}: {p.__version__}")

# %%
sdata_visium = sd.read_zarr("../data/visium_glioblastoma_subset.zarr")

# adata_visium = sdata_visium.tables["table"].copy()

print("Loaded {sdata_visium.tables['table'].n_obs} spots with {sdata_visium.tables['table'].n_vars} genes")

# %%
# Calculate QC metrics
sc.pp.calculate_qc_metrics(sdata_visium.tables["table"], percent_top=(20, 50), inplace=True)

# Filter low-quality spots and rare genes
print("Starting with: {sdata_visium.tables['table'].n_obs} spots")
sc.pp.filter_cells(sdata_visium.tables["table"], min_counts=500)
sc.pp.filter_genes(sdata_visium.tables["table"], min_cells=10)
print("After filtering: {sdata_visium.tables['table'].n_obs} spots")

# Standard preprocessing pipeline
sc.pp.normalize_total(sdata_visium.tables["table"], inplace=True)
sc.pp.log1p(sdata_visium.tables["table"])
sc.pp.highly_variable_genes(sdata_visium.tables["table"])
sc.pp.pca(sdata_visium.tables["table"], use_highly_variable=True)
sc.pp.neighbors(sdata_visium.tables["table"])
sc.tl.leiden(sdata_visium.tables["table"], key_added="leiden_clusters")
sc.tl.umap(sdata_visium.tables["table"])

n_clusters = len(sdata_visium.tables["table"].obs['leiden_clusters'].unique())
print("Identified {n_clusters} tissue regions/clusters")

# %%
(
    sdata_visium
    .pl.render_shapes(color="leiden_clusters", shape="visium_hex")
    .pl.show("downscaled_hires", title="Leiden clusters")
)

# %%
# Build spatial neighborhood graph
# This connects each spot to its spatial neighbors
sq.gr.spatial_neighbors(sdata_visium.tables["table"])

print("Built spatial graph with {sdata_visium.tables['table'].obsp['spatial_connectivities'].nnz} connections")
print("Average neighbors per spot: {sdata_visium.tables['table'].obsp['spatial_connectivities'].nnz / sdata_visium.tables['table'].n_obs:.1f}")

# %%
# Calculate Moran's I for spatially variable gene detection
# We'll test highly variable genes for computational efficiency
hvg_genes = sdata_visium.tables["table"].var_names[sdata_visium.tables["table"].var['highly_variable']]

sq.gr.spatial_autocorr(
    sdata_visium.tables["table"],
    mode="moran",
    genes=hvg_genes,
    n_perms=100,  # Number of permutations for statistical testing
    n_jobs=4      # Parallel processing
)

# Display the top spatially variable genes
moran_results = sdata_visium.tables["table"].uns["moranI"].sort_values(by="I", ascending=False)
print("\nTop 10 spatially variable genes:")
print(moran_results.head(10)[['I', 'pval_sim']].round(4))

print("\nBottom 10 spatially variable genes (most random):")
print(moran_results.tail(10)[['I', 'pval_sim']].round(4))

# %%
# Visualize top spatially variable genes
top_genes = moran_results.head(3).index.tolist()
bottom_genes = moran_results.tail(3).index.tolist()
genes_to_plot = top_genes + bottom_genes

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

for idx, gene in enumerate(genes_to_plot):
    moran_score = moran_results.loc[gene, 'I']
    p_value = moran_results.loc[gene, 'pval_sim']
    
    # Get gene expression values
    # gene_expr = adata_visium[:, gene].X.toarray().flatten() if hasattr(adata_visium.X, 'toarray') else adata_visium[:, gene].X.flatten()

    (
        sdata_visium
        .pl.render_shapes(
            color=gene,
            shape="visium_hex",
        )
        .pl.show(
            "downscaled_hires",
            title=f'{gene}\nMoran\'s I = {moran_score:.3f} (p = {p_value:.3f})',
            ax=axs[idx],
            colorbar=False,
        )
    )

fig.suptitle('Spatially Variable Genes\nTop row: High spatial coherence, Bottom row: Random patterns', fontsize=16)
fig.tight_layout()
fig.show()

print("High Moran's I (top row) = genes with spatially coherent expression")
print("Low Moran's I (bottom row) = genes with spatially random expression")

# %%
# Calculate neighborhood enrichment between tissue regions
sq.gr.nhood_enrichment(sdata_visium.tables["table"], cluster_key="leiden_clusters")

# Visualize the enrichment matrix
fig, ax = plt.subplots(figsize=(10, 8))
sq.pl.nhood_enrichment(
    sdata_visium.tables["table"], 
    cluster_key="leiden_clusters",
    method="ward",  # Hierarchical clustering to group similar patterns
    cmap="RdBu_r",  # Red-blue colormap (red=enriched, blue=depleted)
    ax=ax
)
plt.title('Neighborhood Enrichment Between Tissue Regions')
plt.show()

print("\nInterpretation:")
print("Red (positive Z-score): Regions are neighbors more often than expected by chance")
print("Blue (negative Z-score): Regions avoid each other spatially") 
print("White (Z-score ≈ 0): Random spatial association")

# %%
# Calculate co-occurrence across spatial dimensions
sq.gr.co_occurrence(sdata_visium.tables["table"], cluster_key="leiden_clusters")

# Visualize co-occurrence for the most abundant cluster
cluster_counts = sdata_visium.tables["table"].obs['leiden_clusters'].value_counts()
most_abundant_cluster = cluster_counts.index[0]

# Don't pass ax parameter - let squidpy handle the plotting
plt.figure(figsize=(12, 6))
sq.pl.co_occurrence(
    sdata_visium.tables["table"],
    cluster_key="leiden_clusters", 
    clusters=most_abundant_cluster,
    figsize=(12, 6)  # Use figsize parameter instead of ax
)
plt.suptitle(f'Co-occurrence Analysis: Region {most_abundant_cluster}')
plt.show()

print(f"Analyzed co-occurrence for region {most_abundant_cluster} ({cluster_counts[most_abundant_cluster]} spots)")
print("Co-occurrence score = conditional probability of finding regions together at different distances")

# %%
# Interactive napari demonstration (instructor will run this live)
# Uncomment for live interactive session:
# import napari_spatialdata as nsd
# viewer = nsd.Interactive(sdata_visium)
# print("Napari viewer launched - follow along on instructor's screen")
