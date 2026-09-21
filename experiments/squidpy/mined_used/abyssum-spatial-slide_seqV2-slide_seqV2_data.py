# mined from: https://github.com/abyssum/spatial/blob/8d0f5e8eeefeda47a2960cd4a6cabda863a18c13/slide_seqV2/slide_seqV2_data.ipynb
# symbols: squidpy.datasets.slideseqv2, squidpy.gr.ligrec, squidpy.gr.nhood_enrichment, squidpy.gr.ripley, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.ligrec, squidpy.pl.nhood_enrichment, squidpy.pl.ripley, squidpy.pl.spatial_scatter

# %%
# %matplotlib inline

# %%
import squidpy as sq

print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
adata = sq.datasets.slideseqv2()
adata

# %%
sq.pl.spatial_scatter(adata, color="cluster", size=1, shape=None)

# %%
# Neighborhood enrichment analysis

# %%
sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.nhood_enrichment(adata, cluster_key="cluster")
sq.pl.nhood_enrichment(
    adata, cluster_key="cluster", method="single", cmap="inferno", vmin=-50, vmax=100
)

# %%
sq.pl.spatial_scatter(
    adata,
    shape=None,
    color="cluster",
    groups=["Endothelial_Tip", "Ependymal", "Oligodendrocytes", "Polydendrocytes"],
    size=3,
)

# %%
# Ripley’s statistics

# %%
mode = "L"
sq.gr.ripley(adata, cluster_key="cluster", mode=mode, max_dist=500)
sq.pl.ripley(adata, cluster_key="cluster", mode=mode)

# %%
sq.pl.spatial_scatter(
    adata,
    color="cluster",
    groups=["Mural", "CA1_CA2_CA3_Subiculum"],
    size=3,
    shape=None,
)

# %%
# Ligand-receptor interaction analysis

# %%
sq.gr.ligrec(
    adata,
    n_perms=100,
    cluster_key="cluster",
    clusters=["Polydendrocytes", "Oligodendrocytes"],
)
sq.pl.ligrec(
    adata,
    cluster_key="cluster",
    source_groups="Oligodendrocytes",
    target_groups=["Polydendrocytes"],
    pvalue_threshold=0.05,
    swap_axes=True,
)

# %%
# Spatially variable genes with spatial autocorrelation statistics

# %%
sq.gr.spatial_autocorr(adata, mode="moran")
adata.uns["moranI"].head(10)

# %%
sq.pl.spatial_scatter(
    adata,
    shape=None,
    color=["Ttr", "Plp1", "Mbp", "Hpca", "Enpp2"],
    size=0.1,
)

# %%

