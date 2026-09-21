# mined from: https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/9b94a3cde1d03e30639d2469d3d8672749898a8a/Skills/Genomics/Single_Cell/scverse_Ecosystem/squidpy/tutorials/repo/examples/tools/compute_var_by_distance.ipynb
# symbols: squidpy.datasets.mibitof, squidpy.pl.var_by_distance, squidpy.tl.var_by_distance

# %%
# %matplotlib inline

# %%
import squidpy as sq

# %%
adata = sq.datasets.mibitof()

# %%
adata.obs

# %%
sq.tl.var_by_distance(
    adata=adata,
    groups="Epithelial",
    cluster_key="Cluster",
    library_key="library_id",
    covariates=["category", "donor"],
)

# %%
adata.obsm["design_matrix"]

# %%
sq.pl.var_by_distance(
    adata=adata,
    design_matrix_key="design_matrix",
    var="CD98",
    anchor_key="Epithelial",
    covariate="donor",
    line_palette=["blue", "orange"],
    show_scatter=False,
    figsize=(5, 4),
)
