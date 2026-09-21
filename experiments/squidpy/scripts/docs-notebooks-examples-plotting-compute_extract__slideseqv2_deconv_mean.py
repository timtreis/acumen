import squidpy as sq

adata = sq.datasets.slideseqv2()

# deconvolution results (per-spot cell-type proportions) are precomputed and
# stored in adata.obsm["deconvolution_results"]; the notebook moves them into
# adata.obs (via sq.pl.extract) so they can be summarized/plotted like any
# other observation-level annotation.
deconv = adata.obsm["deconvolution_results"]

for cell_type in ["Astrocytes", "Mural"]:
    mean_prop = deconv[cell_type].mean()
    print(cell_type, round(mean_prop, 4), "-> rounded to 2dp:", round(mean_prop, 2))
