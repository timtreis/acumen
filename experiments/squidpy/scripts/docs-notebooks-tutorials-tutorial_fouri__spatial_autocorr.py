import squidpy as sq

adata = sq.datasets.four_i()
adata.var_names_make_unique()

sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.spatial_autocorr(adata, mode="moran")

moran = adata.uns["moranI"]
print("train: most spatially variable marker:", moran.index[0], moran["I"].iloc[0])
print("test: least spatially variable marker:", moran.index[-1], moran["I"].iloc[-1])
