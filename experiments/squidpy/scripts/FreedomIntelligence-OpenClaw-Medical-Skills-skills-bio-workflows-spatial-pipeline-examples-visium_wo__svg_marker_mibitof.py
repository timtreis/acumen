import squidpy as sq

a = sq.datasets.mibitof()

for lib in ["point16", "point8"]:
    sub = a[a.obs.library_id == lib].copy()
    sq.gr.spatial_neighbors(sub, coord_type="generic", n_neighs=6)
    sq.gr.spatial_autocorr(sub, mode="moran", genes=sub.var_names.tolist())
    top = sub.uns["moranI"].sort_values("I", ascending=False).head(1)
    print(lib, "-> top spatially autocorrelated marker:", top.index[0], round(float(top["I"].iloc[0]), 3))

# train (point16) answer: CK
# test (point8) answer: CD45
