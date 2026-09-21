import squidpy as sq

adata = sq.datasets.merfish()

# train: Bregma -29 (batch '0'); test: Bregma 1 (batch '6')
for label, batch in [("train", "0"), ("test", "6")]:
    sub = adata[adata.obs["batch"] == batch].copy()
    sub.obs["Cell_class"] = sub.obs["Cell_class"].cat.remove_unused_categories()
    sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
    sq.gr.spatial_autocorr(sub, mode="moran", n_jobs=1)
    top_gene = sub.uns["moranI"].index[0]
    print(label, batch, "->", top_gene, sub.uns["moranI"].iloc[0]["I"])
