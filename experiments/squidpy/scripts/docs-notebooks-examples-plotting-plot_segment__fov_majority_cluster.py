import squidpy as sq

adata = sq.datasets.mibitof()

for lib in ["point8", "point23"]:
    sub = adata.obs[adata.obs["library_id"] == lib]
    vc = sub["Cluster"].value_counts()
    print(lib, "->", vc.idxmax(), vc.to_dict())
