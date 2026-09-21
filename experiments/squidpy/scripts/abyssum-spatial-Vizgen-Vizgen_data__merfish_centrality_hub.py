import squidpy as sq

adata = sq.datasets.merfish()

# train: Bregma -29 (batch '0'); test: Bregma 1 (batch '6')
for label, batch in [("train", "0"), ("test", "6")]:
    sub = adata[adata.obs["batch"] == batch].copy()
    sub.obs["Cell_class"] = sub.obs["Cell_class"].cat.remove_unused_categories()
    sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
    sq.gr.centrality_scores(sub, cluster_key="Cell_class")
    df = sub.uns["Cell_class_centrality_scores"]
    top = df["closeness_centrality"].idxmax()
    print(label, batch, "->", top, df.loc[top, "closeness_centrality"])
