import scanpy as sc
import squidpy as sq

adata = sq.datasets.merfish()

# --- train: Bregma section -9.0 ---
sub = adata[adata.obs["Bregma"] == -9.0].copy()
sc.pp.normalize_total(sub)
sc.pp.log1p(sub)
sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(sub, mode="moran")
df = sub.uns["moranI"]
print("TRAIN answer:", df.index[0], df["I"].iloc[0])

# --- test: Bregma section 26.0 ---
sub2 = adata[adata.obs["Bregma"] == 26.0].copy()
sc.pp.normalize_total(sub2)
sc.pp.log1p(sub2)
sq.gr.spatial_neighbors(sub2, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(sub2, mode="moran")
df2 = sub2.uns["moranI"]
print("TEST answer:", df2.index[0], df2["I"].iloc[0])
