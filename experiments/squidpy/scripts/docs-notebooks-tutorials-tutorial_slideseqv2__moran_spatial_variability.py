import scanpy as sc
import squidpy as sq

adata = sc.read_h5ad("/tmp/slideseqv2.h5ad")  # sq.datasets.slideseqv2()

sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.spatial_autocorr(adata, mode="moran")

df = adata.uns["moranI"]
most_variable = df["I"].idxmax()
least_variable = df["I"].abs().idxmin()

print("Most spatially variable gene (highest Moran's I):", most_variable, df.loc[most_variable, "I"])
print("Least spatially variable gene (Moran's I closest to 0):", least_variable, df.loc[least_variable, "I"])
