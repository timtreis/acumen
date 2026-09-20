"""Confirmation script for task 'moran_spatial_gene' (train + test variants).

Goal: on the slideseqv2 bundled dataset, build a spatial neighbor graph and
run Moran's I spatial autocorrelation over gene expression to rank genes by
how spatially structured their expression pattern is.

train answer = gene with the highest Moran's I (most spatially organized)
test answer  = gene with the lowest Moran's I (least spatially organized /
               closest to spatially random among the genes tested)
"""
import squidpy as sq

adata = sq.datasets.slideseqv2()

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=15, spatial_key="spatial")
sq.gr.spatial_autocorr(adata, mode="moran")

moran_results = adata.uns["moranI"].sort_values("I", ascending=False)

top_gene = moran_results.index[0]
bottom_gene = moran_results.index[-1]

print("train answer (highest Moran's I):", top_gene)
print("test answer (lowest Moran's I):", bottom_gene)
print(moran_results.head(5))
print(moran_results.tail(5))
