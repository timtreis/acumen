import squidpy as sq

adata = sq.datasets.merfish()

for bregma in [-9.0, 21.0]:
    adata_slice = adata[adata.obs.Bregma == bregma].copy()
    sq.gr.spatial_neighbors(adata_slice, coord_type="generic")
    sq.gr.spatial_autocorr(adata_slice, mode="moran")
    top_gene = adata_slice.uns["moranI"].index[0]
    print(bregma, "-> top gene by Moran's I:", top_gene)

# train answer (Bregma -9.0): Nnat
# test answer (Bregma 21.0): Mbp
