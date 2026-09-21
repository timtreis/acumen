# Confirms the answer for task "moran_top_svg" (train + test).
import squidpy as sq


def top_moran_gene(adata):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.spatial_autocorr(adata, mode="moran")
    return adata.uns["moranI"].head(5)


# train: slideseqv2
adata_train = sq.datasets.slideseqv2()
print("train (slideseqv2):")
print(top_moran_gene(adata_train))

# test: seqfish
adata_test = sq.datasets.seqfish()
print("test (seqfish):")
print(top_moran_gene(adata_test))
