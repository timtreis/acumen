import scanpy as sc
import squidpy as sq


def top_svg(adata):
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None, n_jobs=1)
    moran = adata.uns["moranI"]
    svg = moran[moran["pval_norm_fdr_bh"] < 0.05].sort_values("I", ascending=False)
    return svg.index[0], svg["I"].iloc[0]


def main():
    adata_train = sq.datasets.visium_hne_adata()
    gene, i = top_svg(adata_train)
    print("TRAIN (visium_hne_adata) top SVG:", gene, "I=", i)

    adata_test = sq.datasets.visium_fluo_adata_crop()
    gene2, i2 = top_svg(adata_test)
    print("TEST (visium_fluo_adata_crop) top SVG:", gene2, "I=", i2)


if __name__ == "__main__":
    main()
