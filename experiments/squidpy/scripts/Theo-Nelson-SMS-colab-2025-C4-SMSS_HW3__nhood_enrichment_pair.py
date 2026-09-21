import numpy as np
import squidpy as sq


def main():
    # --- train: imc dataset, cell type annotations ---
    adata = sq.datasets.imc()
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    sq.gr.nhood_enrichment(adata, cluster_key="cell type", n_jobs=1, seed=0)

    z = adata.uns["cell type_nhood_enrichment"]["zscore"]
    cats = adata.obs["cell type"].cat.categories.tolist()
    zz = z.copy()
    np.fill_diagonal(zz, -np.inf)
    i, j = np.unravel_index(np.argmax(zz), zz.shape)
    print("TRAIN answer:", cats[i], cats[j], zz[i, j])

    # --- test: seqfish dataset, mouse organogenesis cell types ---
    adata2 = sq.datasets.seqfish()
    sq.gr.spatial_neighbors(adata2, coord_type="generic", delaunay=True)
    sq.gr.nhood_enrichment(adata2, cluster_key="celltype_mapped_refined", n_jobs=1, seed=0)

    z2 = adata2.uns["celltype_mapped_refined_nhood_enrichment"]["zscore"]
    cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()
    zz2 = z2.copy()
    np.fill_diagonal(zz2, -np.inf)
    i2, j2 = np.unravel_index(np.argmax(zz2), zz2.shape)
    print("TEST answer:", cats2[i2], cats2[j2], zz2[i2, j2])


if __name__ == "__main__":
    main()
