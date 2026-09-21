import numpy as np
import squidpy as sq


def top_pair(adata, cluster_key="cluster"):
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1, seed=0)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"].copy()
    cats = adata.obs[cluster_key].cat.categories.tolist()
    np.fill_diagonal(z, -np.inf)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    return cats[i], cats[j], z[i, j]


def main():
    adata_train = sq.datasets.visium_hne_adata()
    a, b, z = top_pair(adata_train)
    print("TRAIN (visium_hne_adata):", a, "&", b, "z=", z)

    adata_test = sq.datasets.visium_fluo_adata_crop()
    a2, b2, z2 = top_pair(adata_test)
    print("TEST (visium_fluo_adata_crop):", a2, "&", b2, "z=", z2)


if __name__ == "__main__":
    main()
