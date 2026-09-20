import numpy as np
import squidpy as sq


def main():
    adata = sq.datasets.merfish()

    # train: Bregma -29 (batch '0'); test: Bregma 1 (batch '6')
    for label, batch in [("train", "0"), ("test", "6")]:
        sub = adata[adata.obs["batch"] == batch].copy()
        sub.obs["Cell_class"] = sub.obs["Cell_class"].cat.remove_unused_categories()
        sq.gr.spatial_neighbors(sub, coord_type="generic", delaunay=True)
        sq.gr.nhood_enrichment(sub, cluster_key="Cell_class", n_jobs=1)
        z = np.array(sub.uns["Cell_class_nhood_enrichment"]["zscore"], dtype=float).copy()
        cats = sub.obs["Cell_class"].cat.categories
        np.fill_diagonal(z, -np.inf)
        idx = np.unravel_index(np.nanargmax(z), z.shape)
        print(label, batch, "->", cats[idx[0]], "and", cats[idx[1]], "z=", z[idx])


if __name__ == "__main__":
    main()
