import squidpy as sq
import pandas as pd


def main():
    # train: visium_hne_adata, target cluster = Hippocampus
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)
    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    cats = adata.obs["cluster"].cat.categories.tolist()
    idx = cats.index("Hippocampus")
    row = pd.Series(z[idx], index=cats).drop("Hippocampus")
    print("TRAIN top enriched neighbor of Hippocampus:", row.idxmax())
    print(row.sort_values(ascending=False))

    # test: visium_fluo_adata, target cluster = Dentate_gyrus
    adata2 = sq.datasets.visium_fluo_adata()
    sq.gr.spatial_neighbors(adata2)
    sq.gr.nhood_enrichment(adata2, cluster_key="cluster", seed=0, n_jobs=1)
    z2 = adata2.uns["cluster_nhood_enrichment"]["zscore"]
    cats2 = adata2.obs["cluster"].cat.categories.tolist()
    idx2 = cats2.index("Dentate_gyrus")
    row2 = pd.Series(z2[idx2], index=cats2).drop("Dentate_gyrus")
    print("TEST top enriched neighbor of Dentate_gyrus:", row2.idxmax())
    print(row2.sort_values(ascending=False))


if __name__ == "__main__":
    main()
