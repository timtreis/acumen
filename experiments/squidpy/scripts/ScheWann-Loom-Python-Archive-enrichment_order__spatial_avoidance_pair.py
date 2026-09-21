"""
Task: spatial_avoidance_pair

Goal (train + test): among all pairs of annotated categories in a spatial
dataset, find the pair with the most negative off-diagonal z-score in the
neighborhood enrichment matrix -- i.e. the two categories that avoid being
spatial neighbors of each other more than any other pair, relative to a
permutation null.

train answer: run on squidpy's imc dataset, cluster_key="cell type"
test answer:  run on squidpy's four_i dataset, cluster_key="cluster"
"""
import numpy as np
import squidpy as sq


def most_avoiding_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"].copy()
    cats = adata.obs[cluster_key].cat.categories.tolist()
    np.fill_diagonal(z, np.inf)
    i, j = np.unravel_index(np.argmin(z), z.shape)
    print("min z-score:", z[i, j])
    return cats[i], cats[j]


def main():
    print("=== train: imc ===")
    train_adata = sq.datasets.imc()
    train_answer = most_avoiding_pair(train_adata, "cell type")
    print("TRAIN ANSWER:", train_answer)

    print("\n=== test: four_i ===")
    test_adata = sq.datasets.four_i()
    test_answer = most_avoiding_pair(test_adata, "cluster")
    print("TEST ANSWER:", test_answer)


if __name__ == "__main__":
    main()
