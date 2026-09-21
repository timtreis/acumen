import squidpy as sq
import numpy as np

def top_pair(adata, cluster_key, seed=0):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=seed, n_jobs=1)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    n = len(cats)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((z[i, j], cats[i], cats[j]))
    pairs.sort(reverse=True)
    best = pairs[0]
    a, b = sorted([best[1], best[2]], key=str.lower)
    return f"{a}, {b}", pairs[:5]

if __name__ == "__main__":
    # train: imc dataset, "cell type" annotation
    adata_train = sq.datasets.imc()
    train_answer, train_top5 = top_pair(adata_train, "cell type")
    print("TRAIN answer:", train_answer)
    print("TRAIN top5:", train_top5)

    # test: mibitof dataset, "Cluster" annotation
    adata_test = sq.datasets.mibitof()
    test_answer, test_top5 = top_pair(adata_test, "Cluster")
    print("TEST answer:", test_answer)
    print("TEST top5:", test_top5)
