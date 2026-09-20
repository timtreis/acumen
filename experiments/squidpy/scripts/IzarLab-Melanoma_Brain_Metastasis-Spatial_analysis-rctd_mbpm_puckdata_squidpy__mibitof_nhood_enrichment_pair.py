import itertools

import squidpy as sq


def top_pair(library_id):
    adata_full = sq.datasets.mibitof()
    adata = adata_full[adata_full.obs["library_id"] == library_id].copy()
    adata.obs["Cluster"] = adata.obs["Cluster"].cat.remove_unused_categories()

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="Cluster", show_progress_bar=False)

    z = adata.uns["Cluster_nhood_enrichment"]["zscore"]
    cats = adata.obs["Cluster"].cat.categories.tolist()

    pairs = []
    for i, j in itertools.combinations(range(len(cats)), 2):
        pairs.append((cats[i], cats[j], z[i, j]))
    pairs.sort(key=lambda x: -x[2])
    return pairs[0]


if __name__ == "__main__":
    train_pair = top_pair("point16")
    print("train (point16):", train_pair)

    test_pair = top_pair("point23")
    print("test (point23):", test_pair)
