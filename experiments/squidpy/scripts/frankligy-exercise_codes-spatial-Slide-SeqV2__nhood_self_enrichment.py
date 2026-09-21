# Confirms the answer for task "nhood_self_enrichment" (train + test).
import numpy as np
import squidpy as sq


def top_self_enriched_cluster(adata, cluster_key):
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_jobs=1)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    diag = np.diag(z)
    order = np.argsort(-diag)
    return [(cats[i], float(diag[i])) for i in order[:5]]


if __name__ == "__main__":
    # train: slideseqv2
    adata_train = sq.datasets.slideseqv2()
    print("train (slideseqv2):", top_self_enriched_cluster(adata_train, "cluster"))

    # test: seqfish
    adata_test = sq.datasets.seqfish()
    print("test (seqfish):", top_self_enriched_cluster(adata_test, "celltype_mapped_refined"))
