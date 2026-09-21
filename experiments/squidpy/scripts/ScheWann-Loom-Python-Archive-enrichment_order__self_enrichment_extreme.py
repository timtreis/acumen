"""
Task: self_enrichment_extreme

Goal (train + test): among the annotated categories in a spatial dataset, find
the one with the LOWEST diagonal (self) z-score in the neighborhood enrichment
matrix -- i.e. the category least likely to be surrounded by members of its own
kind, relative to a permutation null.

train answer: run on squidpy's imc dataset, cluster_key="cell type"
test answer:  run on squidpy's seqfish dataset, cluster_key="celltype_mapped_refined"
"""
import numpy as np
import pandas as pd
import squidpy as sq


def least_self_enriched(adata, cluster_key):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
    z = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    diag = pd.Series(np.diag(z), index=cats).sort_values()
    print(diag)
    return diag.index[0]


def main():
    print("=== train: imc ===")
    train_adata = sq.datasets.imc()
    train_answer = least_self_enriched(train_adata, "cell type")
    print("TRAIN ANSWER:", train_answer)

    print("\n=== test: seqfish ===")
    test_adata = sq.datasets.seqfish()
    test_answer = least_self_enriched(test_adata, "celltype_mapped_refined")
    print("TEST ANSWER:", test_answer)


if __name__ == "__main__":
    main()
