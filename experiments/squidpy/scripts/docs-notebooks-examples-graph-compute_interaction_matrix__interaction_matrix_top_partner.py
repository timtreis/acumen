"""
Confirmation script for task `interaction_matrix_top_partner`.

Train variant: imc dataset, macrophages -> most frequent spatial neighbor cell type (excluding self)
Test variant: seqfish dataset, NMP cell type -> most frequent spatial neighbor cell type (excluding self)

Both use squidpy.gr.spatial_neighbors (defaults) to build the spatial graph, then
squidpy.gr.interaction_matrix(..., normalized=True) to get the row-normalized interaction
matrix, and for the target row pick the highest-scoring OTHER (non-self) category.
"""
import numpy as np
import squidpy as sq


def top_other_partner(adata, cluster_key, target):
    sq.gr.spatial_neighbors(adata)
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key, normalized=True)
    mat = adata.uns[f"{cluster_key}_interactions"]
    cats = list(adata.obs[cluster_key].cat.categories)
    i = cats.index(target)
    row = mat[i].copy()
    row[i] = -1  # exclude self-interaction
    j = int(np.argmax(row))
    return cats[j], row[j]


if __name__ == "__main__":
    # Train: imc dataset
    adata_imc = sq.datasets.imc()
    partner, score = top_other_partner(adata_imc, "cell type", "macrophages")
    print("TRAIN (imc, macrophages) -> top other partner:", partner, round(float(score), 4))

    # Test: seqfish dataset
    adata_seqfish = sq.datasets.seqfish()
    partner2, score2 = top_other_partner(adata_seqfish, "celltype_mapped_refined", "NMP")
    print("TEST (seqfish, NMP) -> top other partner:", partner2, round(float(score2), 4))
