import squidpy as sq
import numpy as np


def top_partner(adata, cluster_key, reference):
    sq.gr.spatial_neighbors(adata)
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
    m = adata.uns[f"{cluster_key}_interactions"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    ref = cats.index(reference)
    row = [(cats[j], m[ref, j]) for j in range(len(cats)) if j != ref]
    row.sort(key=lambda x: -x[1])
    return row[0][0], row[:5]


if __name__ == "__main__":
    # train: imc dataset, "cell type" annotation, reference = "T cells"
    adata_train = sq.datasets.imc()
    train_answer, train_top5 = top_partner(adata_train, "cell type", "T cells")
    print("TRAIN answer:", train_answer)
    print("TRAIN top5:", train_top5)

    # test: seqfish dataset, "celltype_mapped_refined" annotation, reference = "NMP"
    adata_test = sq.datasets.seqfish()
    test_answer, test_top5 = top_partner(adata_test, "celltype_mapped_refined", "NMP")
    print("TEST answer:", test_answer)
    print("TEST top5:", test_top5)
