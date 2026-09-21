import squidpy as sq
import numpy as np


def top_partner_closest(adata, cluster_key, reference):
    sq.gr.co_occurrence(adata, cluster_key=cluster_key)
    res = adata.uns[f"{cluster_key}_co_occurrence"]
    occ = res["occ"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    ref = cats.index(reference)
    vals = occ[ref, :, 0]
    order = np.argsort(vals)[::-1]
    top = [(cats[i], vals[i]) for i in order if i != ref]
    return top[0][0], top[:5]


if __name__ == "__main__":
    # train: imc dataset, "cell type" annotation, reference = "T cells"
    adata_train = sq.datasets.imc()
    train_answer, train_top5 = top_partner_closest(adata_train, "cell type", "T cells")
    print("TRAIN answer:", train_answer)
    print("TRAIN top5:", train_top5)

    # test: seqfish dataset, "celltype_mapped_refined" annotation, reference = "Endothelium"
    adata_test = sq.datasets.seqfish()
    test_answer, test_top5 = top_partner_closest(adata_test, "celltype_mapped_refined", "Endothelium")
    print("TEST answer:", test_answer)
    print("TEST top5:", test_top5)
