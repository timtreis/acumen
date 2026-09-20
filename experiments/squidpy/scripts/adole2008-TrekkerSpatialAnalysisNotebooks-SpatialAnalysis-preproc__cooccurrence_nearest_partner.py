"""Task: cooccurrence_nearest_partner
For a given reference cell type, which OTHER annotated cell type has the
highest co-occurrence probability at the shortest spatial distance bin.
train -> seqfish / celltype_mapped_refined, reference = Cardiomyocytes
test  -> imc / cell type, reference = T cells
"""
import numpy as np
import squidpy as sq


def nearest_partner(adata, cluster_key, reference):
    occ, _ = sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=True)
    cats = adata.obs[cluster_key].cat.categories.tolist()
    ri = cats.index(reference)
    probs = occ[:, ri, 0].copy()
    probs[ri] = -np.inf
    order = np.argsort(probs)[::-1]
    return [(cats[i], probs[i]) for i in order]


if __name__ == "__main__":
    train_adata = sq.datasets.seqfish()
    train_ranked = nearest_partner(train_adata, "celltype_mapped_refined", "Cardiomyocytes")
    print("TRAIN (seqfish, reference=Cardiomyocytes) top partners:")
    for name, p in train_ranked[:5]:
        print(f"  {name}: {p:.3f}")
    print("TRAIN ANSWER:", train_ranked[0][0])

    test_adata = sq.datasets.imc()
    test_ranked = nearest_partner(test_adata, "cell type", "T cells")
    print("\nTEST (imc, reference=T cells) top partners:")
    for name, p in test_ranked[:5]:
        print(f"  {name}: {p:.3f}")
    print("TEST ANSWER:", test_ranked[0][0])
