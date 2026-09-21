import squidpy as sq
import pandas as pd


def top_co_occurring_partner(adata, cluster_key, target):
    sq.gr.co_occurrence(adata, cluster_key=cluster_key)
    res = adata.uns[f"{cluster_key}_co_occurrence"]
    occ = res["occ"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    idx = cats.index(target)
    # first interval = shortest distance range
    sub = pd.Series(occ[idx][:, 0], index=cats).drop(target)
    return sub.sort_values(ascending=False).index[0], sub.sort_values(ascending=False).iloc[0]


# --- train: seqfish dataset ---
adata_train = sq.datasets.seqfish()
partner, score = top_co_occurring_partner(adata_train, "celltype_mapped_refined", "Lateral plate mesoderm")
print("TRAIN answer:", partner, score)

# --- test: imc dataset ---
adata_test = sq.datasets.imc()
partner, score = top_co_occurring_partner(adata_test, "cell type", "T cells")
print("TEST answer:", partner, score)
