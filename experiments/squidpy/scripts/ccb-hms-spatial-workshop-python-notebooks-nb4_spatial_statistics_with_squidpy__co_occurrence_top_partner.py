import warnings
warnings.filterwarnings("ignore")
import squidpy as sq
import numpy as np

def top_co_occurring_partner(loader):
    adata = loader()
    cats = adata.obs["cluster"].cat.categories.tolist()
    counts = adata.obs["cluster"].value_counts()
    most_abundant = counts.index[0]
    idx_self = cats.index(most_abundant)

    sq.gr.co_occurrence(adata, cluster_key="cluster")
    occ = adata.uns["cluster_co_occurrence"]["occ"]
    sub = occ[idx_self, :, :].copy()
    sub[idx_self, :] = -np.inf
    j, d = np.unravel_index(np.argmax(sub), sub.shape)
    return most_abundant, cats[j], sub[j, d]

if __name__ == "__main__":
    ref, partner, score = top_co_occurring_partner(sq.datasets.visium_hne_adata)
    print("train (visium_hne): most abundant =", ref, "-> top partner =", partner, score)

    ref2, partner2, score2 = top_co_occurring_partner(sq.datasets.visium_fluo_adata)
    print("test (visium_fluo): most abundant =", ref2, "-> top partner =", partner2, score2)
