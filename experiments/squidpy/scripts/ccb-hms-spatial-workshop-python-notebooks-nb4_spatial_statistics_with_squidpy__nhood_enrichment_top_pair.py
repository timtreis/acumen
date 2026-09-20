import warnings
warnings.filterwarnings("ignore")
import squidpy as sq
import numpy as np

def top_enriched_pair(loader):
    adata = loader()
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1, show_progress_bar=False)
    z = np.array(adata.uns["cluster_nhood_enrichment"]["zscore"], dtype=float)
    cats = adata.obs["cluster"].cat.categories.tolist()
    n = len(cats)
    offdiag = z.copy()
    for i in range(n):
        offdiag[i, i] = -np.inf
    i, j = np.unravel_index(np.argmax(offdiag), offdiag.shape)
    return cats[i], cats[j], offdiag[i, j]

if __name__ == "__main__":
    a, b, score = top_enriched_pair(sq.datasets.visium_hne_adata)
    print("train (visium_hne):", a, "&", b, score)

    a2, b2, score2 = top_enriched_pair(sq.datasets.visium_fluo_adata)
    print("test (visium_fluo):", a2, "&", b2, score2)
