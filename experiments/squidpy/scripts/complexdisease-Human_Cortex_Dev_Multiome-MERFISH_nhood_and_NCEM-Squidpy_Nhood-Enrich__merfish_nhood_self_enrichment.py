import numpy as np
import pandas as pd
import squidpy as sq


def top_self_enrichment(bregma, seed=0):
    adata = sq.datasets.merfish()
    adata = adata[adata.obs.Bregma == bregma].copy()

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="Cell_class", seed=seed, show_progress_bar=False)

    zscore = adata.uns["Cell_class_nhood_enrichment"]["zscore"]
    cats = list(adata.obs["Cell_class"].cat.categories)
    diag = pd.Series(np.diag(zscore), index=cats).sort_values(ascending=False)
    return diag


if __name__ == "__main__":
    train = top_self_enrichment(-9.0)
    print("train (Bregma -9) top self-enrichment:")
    print(train.head(3))

    test = top_self_enrichment(21.0)
    print("\ntest (Bregma 21) top self-enrichment:")
    print(test.head(3))
