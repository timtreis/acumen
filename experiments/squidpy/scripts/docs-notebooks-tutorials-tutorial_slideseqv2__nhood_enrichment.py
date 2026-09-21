import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq


def main():
    adata = sc.read_h5ad("/tmp/slideseqv2.h5ad")  # sq.datasets.slideseqv2()

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)

    zscore = np.array(adata.uns["cluster_nhood_enrichment"]["zscore"], copy=True)
    cats = adata.obs["cluster"].cat.categories.tolist()
    vals = zscore.copy()
    np.fill_diagonal(vals, -np.inf)
    df = pd.DataFrame(vals, index=cats, columns=cats)

    for target in ["Ependymal", "Oligodendrocytes"]:
        row = df.loc[target].drop(target)
        print(target, "-> most enriched with:", row.idxmax(), row.max())


if __name__ == "__main__":
    main()
