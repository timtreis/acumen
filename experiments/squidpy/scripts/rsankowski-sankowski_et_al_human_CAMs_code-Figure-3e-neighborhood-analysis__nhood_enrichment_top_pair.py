import squidpy as sq
import numpy as np
import pandas as pd


def main():
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, show_progress_bar=False)
    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    cats = adata.obs["cluster"].cat.categories.tolist()
    df = pd.DataFrame(z, index=cats, columns=cats)

    # --- train: overall strongest-enriched pair of distinct clusters ---
    mat = df.values.copy()
    np.fill_diagonal(mat, -np.inf)
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    print("TRAIN answer (top pair):", cats[idx[0]], "&", cats[idx[1]], "z=", mat[idx])

    # --- test: strongest enrichment partner of the Fiber_tract cluster ---
    row = df.loc["Fiber_tract"].copy()
    row["Fiber_tract"] = -np.inf
    print("TEST answer (Fiber_tract's top partner):", row.idxmax(), "z=", row.max())


if __name__ == "__main__":
    main()
