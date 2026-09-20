import numpy as np
import pandas as pd
import squidpy as sq


def top_interaction_partner(adata, cluster_key, target):
    mat = np.array(adata.uns[f"{cluster_key}_interactions"], dtype=float).copy()
    cats = list(adata.obs[cluster_key].cat.categories)
    df = pd.DataFrame(mat, index=cats, columns=cats)
    col = df[target].copy()
    col[target] = -1
    return col.idxmax(), float(col.max())


if __name__ == "__main__":
    adata = sq.datasets.seqfish()
    key = "celltype_mapped_refined"

    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=False, n_neighs=10)
    sq.gr.interaction_matrix(adata, cluster_key=key, normalized=False)

    train_partner, train_n = top_interaction_partner(adata, key, "Lateral plate mesoderm")
    print("TRAIN (target='Lateral plate mesoderm'):", train_partner, train_n)

    test_partner, test_n = top_interaction_partner(adata, key, "Gut tube")
    print("TEST (target='Gut tube'):", test_partner, test_n)
