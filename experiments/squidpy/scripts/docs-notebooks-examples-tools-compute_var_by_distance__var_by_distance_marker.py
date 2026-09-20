"""
Ground-truth script for task `var_by_distance_marker`.

For the four_i dataset, build a per-cell design matrix of each pixel's
distance to the nearest pixel of a given subcellular-structure anchor
(using cell_id as the library/grouping key so distances are computed
within each imaged cell separately), then find which protein marker's
intensity correlates most strongly (by absolute Pearson correlation)
with that distance.

Train anchor: ER_mitochondria_1  -> answer: VINC
Test anchor:  Nucleolus          -> answer: NUPS
"""

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

import squidpy as sq

warnings.filterwarnings("ignore")


def top_marker_by_distance(adata, anchor):
    ad = adata.copy()
    sq.tl.var_by_distance(
        ad,
        groups=anchor,
        cluster_key="cluster",
        library_key="cell_id",
        design_matrix_key="design_matrix",
    )
    df = ad.obsm["design_matrix"]
    dist = df[anchor].values
    mask = ~np.isnan(dist)

    X = ad.X
    if sp.issparse(X):
        X = X.toarray()

    corrs = {}
    for i, gene in enumerate(ad.var_names):
        vals = X[:, i]
        corrs[gene] = np.corrcoef(dist[mask], vals[mask])[0, 1]

    s = pd.Series(corrs).sort_values(key=lambda x: -x.abs())
    return s


if __name__ == "__main__":
    adata = sq.datasets.four_i()

    train_scores = top_marker_by_distance(adata, "ER_mitochondria_1")
    print("TRAIN (anchor=ER_mitochondria_1)")
    print(train_scores.head(5))
    print("TRAIN ANSWER:", train_scores.index[0])
    print()

    test_scores = top_marker_by_distance(adata, "Nucleolus")
    print("TEST (anchor=Nucleolus)")
    print(test_scores.head(5))
    print("TEST ANSWER:", test_scores.index[0])
