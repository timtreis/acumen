"""Ground-truth script for task `sepal_score` (train + test variants).

Computes the Sepal score (diffusion-based spatial variability score,
squidpy.gr.sepal) for a single named gene on a Visium dataset. The score for
one gene is independent of any other genes passed alongside it (verified:
running sepal on a single gene reproduces the value obtained when the gene is
scored as part of a larger batch), so the answer does not depend on which
gene subset an agent chooses to compute over.
"""

import squidpy as sq

# --- train variant: visium_hne data, gene Ecel1 ---
adata_train = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata_train)
sq.gr.sepal(adata_train, max_neighs=6, genes=["Ecel1"], n_jobs=1, show_progress_bar=False)
train_score = adata_train.uns["sepal_score"].loc["Ecel1", "sepal_score"]
print("TRAIN Ecel1 sepal score (exact):", train_score)
print("TRAIN Ecel1 sepal score (rounded 2dp):", round(train_score, 2))

# --- test variant: visium_fluo data, gene Fzd5 ---
adata_test = sq.datasets.visium_fluo_adata()
sq.gr.spatial_neighbors(adata_test)
sq.gr.sepal(adata_test, max_neighs=6, genes=["Fzd5"], n_jobs=1, show_progress_bar=False)
test_score = adata_test.uns["sepal_score"].loc["Fzd5", "sepal_score"]
print("TEST Fzd5 sepal score (exact):", test_score)
print("TEST Fzd5 sepal score (rounded 2dp):", round(test_score, 2))
