"""Confirmation script for task: image_summary_extreme_spot (train + test)."""
import squidpy as sq

# ---- train: visium_hne ----
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()

sq.im.calculate_image_features(
    adata, img, features="summary", key_added="features", show_progress_bar=False
)
df = adata.obsm["features"]
top = df["summary_ch-0_mean"].sort_values(ascending=False)
print("TRAIN (visium_hne) answer:", top.index[0], top.iloc[0])

# ---- test: visium_fluo ----
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata2, img2, features="summary", key_added="features", show_progress_bar=False
)
df2 = adata2.obsm["features"]
top2 = df2["summary_ch-0_mean"].sort_values(ascending=False)
print("TEST (visium_fluo) answer:", top2.index[0], top2.iloc[0])
