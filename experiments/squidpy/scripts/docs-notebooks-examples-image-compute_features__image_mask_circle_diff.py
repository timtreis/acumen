"""Confirmation script for task: image_mask_circle_diff (train + test)."""
import squidpy as sq

# ---- train: visium_hne ----
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()

top_spot = adata.obs["total_counts"].sort_values(ascending=False).index[0]
sub = adata[[top_spot]].copy()

sq.im.calculate_image_features(
    sub, img, features="summary", key_added="feat_square", show_progress_bar=False
)
sq.im.calculate_image_features(
    sub, img, features="summary", key_added="feat_circle", mask_circle=True, show_progress_bar=False
)

sq_val = sub.obsm["feat_square"]["summary_ch-0_mean"].iloc[0]
ci_val = sub.obsm["feat_circle"]["summary_ch-0_mean"].iloc[0]
print("TRAIN (visium_hne) top spot:", top_spot)
print("TRAIN answer (square - circle), rounded:", f"{sq_val - ci_val:.2f}")

# ---- test: visium_fluo ----
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()

top_spot2 = adata2.obs["total_counts"].sort_values(ascending=False).index[0]
sub2 = adata2[[top_spot2]].copy()

sq.im.calculate_image_features(
    sub2, img2, features="summary", key_added="feat_square", show_progress_bar=False
)
sq.im.calculate_image_features(
    sub2, img2, features="summary", key_added="feat_circle", mask_circle=True, show_progress_bar=False
)

sq_val2 = sub2.obsm["feat_square"]["summary_ch-0_mean"].iloc[0]
ci_val2 = sub2.obsm["feat_circle"]["summary_ch-0_mean"].iloc[0]
print("TEST (visium_fluo) top spot:", top_spot2)
print("TEST answer (square - circle), rounded:", f"{sq_val2 - ci_val2:.2f}")
