import warnings
warnings.filterwarnings("ignore")

import squidpy as sq

# ---- train variant: visium_hne ----
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()

sq.im.calculate_image_features(
    adata, img, layer="image", features="summary", key_added="img_features",
    n_jobs=1, show_progress_bar=False,
)
feats = adata.obsm["img_features"]
mean_cols = [c for c in feats.columns if c.endswith("_mean")]
overall = feats[mean_cols].mean(axis=1)
train_answer = overall.idxmax()
print("train (visium_hne) top spot:", train_answer)

# ---- test variant: visium_fluo ----
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata2, img2, layer="image", features="summary", key_added="img_features",
    n_jobs=1, show_progress_bar=False,
)
feats2 = adata2.obsm["img_features"]
mean_cols2 = [c for c in feats2.columns if c.endswith("_mean")]
overall2 = feats2[mean_cols2].mean(axis=1)
test_answer = overall2.idxmax()
print("test (visium_fluo) top spot:", test_answer)
