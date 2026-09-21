import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata,
    img,
    features="histogram",
    features_kwargs={"histogram": {"bins": 3, "channels": [0, 1]}},
    key_added="histogram_features",
    n_jobs=1,
    show_progress_bar=False,
)

df = adata.obsm["histogram_features"]

# TRAIN: channel 0, highest-intensity (last) bin
train_col = "histogram_ch-0_bin-2"
print("TRAIN answer:", df[train_col].idxmax(), df[train_col].max())

# TEST: channel 1, highest-intensity (last) bin
test_col = "histogram_ch-1_bin-2"
print("TEST answer:", df[test_col].idxmax(), df[test_col].max())
