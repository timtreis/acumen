import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata, img, layer="image", features="summary", key_added="sf",
    n_jobs=1, show_progress_bar=False,
)

# train: first fluorescence channel
adata.obs["mean_ch0"] = adata.obsm["sf"]["summary_ch-0_mean"]
means_ch0 = adata.obs.groupby("cluster")["mean_ch0"].mean().sort_values(ascending=False)
print(means_ch0)
print("TRAIN ANSWER:", means_ch0.index[0])

# test: third fluorescence channel
adata.obs["mean_ch2"] = adata.obsm["sf"]["summary_ch-2_mean"]
means_ch2 = adata.obs.groupby("cluster")["mean_ch2"].mean().sort_values(ascending=False)
print(means_ch2)
print("TEST ANSWER:", means_ch2.index[0])
