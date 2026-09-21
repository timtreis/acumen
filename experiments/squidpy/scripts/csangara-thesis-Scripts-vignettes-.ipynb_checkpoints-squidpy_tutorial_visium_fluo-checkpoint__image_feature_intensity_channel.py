import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()

sq.im.calculate_image_features(
    adata, img, features="summary", layer="image", key_added="feat_summary",
    n_jobs=4, show_progress_bar=False,
)
df = adata.obsm["feat_summary"]
means = {c.split("_")[1]: df[c].mean() for c in df.columns if c.endswith("_mean")}
# means keys like 'ch-0', 'ch-1', 'ch-2'
ranked = sorted(means.items(), key=lambda kv: kv[1])
print("means:", means)
print("lowest channel (test answer):", ranked[0][0].split("-")[1])
print("highest channel (train answer):", ranked[-1][0].split("-")[1])
