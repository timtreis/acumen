import squidpy as sq


def top_feature(img_fn, adata_fn):
    img = img_fn()
    adata = adata_fn()
    sq.im.calculate_image_features(
        adata, img, features=["summary", "histogram"], n_jobs=1, show_progress_bar=False
    )
    feats = adata.obsm["img_features"]
    means = feats.mean(axis=0).sort_values(ascending=False)
    return means.index[0], float(means.iloc[0])


# train: visium fluorescence crop dataset
name, val = top_feature(sq.datasets.visium_fluo_image_crop, sq.datasets.visium_fluo_adata_crop)
print("train (visium_fluo_crop) top feature:", name, val)

# test: visium H&E crop dataset
name, val = top_feature(sq.datasets.visium_hne_image_crop, sq.datasets.visium_hne_adata_crop)
print("test (visium_hne_crop) top feature:", name, val)
