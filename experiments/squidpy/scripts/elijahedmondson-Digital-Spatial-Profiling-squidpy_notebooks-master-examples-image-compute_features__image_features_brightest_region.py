"""
Which annotated tissue region has the highest average image brightness
across its spots?

Train: visium_hne dataset. Test: visium_fluo dataset.
"""
import squidpy as sq


def brightest_region(adata, img):
    sq.im.calculate_image_features(adata, img, features="summary", key_added="features", show_progress_bar=False)
    feats = adata.obsm["features"]
    mean_cols = [c for c in feats.columns if c.endswith("_mean")]
    adata.obs["overall_brightness"] = feats[mean_cols].mean(axis=1).values
    grp = adata.obs.groupby("cluster")["overall_brightness"].mean().sort_values(ascending=False)
    return grp


# train
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()
grp = brightest_region(adata, img)
print(grp)
print("ANSWER (train):", grp.index[0])

# test
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()
grp2 = brightest_region(adata2, img2)
print(grp2)
print("ANSWER (test):", grp2.index[0])
