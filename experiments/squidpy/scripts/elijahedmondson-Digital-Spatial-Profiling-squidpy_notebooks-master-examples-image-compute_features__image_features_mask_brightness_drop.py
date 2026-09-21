"""
By how much does masking the image crop to the circular spot area lower the
average per-spot image brightness compared to the default square crop?

Train: visium_hne dataset. Test: visium_fluo dataset.
"""
import squidpy as sq


def brightness_drop(adata, img):
    sq.im.calculate_image_features(adata, img, features="summary", key_added="features_default", show_progress_bar=False)
    sq.im.calculate_image_features(
        adata, img, features="summary", key_added="features_masked", mask_circle=True, show_progress_bar=False
    )
    mean_cols = [c for c in adata.obsm["features_default"].columns if c.endswith("_mean")]
    default_brightness = adata.obsm["features_default"][mean_cols].mean(axis=1).mean()
    masked_brightness = adata.obsm["features_masked"][mean_cols].mean(axis=1).mean()
    return default_brightness - masked_brightness


# train
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()
drop = brightness_drop(adata, img)
print("train drop:", drop, "rounded:", round(drop, 2))

# test
img2 = sq.datasets.visium_fluo_image_crop()
adata2 = sq.datasets.visium_fluo_adata_crop()
drop2 = brightness_drop(adata2, img2)
print("test drop:", drop2, "rounded:", round(drop2, 2))
