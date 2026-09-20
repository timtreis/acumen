"""
Reproduces the mibitof tutorial's image-feature-extraction analysis:
recover per-cell mean marker intensity directly from the raw MIBI-TOF
images + segmentation mask (converting RGB -> CMYK first, as the three
markers are encoded in C/M/Y), then correlate the recovered per-cell
values against the marker values already stored in adata.X.

Produces both the train answer (weakest agreement -> lowest correlation)
and the test answer (strongest agreement -> highest correlation).
"""

import os
import tempfile

os.makedirs("/tmp/short_tmp_mibitof", exist_ok=True)
os.environ["TMPDIR"] = "/tmp/short_tmp_mibitof"
tempfile.tempdir = None

import numpy as np
import squidpy as sq

adata = sq.datasets.mibitof()

imgs = []
for library_id in adata.uns["spatial"].keys():
    img = sq.im.ImageContainer(
        adata.uns["spatial"][library_id]["images"]["hires"], library_id=library_id
    )
    img.add_img(
        adata.uns["spatial"][library_id]["images"]["segmentation"],
        library_id=library_id,
        layer="segmentation",
    )
    img["segmentation"].attrs["segmentation"] = True
    imgs.append(img)
img = sq.im.ImageContainer.concat(imgs)


def rgb2cmyk(arr):
    """Convert arr from RGB to CMYK color space."""
    R = arr[..., 0] / 255
    G = arr[..., 1] / 255
    B = arr[..., 2] / 255
    K = 1 - (np.max(arr, axis=-1) / 255)
    C = (1 - R - K) / (1 - K + np.finfo(float).eps)
    M = (1 - G - K) / (1 - K + np.finfo(float).eps)
    Y = (1 - B - K) / (1 - K + np.finfo(float).eps)
    return np.stack([C, M, Y, K], axis=3)


img.apply(rgb2cmyk, layer="image", new_layer="image_cmyk", copy=False)


def segmentation_image_intensity(arr, image_cmyk):
    """Calculate per-channel mean intensity of the center segment."""
    import skimage.measure

    s = arr.shape[0]
    mask = (arr == arr[s // 2, s // 2, 0, 0]).astype(int)
    features = []
    for c in range(image_cmyk.shape[-1]):
        feature = skimage.measure.regionprops_table(
            np.squeeze(mask),
            intensity_image=np.squeeze(image_cmyk[:, :, :, c]),
            properties=["mean_intensity"],
        )["mean_intensity"][0]
        features.append(feature)
    return features


sq.im.calculate_image_features(
    adata,
    img,
    library_id="library_id",
    features="custom",
    spot_scale=10,
    layer="segmentation",
    n_jobs=1,
    show_progress_bar=False,
    features_kwargs={
        "custom": {
            "func": segmentation_image_intensity,
            "additional_layers": ["image_cmyk"],
        }
    },
)

channels = ["CD45", "CK", "vimentin"]
corrs = {}
for i, ch in enumerate(channels):
    X = np.array(adata[:, ch].X.todense())[:, 0]
    Y = adata.obsm["img_features"][f"segmentation_image_intensity_{i}"]
    corrs[ch] = np.corrcoef(X, Y)[1, 0]

for ch, c in corrs.items():
    print(f"{ch}: pearson r = {c:.4f}")

weakest = min(corrs, key=corrs.get)
strongest = max(corrs, key=corrs.get)
print(f"\ntrain answer (weakest agreement): {weakest}")
print(f"test answer (strongest agreement): {strongest}")
