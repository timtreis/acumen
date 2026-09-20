"""Ground-truth script for task `image_smooth_std`.

Applies squidpy's Gaussian image-smoothing (sq.im.process with method="smooth",
sigma=2) to two bundled image datasets and reports the standard deviation of
pixel intensities in the resulting smoothed image.

Train variant: visium_hne_image_crop (H&E)
Test variant:  visium_fluo_image_crop (fluorescence)
"""

import squidpy as sq


def smoothed_std(loader, sigma=2):
    img = loader()
    sq.im.process(img, layer="image", method="smooth", sigma=sigma)
    smoothed = img["image_smooth"].values
    return round(float(smoothed.std()), 2)


if __name__ == "__main__":
    train_answer = smoothed_std(sq.datasets.visium_hne_image_crop)
    print("train (visium_hne_image_crop):", train_answer)

    test_answer = smoothed_std(sq.datasets.visium_fluo_image_crop)
    print("test (visium_fluo_image_crop):", test_answer)
