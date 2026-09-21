import numpy as np
import squidpy as sq


def rgb2gray(x):
    """Return the mean of numpy array along axis 3."""
    return np.mean(x, axis=3)


def main():
    # --- train variant: Visium H&E tissue image crop ---
    img = sq.datasets.visium_hne_image_crop()
    gray = img.apply(rgb2gray)
    arr = gray["image"].values
    train_answer = round(float(np.mean(arr)), 2)
    print("train (visium_hne_image_crop) mean intensity:", train_answer)

    # --- test variant: Visium fluorescence tissue image crop ---
    img2 = sq.datasets.visium_fluo_image_crop()
    gray2 = img2.apply(rgb2gray)
    arr2 = gray2["image"].values
    test_answer = round(float(np.mean(arr2)), 2)
    print("test (visium_fluo_image_crop) mean intensity:", test_answer)


if __name__ == "__main__":
    main()
