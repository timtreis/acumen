import squidpy as sq

# ---- train variant: cropped H&E image + matching spot data ----
adata = sq.datasets.visium_hne_adata_crop()
img = sq.datasets.visium_hne_image_crop()

sq.im.process(img, layer="image", method="smooth")
sq.im.segment(img, channel=0, layer="image_smooth", method="watershed", geq=False, layer_added="segmented_watershed")

sq.im.calculate_image_features(
    adata,
    img,
    layer="image",
    features="segmentation",
    key_added="segmentation_features",
    features_kwargs={
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "area"],
        }
    },
    mask_circle=True,
    n_jobs=1,
    show_progress_bar=False,
)
df = adata.obsm["segmentation_features"]
print("train (visium_hne_adata_crop / visium_hne_image_crop):", round(df["segmentation_area_mean"].mean(), 2))

# ---- test variant: full-resolution H&E image + matching spot data ----
adata2 = sq.datasets.visium_hne_adata()
img2 = sq.datasets.visium_hne_image()

sq.im.process(img2, layer="image", method="smooth")
sq.im.segment(img2, channel=0, layer="image_smooth", method="watershed", geq=False, layer_added="segmented_watershed")

sq.im.calculate_image_features(
    adata2,
    img2,
    layer="image",
    features="segmentation",
    key_added="segmentation_features",
    features_kwargs={
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "area"],
        }
    },
    mask_circle=True,
    n_jobs=1,
    show_progress_bar=False,
)
df2 = adata2.obsm["segmentation_features"]
print("test (visium_hne_adata / visium_hne_image):", round(df2["segmentation_area_mean"].mean(), 2))
