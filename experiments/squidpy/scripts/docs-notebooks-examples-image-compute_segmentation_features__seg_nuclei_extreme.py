if __name__ == "__main__":
    import squidpy as sq

    img = sq.datasets.visium_fluo_image_crop()
    adata = sq.datasets.visium_fluo_adata_crop()

    sq.im.segment(
        img=img,
        layer="image",
        layer_added="segmented_watershed",
        method="watershed",
        channel=0,
    )

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
    )

    df = adata.obsm["segmentation_features"]

    # train: spot with most nuclei detected
    max_label_barcode = df["segmentation_label"].idxmax()
    print("TRAIN answer (most nuclei, barcode):", max_label_barcode, df["segmentation_label"].max())

    # test: spot with largest mean nucleus area
    max_area_barcode = df["segmentation_area_mean"].idxmax()
    print("TEST answer (largest mean nucleus area, barcode):", max_area_barcode, df["segmentation_area_mean"].max())
