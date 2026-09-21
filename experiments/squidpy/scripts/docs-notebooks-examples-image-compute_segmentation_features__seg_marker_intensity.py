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
                "props": ["label", "mean_intensity"],
                "channels": [1, 2],
            }
        },
        mask_circle=True,
        n_jobs=1,
    )

    df = adata.obsm["segmentation_features"]

    # channel 1 = anti-NEUN (neuronal marker)
    max_ch1_barcode = df["segmentation_ch-1_mean_intensity_mean"].idxmax()
    print("TRAIN answer (highest mean anti-NEUN intensity, barcode):", max_ch1_barcode,
          df["segmentation_ch-1_mean_intensity_mean"].max())

    # channel 2 = anti-GFAP (glial marker)
    max_ch2_barcode = df["segmentation_ch-2_mean_intensity_mean"].idxmax()
    print("TEST answer (highest mean anti-GFAP intensity, barcode):", max_ch2_barcode,
          df["segmentation_ch-2_mean_intensity_mean"].max())
