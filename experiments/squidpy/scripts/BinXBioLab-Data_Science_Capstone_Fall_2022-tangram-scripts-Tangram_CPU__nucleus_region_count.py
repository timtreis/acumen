import squidpy as sq


def region_with_most_nuclei(image_loader, adata_loader):
    img = image_loader()
    adata = adata_loader()

    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0)

    features_kwargs = {
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label"],
            "channels": [1, 2],
        }
    }
    sq.im.calculate_image_features(
        adata,
        img,
        layer="image",
        key_added="image_features",
        n_jobs=1,
        show_progress_bar=False,
        features_kwargs=features_kwargs,
        features="segmentation",
        mask_circle=True,
    )

    adata.obs["cell_count"] = adata.obsm["image_features"]["segmentation_label"]
    totals = adata.obs.groupby("cluster")["cell_count"].sum().sort_values(ascending=False)
    print(totals.head())
    return totals.idxmax()


def main():
    train_answer = region_with_most_nuclei(
        sq.datasets.visium_fluo_image_crop, sq.datasets.visium_fluo_adata_crop
    )
    print("TRAIN ANSWER (fluo):", train_answer)

    test_answer = region_with_most_nuclei(
        sq.datasets.visium_hne_image_crop, sq.datasets.visium_hne_adata_crop
    )
    print("TEST ANSWER (hne):", test_answer)


if __name__ == "__main__":
    main()
