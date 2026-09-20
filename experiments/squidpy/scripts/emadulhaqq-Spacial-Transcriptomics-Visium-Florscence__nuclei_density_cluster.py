import squidpy as sq


def main():
    img = sq.datasets.visium_fluo_image_crop()
    adata = sq.datasets.visium_fluo_adata_crop()

    sq.im.process(img=img, layer="image", method="smooth")
    sq.im.segment(img=img, layer="image_smooth", method="watershed", channel=0, chunks=1000)

    features_kwargs = {"segmentation": {"label_layer": "segmented_watershed"}}
    sq.im.calculate_image_features(
        adata, img, features="segmentation", layer="image",
        key_added="features_segmentation", n_jobs=1, features_kwargs=features_kwargs,
    )

    seg_df = adata.obsm["features_segmentation"]
    adata.obs["n_segments"] = seg_df["segmentation_label"].values

    grp = adata.obs.groupby("cluster")["n_segments"].mean().sort_values(ascending=False)
    print(grp)
    print("train (highest):", grp.idxmax())
    print("test (lowest):", grp.idxmin())


if __name__ == "__main__":
    main()
