import squidpy as sq


def run(dataset_img, dataset_adata, label):
    img = dataset_img()
    adata = dataset_adata()

    sq.im.calculate_image_features(
        adata, img, features="summary", layer="image", key_added="features_summary", n_jobs=1,
        features_kwargs={"summary": {}},
    )
    df = adata.obsm["features_summary"]
    mean_cols = [c for c in df.columns if c.endswith("_mean") and "quantile" not in c]
    stds = df[mean_cols].std().sort_values(ascending=False)
    print(label)
    print(stds)
    top_channel = stds.index[0].split("ch-")[1].split("_")[0]
    print(label, "answer channel:", top_channel)


def main():
    run(sq.datasets.visium_fluo_image_crop, sq.datasets.visium_fluo_adata_crop, "fluo (train)")
    run(sq.datasets.visium_hne_image_crop, sq.datasets.visium_hne_adata_crop, "hne (test)")


if __name__ == "__main__":
    main()
