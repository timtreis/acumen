import squidpy as sq


def top_channel(adata, img):
    sq.im.calculate_image_features(adata, img.compute(), features="summary", n_jobs=1)
    feats = adata.obsm["img_features"]
    means = [c for c in feats.columns if c.endswith("_mean")]
    avg = feats[means].mean()
    return avg.idxmax(), avg


if __name__ == "__main__":
    # TRAIN: visium_hne (H&E) image + adata
    adata = sq.datasets.visium_hne_adata()
    img = sq.datasets.visium_hne_image()
    top, avg = top_channel(adata, img)
    print("TRAIN visium_hne top channel:", top)
    print(avg)

    # TEST: visium_fluo (fluorescence) crop image + matching crop adata
    adata2 = sq.datasets.visium_fluo_adata_crop()
    img2 = sq.datasets.visium_fluo_image_crop()
    top2, avg2 = top_channel(adata2, img2)
    print("TEST visium_fluo_crop top channel:", top2)
    print(avg2)
