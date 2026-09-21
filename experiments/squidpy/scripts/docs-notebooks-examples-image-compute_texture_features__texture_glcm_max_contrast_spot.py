"""
Ground-truth script for task id: texture_glcm_max_contrast_spot

Goal: compute GLCM-based texture features per spot (using a wider
context crop around each spot, as recommended in the compute_texture_features
tutorial), then find which spot has the highest average contrast texture
value in the first image channel (averaged over the default angles).

Train variant uses the visium_fluo dataset; test variant uses visium_hne.
"""

import squidpy as sq


def top_contrast_barcode(img, adata, channel=0, spot_scale=2):
    sq.im.calculate_image_features(
        adata,
        img,
        features="texture",
        key_added="texture_features",
        spot_scale=spot_scale,
        show_progress_bar=False,
    )
    df = adata.obsm["texture_features"]
    cols = [c for c in df.columns if c.startswith(f"texture_ch-{channel}_contrast_")]
    scores = df[cols].mean(axis=1)
    return scores.idxmax(), scores.max()


if __name__ == "__main__":
    # train: visium_fluo
    img_fluo = sq.datasets.visium_fluo_image_crop()
    adata_fluo = sq.datasets.visium_fluo_adata_crop()
    barcode, value = top_contrast_barcode(img_fluo, adata_fluo)
    print("train (visium_fluo) answer:", barcode, "value:", value)

    # test: visium_hne
    img_hne = sq.datasets.visium_hne_image_crop()
    adata_hne = sq.datasets.visium_hne_adata_crop()
    barcode, value = top_contrast_barcode(img_hne, adata_hne)
    print("test (visium_hne) answer:", barcode, "value:", value)
