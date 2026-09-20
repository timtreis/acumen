"""
Confirms the answer for task `zstack_section_summary_intensity` (train and test variants).

Builds a z-stack ImageContainer from two serial Visium sagittal mouse-brain sections
(following squidpy's tutorial_image_container_zstacks tutorial), computes per-spot
"summary" image features (mean/std/quantiles per color channel) for one of the two
sections, and reports the average channel-0 mean-intensity feature across its spots.
"""

import warnings

import anndata as ad
import scanpy as sc
import squidpy as sq

warnings.filterwarnings("ignore")


def run(library_ids):
    adatas, imgs = [], []
    for library_id in library_ids:
        a = sc.datasets.visium_sge(library_id, include_hires_tiff=False)
        a.var_names_make_unique()
        adatas.append(a)
        imgs.append(
            sq.im.ImageContainer(
                a.uns["spatial"][library_id]["images"]["hires"],
                scale=a.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"],
            )
        )

    adata = ad.concat(adatas, uns_merge="only", label="library_id", keys=library_ids, index_unique="-")
    img = sq.im.ImageContainer.concat(imgs, library_ids=library_ids)

    target = library_ids[0]
    sub = adata[adata.obs["library_id"] == target].copy()
    sq.im.calculate_image_features(
        sub, img, layer="image", library_id=target, features="summary", show_progress_bar=False
    )
    feats = sub.obsm["img_features"]
    value = round(float(feats["summary_ch-0_mean"].mean()), 3)
    return value


if __name__ == "__main__":
    train_value = run(
        ["V1_Mouse_Brain_Sagittal_Posterior", "V1_Mouse_Brain_Sagittal_Posterior_Section_2"]
    )
    print("train answer:", train_value)

    test_value = run(
        ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Anterior_Section_2"]
    )
    print("test answer:", test_value)
