import numpy as np
import squidpy as sq

adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()
img = sq.datasets.visium_fluo_image_crop()

sq.im.process(img=img, layer="image", method="smooth")
sq.im.segment(
    img=img,
    layer="image_smooth",
    method="watershed",
    channel=0,
)

features_kwargs = {
    "segmentation": {
        "label_layer": "segmented_watershed",
        "props": ["label", "centroid"],
        "channels": [1, 2],
    }
}
sq.im.calculate_image_features(
    adata_st,
    img,
    layer="image",
    key_added="image_features",
    features_kwargs=features_kwargs,
    features="segmentation",
    mask_circle=True,
    n_jobs=1,
    show_progress_bar=False,
)

adata_st.obs["cell_count"] = adata_st.obsm["image_features"]["segmentation_label"]

per_cluster = adata_st.obs.groupby("cluster")["cell_count"].sum()
print(per_cluster)

print("TRAIN answer (cluster with most nuclei):", per_cluster.idxmax())
print("TEST answer (cluster with fewest nuclei):", per_cluster.idxmin())
