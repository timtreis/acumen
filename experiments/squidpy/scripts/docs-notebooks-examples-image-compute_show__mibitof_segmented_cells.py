"""
Ground-truth script for task `mibitof_segmented_cells`.

Reproduces the workflow shown in docs/notebooks/examples/image/compute_show.ipynb:
build a squidpy ImageContainer per mibitof tissue sample, attach the matching
segmentation mask as a marked segmentation layer, concatenate the containers,
then read off how many distinct (nonzero) cell labels appear in a given
sample's segmentation layer.

train target: point16
test target:  point23
"""

import numpy as np
import squidpy as sq

adata = sq.datasets.mibitof()

imgs = []
for library_id in adata.uns["spatial"].keys():
    img = sq.im.ImageContainer(
        adata.uns["spatial"][library_id]["images"]["hires"], library_id=library_id
    )
    img.add_img(
        adata.uns["spatial"][library_id]["images"]["segmentation"],
        library_id=library_id,
        layer="segmentation",
    )
    img["segmentation"].attrs["segmentation"] = True
    imgs.append(img)

img = sq.im.ImageContainer.concat(imgs)

for target in ["point16", "point23", "point8"]:
    seg = img["segmentation"].sel(z=target).values
    uniq = np.unique(seg)
    n_cells = int((uniq != 0).sum())
    print(target, "-> distinct segmented cells:", n_cells)
