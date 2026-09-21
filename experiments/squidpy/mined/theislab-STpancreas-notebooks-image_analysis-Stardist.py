# mined from: https://github.com/theislab/STpancreas/blob/6f9ca0d5ab38422d24db00e8fb4925a856bb6026/notebooks/image_analysis/Stardist.ipynb
# symbols: squidpy.im.ImageContainer, squidpy.im.segment

# %%
from stardist.models import StarDist2D
from csbdeep.utils import normalize

import squidpy as sq
import numpy as np
import matplotlib.pyplot as plt
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 1029959493

# %%
img = sq.im.ImageContainer(
    "/Volumes/external-HDD/STpancreas/dat/SP01-2020_GE_HE_JSON/GE1_V19S18-084_CA1A.jpg",
    library_id="spaceranger",
)

# %%
StarDist2D.from_pretrained("2D_versatile_he")

# %%
def stardist_2D_versatile_fluo(img, nms_thresh=None, prob_thresh=None):
    img = normalize(img, 1, 99.8, axis=(0,1))
    model = StarDist2D.from_pretrained('2D_versatile_he')
    labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh)
    return labels

# %%
for i in np.arange(0.3,1.6,0.1):
    for j in np.arange(0.3,1.6,0.1):
        print(f"Using nms threshold {i} and prob threshold {j}")
        sq.im.segment(
                    img=img,
                    layer="image",
                    channel=0,
                    method=stardist_2D_versatile_fluo,
                    layer_added=f'segmented_stardist_nms{i}_prob{j}',
                    nms_thresh=i,
                    prob_thresh=j
                    )

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 20))
img.show("image", channel=None, ax=axes[0])
_ = axes[0].set_title("H&E")
img.show("segmented_stardist_", cmap="jet", interpolation="none", ax=axes[1])#choose the value of nms and prob threshold to plot from the grid search
_ = axes[1].set_title("Stardist segmentation")

# %%

