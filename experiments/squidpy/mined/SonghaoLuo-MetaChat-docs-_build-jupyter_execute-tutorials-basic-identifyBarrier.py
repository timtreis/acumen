# mined from: https://github.com/SonghaoLuo/MetaChat/blob/6f40cf6e38e92bc1682aeb823fda5db1bca9a563/docs/_build/jupyter_execute/tutorials/basic/identifyBarrier.ipynb
# symbols: squidpy.im.ImageContainer

# %%
# Importing packages
import os
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import metachat as mc

# %%
# Setting your work directory
os.chdir("/Users/songhao/Library/CloudStorage/OneDrive-UCIrvine/2_Unpublished_work/1_Metabolite_Chat/4_Codes/tutorials/")

# %%
import sys
sys.path.append("/Users/songhao/Library/CloudStorage/OneDrive-UCIrvine/2_Unpublished_work/1_Metabolite_Chat/4_Codes/tutorials/MetaChat-main/")
import metachat_new as mc

# %%
adata = sc.read('datasets/mouse_small_intestine/adata_combined_subset.h5ad')
image = adata.uns['spatial']['Visium_HD_Mouse_Small_Intestine_lowres_image']['images']['lowres']
img = sq.im.ImageContainer(image, 
                           library_id='Visium_HD_Mouse_Small_Intestine_lowres_image', 
                           scale=adata.uns['spatial']['Visium_HD_Mouse_Small_Intestine_lowres_image']['scalefactors']['tissue_lowres_scalef'])

# %%
viewer = img.interactive(adata)

# %%
dict_scale = adata.uns['spatial']['Visium_HD_Mouse_Small_Intestine_lowres_image']['scalefactors']
scale = dict_scale['tissue_hires_scalef']/dict_scale['tissue_lowres_scalef']
barrier_segs = mc.pp.load_barrier_segments("datasets/mouse_small_intestine/barrier.csv", coord_cols=("axis-2", "axis-1"), scale=scale)

# %%
# Check barriers
clusters = adata.obs["Cluster"] if "Cluster" in adata.obs else None
XY = np.asarray(adata.obsm["spatial"], float)

fig, ax = plt.subplots(figsize=(5,5))
cats = pd.Categorical(clusters)
ax.scatter(XY[:,0], XY[:,1], s=8, c=cats.codes, cmap="tab20", edgecolor="none")

for (a, b) in barrier_segs:
    x = [a[0], b[0]]
    y = [a[1], b[1]]
    ax.plot(x, y, lw=2, c="black")

ax.set_aspect("equal", adjustable="box")
ax.invert_yaxis()
ax.set_title("Clusters with barrier segments")
plt.show()

# %%


# %%

