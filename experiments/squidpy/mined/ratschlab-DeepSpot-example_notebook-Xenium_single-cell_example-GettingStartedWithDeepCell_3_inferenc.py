# mined from: https://github.com/ratschlab/DeepSpot/blob/253c334f878da2ca1c558779dfffd7476e8687c5/example_notebook/Xenium_single-cell_example/GettingStartedWithDeepCell_3_inference_xenium.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
os.chdir('../../')

# %%
from deepspot.utils.utils_image import predict_cell_spatial_transcriptomics_from_image_path
from deepspot.utils.utils_image import get_morphology_model_and_preprocess
from deepspot.utils.utils_image import crop_tile

from deepspot.cell import DeepCell

import matplotlib.image as mpimg
from openslide import open_slide
import matplotlib.pyplot as plt
from tqdm import tqdm
import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import pyvips
import torch
import math
import yaml
import PIL

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
out_folder = "example_data"
image_feature_model = 'inception' 
cell_diameter = 20 # cell diameter
n_neighbors = 45 # the n_neighbors used to compute the neighbors around 
downsample_factor = 10 # downsampling the image used for visualisation in squidpy
model_weights = 'pretrained_model_weights/example_model/weights_Xenium.pkl'
model_hparam = 'pretrained_model_weights/example_model/hparam_Xenium.yaml'
gene_path = f"{out_folder}/data/info_highly_variable_genes_Xenium.csv"
sample = 'NCBI858'
image_path = f'example_data/data/image/{sample}.jpg'

# %%
genes = pd.read_csv(gene_path)
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]
genes_to_predict.sort_values("highly_variable_rank")

# %%
# Load the image
image = mpimg.imread(image_path)

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off axis labels
plt.show()

# %%
coord = sc.read_h5ad(f"example_data/data/h5ad/{sample}.h5ad").obs[["x_pixel", "y_pixel"]].copy()
coord

# %%
counts = np.empty((len(coord), selected_genes_bool.sum())) # empty count matrix 
adata = ad.AnnData(counts).copy()
adata.obs.index = coord.index
adata.var.index = genes[selected_genes_bool].gene_name.values
adata.obs = adata.obs.merge(coord, left_index=True, right_index=True)
adata.obs['sampleID'] = sample
adata.obs['barcode'] = adata.obs.index
adata

# %%
### CREATE IMAGE
img = open_slide(image_path)
n_level = len(img.level_dimensions) - 1 # 0 based


large_w, large_h = img.dimensions
new_w = math.floor(large_w / downsample_factor)
new_h = math.floor(large_h / downsample_factor)
print(large_w, large_h, new_w, new_h)
whole_slide_image = img.read_region((0, 0), n_level, img.level_dimensions[-1])
whole_slide_image = whole_slide_image.convert("RGB")
img_downsample = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)


adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
# adjust coordinates to new image dimensions
adata.obsm['spatial'] = adata.obsm['spatial'] / downsample_factor
# create 'spatial' entries
adata.uns['spatial'] = dict()
adata.uns['spatial']['library_id'] = dict()
adata.uns['spatial']['library_id']['images'] = dict()
adata.uns['spatial']['library_id']['images']['hires'] = np.array(img_downsample)
img_downsample.width, img_downsample.height

# %%
# Load the YAML file into a regular Python dictionary
with open(model_hparam, 'r') as yaml_file:
    model_hparam = yaml.safe_load(yaml_file)
model_hparam

# %%
model_expression = torch.load(model_weights, map_location=device)
model_expression.to(device)
model_expression.eval()
""

# %%
morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name=image_feature_model, device=device)
morphology_model.to(device)
morphology_model.eval()
""

# %%
counts = predict_cell_spatial_transcriptomics_from_image_path(image_path, 
                                                        adata,
                                                        cell_diameter,
                                                        n_neighbors,
                                                        preprocess, 
                                                        morphology_model, 
                                                        model_expression, 
                                                        device)

# %%
counts = model_expression.inverse_transform(counts)
counts

# %%
counts[counts < 0] = 0

# %%
adata_predicted = ad.AnnData(counts, 
                             var=adata.var,
                             obs=adata.obs, 
                             uns=adata.uns, 
                             obsm=adata.obsm).copy()
adata_predicted.obs

# %%
sq.pl.spatial_scatter(adata_predicted, 
                      color=['CCL18', 'POSTN', 
                             'PGC', 'LAMP3'], 
                      wspace=0,
                      ncols=2,
                      size=1)

# %%

