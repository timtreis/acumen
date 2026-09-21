# mined from: https://github.com/ratschlab/DeepSpot/blob/253c334f878da2ca1c558779dfffd7476e8687c5/example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_3_inference.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
os.chdir('../../')

# %%
from deepspot.utils.utils_image import predict_spot_spatial_transcriptomics_from_image_path
from deepspot.utils.utils_image import get_morphology_model_and_preprocess
from deepspot.utils.utils_image import crop_tile

from deepspot.spot import DeepSpot

import matplotlib.image as mpimg
from openslide import open_slide
import matplotlib.pyplot as plt
from tqdm import tqdm
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
n_mini_tiles = 9 # number of non-overlaping subspots
spot_diameter = 60 # spot diameter
spot_distance = 110 # distance between spots
white_cutoff = 200  # recommended, but feel free to explore
downsample_factor = 10 # downsampling the image used for visualisation in squidpy
model_weights = 'pretrained_model_weights/example_model/weights_Visium.pkl'
model_hparam = 'pretrained_model_weights/example_model/hparam_Visium.yaml'
gene_path = f"{out_folder}/data/info_highly_variable_genes_Visium.csv"
sample = 'ZEN38'
image_path = f'example_data/data/image/{sample}_without_fud.jpg'

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
image = pyvips.Image.new_from_file(image_path)

coord = []
for i, x in enumerate(range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)):
    for j, y in enumerate(range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)):
        coord.append([i, j, x, y])
coord = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
coord.index = coord.index.astype(str)

# %%
is_white = []
counts = []
for _, row in tqdm(coord.iterrows()):
    x = row.x_pixel - int(spot_diameter // 2)
    y = row.y_pixel - int(spot_diameter // 2)
    
    main_tile = crop_tile(image, x, y, spot_diameter)
    main_tile = main_tile[:,:,:3]
    white = np.mean(main_tile)
    is_white.append(white)

counts = np.empty((len(is_white), selected_genes_bool.sum())) # empty count matrix 

coord['is_white'] = is_white

# %%
adata = ad.AnnData(counts)
adata.var.index = genes[selected_genes_bool].gene_name.values
adata.obs = adata.obs.merge(coord, left_index=True, right_index=True)
adata.obs['is_white'] = coord['is_white'].values
adata.obs['is_white_bool'] = (coord['is_white'].values > white_cutoff).astype(int)
adata.obs['sampleID'] = sample
adata.obs['barcode'] = adata.obs.index
adata = adata[adata.obs.is_white_bool == 0, ]
adata

# %%
### CREATE IMAGE
img = open_slide(image_path)
n_level = len(img.level_dimensions) - 1 # 0 based


large_w, large_h = img.dimensions
new_w = math.floor(large_w / downsample_factor)
new_h = math.floor(large_h / downsample_factor)

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

# %%


# %%
# Load the YAML file into a regular Python dictionary
with open(model_hparam, 'r') as yaml_file:
    model_hparam = yaml.safe_load(yaml_file)
model_hparam

# %%
model_expression = torch.load(model_weights, map_location=device)
model_expression.to(device)
model_expression.eval()

# %%
morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name=image_feature_model, device=device)
morphology_model.to(device)
morphology_model.eval()

# %%
counts = predict_spot_spatial_transcriptomics_from_image_path(image_path, 
                                                        adata,
                                                        spot_diameter,
                                                        n_mini_tiles,
                                                        preprocess, 
                                                        morphology_model, 
                                                        model_expression, 
                                                        device,
                                                        super_resolution=False,
                                                        neighbor_radius=1)

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
adata_predicted

# %%
sq.pl.spatial_scatter(adata_predicted, 
                      color=['MUC2', 'ITLN1', 
                             'CLCA1', 'FCGBP'], 
                      wspace=0,
                      ncols=2,
                      size=5)

# %%


# %%


# %%


# %%

