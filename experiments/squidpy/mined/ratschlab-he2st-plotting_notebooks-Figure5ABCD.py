# mined from: https://github.com/ratschlab/he2st/blob/17087753ce410a9fbd928255679c8c42c088bc58/plotting_notebooks/Figure5ABCD.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import squidpy as sq
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt 
import plotnine as p9

import sys
sys.path.append('../')
from src.utils import load_data

# %%
translate = {
    "TUM": "Tumor",
    "NOR": "Normal\nlymphoid",
    "UNASSIGNED": "Unassigned",
    "TLS": "TLS",
    "INFL": "Inflamated"
}

# %%
dataset = "USZ"

# %%
with open(f"../{dataset}/config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)

models = config_dataset["MODEL"]
all_samples = set(config_dataset["SAMPLE_LQ"])
known_genes = np.array(config_dataset["known_genes"])
top_n_genes_to_predict = int(config_dataset["top_n_genes_to_predict"])
top_n_genes_to_predict

# %%
out_folder = "out_benchmark"
genes = pd.read_csv(f"../{dataset}/{out_folder}/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]
genes_to_predict

# %%
sample = 'KC2'
model = "DeepSpot"
adata_true = sc.read_h5ad(f"../{dataset}/out_benchmark/data/h5ad/{sample}.h5ad")
sc.pp.normalize_total(adata_true, target_sum=1e4)
sc.pp.log1p(adata_true)
adata_true = adata_true[:,adata_true.var.index.isin(genes_to_predict.gene_name)]
sc.pp.log1p(adata_true)
adata_true.var["method"] = "Visium, 10x Genomics"
adata_true.obs["method"] = "Visium, 10x Genomics"
adata_true.obs["sample_id"] = sample

adata_pred = sc.read_h5ad(f"../{dataset}/out_benchmark/prediction/{model}/data/h5ad/{sample}.h5ad")

adata_pred.X[adata_pred.X < 0] = 0

adata_pred.raw = adata_pred

adata_pred.obs["method"] = model
adata_pred.var["method"] = model
adata_pred.obs["sample_id"] = sample

    

# %%
adatas_pred = adatas_pred[adatas_pred.obs.aestetik_manual_anno != "UNASSIGNED"].copy()
adatas_true = adatas_true[adatas_true.obs.aestetik_manual_anno != "UNASSIGNED"].copy()

# %%
adata_pred.obs["Pathology annotation"] = adata_pred.obs.aestetik_manual_anno.apply(lambda x: translate[x])
adata_true.obs["Pathology annotation"] = adata_true.obs.aestetik_manual_anno.apply(lambda x: translate[x])

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


adata_example = ad.concat((adata_true, adata_pred), axis=1, merge="same", uns_merge="same").copy()
adata_example.var.index = [f"{row.name} {row.method}" for _, row in adata_example.var.iterrows()]
adata_example.var

# %%
pad = 400
x, y = 1724.857175,  567.53425 

bounds = (x - 100, 
              y - 100,
               x + 600,
               y + 600)

adata_example.obs["H&E image"] = np.nan
adata_example.obs["Pathology annotation"] = adata_true.obs.aestetik_manual_anno.apply(lambda x: translate[x])

# %%
sq.pl.spatial_scatter(adata_example, 
                      color=["H&E image", "Pathology annotation"], 
                      img_alpha=0.9, 
                      crop_coord=bounds, 
                      wspace=0, 
                      hspace=0.1,
                      size=10,      
                      ncols=1, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure5A-tls_{sample}_h&e_anno.png", 
                      dpi=300,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(7, 6))

# %%
genes = ["LTB", "CXCL13", "MS4A1", "CCL19"]
color = [f"{g} {m}" for m in ['Visium, 10x Genomics', 'DeepSpot'] for g in genes]

# %%
plt.rcParams.update({'font.size': 20})
sq.pl.spatial_scatter(adata_example, 
                      color=color, 
                      img_alpha=0.9,
                      crop_coord=bounds, 
                      wspace=0.1, 
                      hspace=0.1,
                      size=10,      
                      ncols=len(genes), 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure5B_tls_{sample}_{'_'.join(genes)}.png", 
                      dpi=300,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(7, 7))

# %%
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import squidpy as sq
import scanpy as sc
import numpy as np
import pyvips
import math

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def get_spot(image, x, y, spot_diameter_fullres):
    x = x - int(spot_diameter_fullres // 2)
    y = y - int(spot_diameter_fullres // 2)
    spot = image.crop(x, y, spot_diameter_fullres, spot_diameter_fullres)
    spot_array = np.ndarray(buffer=spot.write_to_memory(),
                            dtype=format_to_dtype[spot.format],
                            shape=[spot.height, spot.width, spot.bands])
    return spot_array

def compute_centroid(adata, topN=5):
    adata = adata.copy()
    sc.pp.pca(adata)
    nc = NearestCentroid()
    nc.fit(adata.obsm["X_pca"], adata.obs["Pathology annotation"])

    dist_from_centroid = cdist(nc.centroids_, adata.obsm["X_pca"])

    adata.obs["centroid"] = np.nan

    topN_centroid_idx = np.argpartition(dist_from_centroid, topN, axis=1)[
        :, :topN].reshape(-1, order="F")
    topN_centroid_label = np.tile(nc.classes_, topN)

    return topN_centroid_idx

def plot_spots(img_path, adata, indeces_to_plot, spot_diameter_fullres, label=None):
    image = pyvips.Image.new_from_file(img_path)
    tab = adata.obs.iloc[indeces_to_plot]

    # Determine the number of labels and layout
    n_labels = np.unique(tab[label]).size
    columns = min(10, n_labels)
    rows = math.ceil(len(tab) / columns)

    # Create the figure with tighter layout
    fig = plt.figure(figsize=(columns * 3, rows * 3))
    
    # Loop through each spot to plot
    for i in range(len(tab)):
        row = tab.iloc[i]
        img = get_spot(image, row.y_pixel, row.x_pixel, spot_diameter_fullres)
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(img)
        ax.set_title(f"{row[label]}", fontsize=15)
        ax.axis('off')
    # Remove spacing between subplots
    plt.subplots_adjust(wspace=-0.5, hspace=0)
    # Use tight layout to minimize spacing
    plt.tight_layout(pad=0)
    

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

img_path = f'../{dataset}/out_benchmark/data/image/{sample}.tif'

tab = adata_pred[~adata_pred.obs["Pathology annotation"].isin(["Inflamated", "Unassigned"])].copy()

topN_centroid_idx = compute_centroid(tab, 
                                     topN=2)

plot_spots(img_path,
            tab,
            topN_centroid_idx,
            100,
            "Pathology annotation")

# %%
selected_genes = ["LTB", "BLK", "CD19", "CXCL13", "MS4A1", "SPIB", "CORO1A", "CCL19"]#, "DUSP9", 'KREMEN1']

# %%
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import seaborn as sns

# %%
df_pred = pd.DataFrame(adata_pred.X, columns=adata_pred.var.index)
df_true = pd.DataFrame(adata_true.X, columns=adata_true.var.index)

# Compute the correlation matrices
corr_true = df_true[selected_genes].corr("pearson").fillna(0)
corr_pred = df_pred[selected_genes].corr("pearson").fillna(0)

def cluster_corr_matrix(corr_matrix):
    # Compute the distance matrix
    distance_matrix = 1 - corr_matrix.abs()
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Get the ordered indices
    order = leaves_list(linkage_matrix)
    
    # Reorder the correlation matrix
    return corr_matrix.iloc[order, order], order

# Cluster the correlation matrices
clustered_corr_true, order_true = cluster_corr_matrix(corr_true)
clustered_corr_pred, order_pred = cluster_corr_matrix(corr_pred)

def extract_upper_triangle(matrix):
    # Create a boolean mask for the upper triangle
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    return matrix.where(mask)

def extract_lower_triangle(matrix):
    # Create a boolean mask for the lower triangle
    mask = np.tril(np.ones_like(matrix, dtype=bool))
    return matrix.where(mask)

# Extract upper triangle of the ground truth matrix
upper_corr_true = extract_upper_triangle(clustered_corr_true)

# Extract lower triangle of the predicted matrix
lower_corr_pred = extract_lower_triangle(clustered_corr_pred)

# %%
np.fill_diagonal(lower_corr_pred.values, np.nan)
np.fill_diagonal(upper_corr_true.values, np.nan)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
# Calculate the min and max values for scaling
min_val = min(upper_corr_true.min().min(), lower_corr_pred.min().min())
max_val = max(upper_corr_true.max().max(), lower_corr_pred.max().max())

# Set up the matplotlib figure
plt.figure(figsize=(8, 10))

# Plot for the ground truth (upper triangle)
plt.subplot(2, 1, 1)
sns.heatmap(upper_corr_true, cmap='viridis', vmin=min_val, vmax=max_val)
plt.title("Pearson correlation Visium, 10x Genomics")

# Plot for the predictions (lower triangle)
plt.subplot(2, 1, 2)
sns.heatmap(lower_corr_pred, cmap='viridis', vmin=min_val, vmax=max_val)
plt.title("Pearson correlation DeepSpot")

# Adjust layout
plt.tight_layout()
plt.savefig("figures/Figure5C-tls_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
import numpy as np
from IPython.display import Image
import pyvips
import openslide
import sys
sys.path.append('../')
from src.preprocess_utils.preprocess_image import get_low_res_image
from deepspot.utils.utils_image import predict_spatial_transcriptomics_from_image_path

from src.morphology_model import get_morphology_model_and_preprocess
import matplotlib.pyplot as plt 
import torch
from openslide import open_slide
import pandas as pd
import scanpy as sc
import anndata as ad
import math
import PIL
import squidpy as sq
from tqdm import tqdm
import json
import yaml
from pickle import load

from src.utils import preprocess_adata
from sklearn.preprocessing import MinMaxScaler
import glob

# %%
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# %%
model = "DeepSpot"
sample = 'B07-30616-7'
out_folder = "out_benchmark"
source_data_path = "UZH"

# %%
adata = sc.read_h5ad(f"../TLS_data/out_benchmark/data/h5ad/{sample}.h5ad")
adata.obs.query("x_array == 10")

# %%
def compute_transcriptomics(model, sample, genes, selected_genes_bool, white_cutoff=200, spot_distance=500, 
                            super_resolution=False):

    image_path = f"../{source_data_path}/{out_folder}/data/image/{sample}.tif"
    json_path  = f"../{source_data_path}/{out_folder}/data/meta/{sample}.json"
    spot_diameter = int(json.load(open(json_path))["spot_diameter_fullres"])
    
    
    image = pyvips.Image.new_from_file(image_path)
    
    coord = []
    for i, x in enumerate(range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)):
        for j, y in enumerate(range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)):
            coord.append([i, j, x, y])
    coord = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
    coord.index = coord.index.astype(str)
    
    is_white = []
    counts = []
    for _, row in tqdm(coord.iterrows()):
        x = row.x_pixel - int(spot_diameter // 2)
        y = row.y_pixel - int(spot_diameter // 2)
        
        spot = image.crop(y, x, spot_diameter, spot_diameter)
        main_tile = np.ndarray(buffer=spot.write_to_memory(),
                          dtype=format_to_dtype[spot.format],
                          shape=[spot.height, spot.width, spot.bands])
        main_tile = main_tile[:,:,:3]
        white = np.mean(main_tile)
        is_white.append(white)
    
    counts = np.empty((len(is_white), selected_genes_bool.sum())) # empty count matrix 
    
    coord['is_white'] = is_white

    ### CREATE ANNDATA
    
    adata = ad.AnnData(counts)
    adata.var.index = genes[selected_genes_bool].gene_name.values
    adata.obs = adata.obs.merge(coord, left_index=True, right_index=True)
    adata.obs['is_white'] = coord['is_white'].values
    adata.obs['is_white_bool'] = (coord['is_white'].values > white_cutoff).astype(int)
    adata.obs['barcode'] = adata.obs.index
    adata.obs["sampleID"] = "dummy"
    adata = adata[adata.obs.is_white_bool == 0, ]
    
    model_path = f'../{source_data_path}/out_benchmark/evaluation/{model}/final_model.pkl'
    model_config_path = f'../{source_data_path}/out_benchmark/evaluation/{model}/top_param_overall.yaml'
    adata_in = f'../{dataset}/{out_folder}/data/h5ad/{sample}.h5ad'
    adata_out = f'../{dataset}/{out_folder}/prediction/{model}/data/h5ad/{sample}.h5ad'
    json_path = f'../{dataset}/{out_folder}/data/meta/{sample}.json'
    
    if model in ["THItoGene", "HisToGene", "Hist2ST"]:
         device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    json_info = json.load(open(json_path))
    image_path = json_info['image_path'] if "image_path" in json_info else glob.glob(f"../{dataset}/{out_folder}/data/image/{sample}*")[0]
    spot_diameter = json_info['spot_diameter_fullres']
    
    genes = pd.read_csv(f"../{source_data_path}/out_benchmark/info_highly_variable_genes.csv")
    selected_genes_bool = genes.isPredicted.values
    
    with open(model_config_path, "r") as stream:
        MODEL_PARAM = yaml.safe_load(stream)
        
    image_feature_model = MODEL_PARAM.get("image_feature_model", None)
    spot_context = MODEL_PARAM.get('spot_context', None)
    top_k = MODEL_PARAM.get('top_k', None)
    
    with open(f"../{source_data_path}/config_dataset.yaml", "r") as stream:
        config_dataset_source = yaml.safe_load(stream)
    
    n_mini_tiles = config_dataset_source['n_mini_tiles'] if not n_mini_tiles else 1
    training_samples = config_dataset_source.get("SAMPLE", None)

    ### LOAD EXPRESSION MODEL
    
    if model in ["MLP", "DeepSpot", "HisToGene", "THItoGene", "Hist2ST", "BLEEP"]: # pytorch
        model_expression = torch.load(model_path, map_location=device)
        model_expression.to(device)
        model_expression.eval()
    else:
        with open(model_path, 'rb') as f:
            model_expression = load(f)
    
    
    ### LOAD MORPHOLOGY MODEL
    
    if image_feature_model:
        morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name=image_feature_model, device=device)
        morphology_model.to(device)
        morphology_model.eval()
    else:
        fig_size = model_expression.fig_size
        import torchvision.transforms as transforms
        preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(fig_size),
                ])
    
    counts = predict_spatial_transcriptomics_from_image_path(image_path, 
                                                    adata,
                                                    spot_diameter,
                                                    n_mini_tiles,
                                                    preprocess, 
                                                    morphology_model, 
                                                    model_expression, 
                                                    device,
                                                    super_resolution=super_resolution,
                                                    neighbor_radius=1)
    
    adata_predicted = ad.AnnData(counts, obs=adata.obs, uns=adata.uns, obsm=adata.obsm).copy()
    adata_predicted.var.index = genes[selected_genes_bool].gene_name.values
    adata_predicted.obsm['spatial'] = adata_predicted.obs[["y_pixel", "x_pixel"]].values
    # adjust coordinates to new image dimensions
    adata_predicted.obsm['spatial'] = adata_predicted.obsm['spatial']
    # create 'spatial' entries
    adata_predicted.uns['spatial'] = dict()
    adata_predicted.uns['spatial']['library_id'] = dict()
    adata_predicted.uns['spatial']['library_id']['images'] = dict()
    adata_predicted.uns['spatial']['library_id']['images']['hires'] = np.array(image)
    return adata_predicted

# %%
out_folder = "out_benchmark"
genes = pd.read_csv(f"../{dataset}/{out_folder}/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]
genes_to_predict

# %%
adata_predicted_low = compute_transcriptomics(model, sample, genes, selected_genes_bool, 
                                              spot_distance=130)
adata_predicted_low.write_h5ad("figures/Figure5D-cells_low.h5ad")

# %%
adata_predicted_high = compute_transcriptomics(model, sample, genes, selected_genes_bool, 
                                               spot_distance=30, super_resolution=True)
adata_predicted_high.write_h5ad("figures/Figure5D-cells_high.h5ad")

# %%
adata_predicted_high = sc.read_h5ad("figures/Figure5D-cells_high.h5ad")
adata_predicted_low = sc.read_h5ad("figures/Figure5D-cells_low.h5ad")

# %%
adata_predicted_high.X[adata_predicted_high.X < 0] = 0
adata_predicted_low.X[adata_predicted_low.X < 0] = 0

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

pad = -10
bounds = (adata_predicted_low.obsm["spatial"][:, 0].min() - pad * 150,
              adata_predicted_low.obsm["spatial"][:, 1].min() - pad * 150,
              adata_predicted_low.obsm["spatial"][:, 0].max() + pad,
              adata_predicted_low.obsm["spatial"][:, 1].max() + pad * 50)
bounds

# %%
sample = 'B07-30616-7'
model = "DeepSpot"
#pad = 400
#y, x = 2294,  6974

#bounds = (x - 500, #left
#              y - 500, # up
#               x + 2400, # right
#               y + 2400) # down
#bounds
# x0=1624.0, y0=1467.0, x1=2324.0, y1=2167.0

sq.pl.spatial_scatter(adata_predicted_low, 
                      img_alpha=0.9, 
                      crop_coord=bounds, 
                      wspace=0, 
                      hspace=0.1,
                      color=["LTB"],
                      title="LTB - spot distance 130px; spot diameter 81px",
                      size=50,      
                      ncols=1, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/5D1-{sample}_{model}_ltb_low.png", 
                      dpi=300,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(7, 5))

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

sq.pl.spatial_scatter(adata_predicted_high, 
                      img_alpha=0.9, 
                      #alpha=0.9,
                      crop_coord=bounds, 
                      wspace=0, 
                      hspace=0.1,
                      color=["LTB"],
                      title="LTB - spot distance 30px; spot diameter 27px",
                      size=12,      
                      ncols=1, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/5D2-{sample}_{model}_ltb_high.png", 
                      dpi=300,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(8, 5))

# %%

