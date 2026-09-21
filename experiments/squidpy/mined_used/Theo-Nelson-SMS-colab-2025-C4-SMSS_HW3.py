# mined from: https://github.com/Theo-Nelson/SMS-colab/blob/3f075f87d274830891aad09aeb3ca91ea238c797/2025/C4/SMSS_HW3.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.nhood_enrichment, squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.nhood_enrichment, squidpy.pl.spatial_segment, squidpy.read.nanostring

# %%
# !pip install -U squidpy scanpy anndata matplotlib seaborn gdown python-igraph leidenalg --quiet
# !pip install "numcodecs<0.11.0" --quiet

# %%
import os
import tarfile

# Your shared file ID from google drive
file_id = "1ClBjGL3XANRmJteX1-bMRb3lPAmexFFG"
archive_path = "SMSS_data.tar.gz"
extract_dir = "smss_data"
os.makedirs(extract_dir, exist_ok=True)

# Download the archive
if not os.path.exists(archive_path):
    print("Downloading from Google Drive...")
#     !gdown --id {file_id} -O {archive_path}
else:
    print("File already downloaded.")

# Extract the archive
print("Extracting archive...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print("Extraction complete. Files inside:", os.listdir(extract_dir))

# %%
import squidpy as sq

adata = sq.read.nanostring(
    path="smss_data/Lung5_Rep1/Lung5_Rep1-Flat_files_and_images/",
    counts_file="Lung5_Rep1_exprMat_file.csv",
    meta_file="Lung5_Rep1_metadata_file.csv",
    fov_file="Lung5_Rep1_fov_positions_file.csv"
)

adata

# %%
import scanpy as sc

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)


# %%
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import anndata

# Create a 10x10 grid of cells with random expressions
grid_size = 10
coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
expression = np.random.uniform(0, 10, size=(grid_size * grid_size))

# Create AnnData
adata = anndata.AnnData(X=expression[:, None])
adata.obsm["spatial"] = coords
adata.var_names = ["Gene1"]

# Compute spatial neighbors and Moran's I
sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.spatial_autocorr(adata, mode="moran")

# Visualize expression
plt.figure(figsize=(5, 5))
plt.scatter(coords[:, 0], coords[:, 1], c=expression, cmap="coolwarm", s=100)
plt.gca().invert_yaxis()
plt.title("Simulated Expression on 10×10 Grid")
plt.colorbar(label="Gene1 Expression")
plt.axis("off")
plt.show()

# Show Moran's I score
adata.uns["moranI"]["I"]

# %%
# Spatial autocorrelation (Moran's I)
sq.gr.spatial_autocorr(adata, mode='moran')

# %%
lib = "1" # FOV number
adata.uns["spatial"][lib]["images"]["segmentation"] = adata.uns["spatial"][lib]["images"]["segmentation"].copy()
sq.pl.spatial_segment(adata, color=["COL9A2"], library_id=lib, img=False, seg_cell_id="cell_ID", library_key="fov")

# %%
# Neighborhood enrichment between clusters
sq.gr.nhood_enrichment(adata, cluster_key='leiden')
sq.pl.nhood_enrichment(adata, cluster_key='leiden')

# %%
# Co-occurrence score between clusters (optional advanced)
sq.gr.co_occurrence(adata, cluster_key='leiden')
sq.pl.co_occurrence(adata, cluster_key='leiden', clusters="9")
