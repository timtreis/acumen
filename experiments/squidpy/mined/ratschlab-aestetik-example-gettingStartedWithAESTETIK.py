# mined from: https://github.com/ratschlab/aestetik/blob/932bb7ad91f64562b29d54830a95f05a2658bda2/example/gettingStartedWithAESTETIK.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
# Import pyvips first to lock in libvips' libjpeg/libtiff stack
# before scanpy / torch load their own copies (see #14).
import pyvips  # noqa: F401

# %%
from pathlib import Path

# Resolve the repository root (parent of the example/ directory).
REPO_ROOT = Path.cwd() if (Path.cwd() / "test_data").is_dir() else Path.cwd().parent
TEST_DATA = REPO_ROOT / "test_data"

# %%
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
import squidpy as sq
import scanpy as sc
import torch
import json

# %%
from aestetik.utils.utils_morphology import extract_morphology_embeddings
from aestetik.utils.utils_transcriptomics import preprocess_adata
from aestetik.utils.utils_visualization import visualize
from aestetik import AESTETIK
AESTETIK.version()

# %%
import logging
# Configure the logging module
logging.basicConfig(level=logging.INFO)  # Set the desired logging level
logging.getLogger("pyvips").setLevel(logging.CRITICAL)

# %%
img_path = str(TEST_DATA / "151676.png")
adata_in = str(TEST_DATA / "151676.h5ad")
json_path = str(TEST_DATA / "151676.json")

# %%
n_components = 15
spot_diameter_fullres = json.load(open(json_path))["spot_diameter_fullres"]
dot_size = json.load(open(json_path))["dot_size"]

print(f"spot_diameter_fullres: {spot_diameter_fullres}")
print(f"dot_size: {dot_size}")

# %%
adata = sc.read_h5ad(adata_in)
#adata = adata[adata.obs.sample(100).index,:] # to speed up, we only select 100 spots.
adata = preprocess_adata(adata)

print(adata)

# %%
adata.obs.head()

# %%
weights = Inception_V3_Weights.DEFAULT
morphology_model = inception_v3(weights=weights)
morphology_model.fc = torch.nn.Identity()

morphology_model.eval()    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
feature_dim = 2048

preprocess = transforms.Compose([
                transforms.ToTensor(),
                weights.transforms(antialias=True),
            ])

features_inception_v3 = extract_morphology_embeddings(img_path, 
                                             morphology_model,
                                             x_pixel=adata.obs.y_pixel, 
                                             y_pixel=adata.obs.x_pixel, 
                                             spot_diameter=spot_diameter_fullres,
                                             device=device,
                                             n_components=n_components,
                                             feature_dim=feature_dim,
                                             preprocess=preprocess,
                                             apply_pca=True)

# %%
# we set the transcriptomics modality
adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"][:,0:n_components]

print(f"{adata.obsm['X_pca_transcriptomics'].shape[0]} spots x {adata.obsm['X_pca_transcriptomics'].shape[1]} transcriptomics PCA components")

# %%
# we set the morphology modality
adata.obsm["X_pca_morphology"] = features_inception_v3[:,0:n_components]

print(f"{adata.obsm['X_pca_morphology'].shape[0]} spots x {adata.obsm['X_pca_morphology'].shape[1]} morphology PCA components")

# %%
parameters =    {'morphology_weight': 0,
                 'refine_cluster': True,
                 'window_size': 7
                }
parameters

# %%
model = AESTETIK(n_cluster=adata.obs.ground_truth.unique().size,
                 **parameters)

# %%
model.fit(adata)

# %%
# sklearn-style: transform returns the embedding, predict returns the labels.
adata.obsm['AESTETIK'] = model.transform(adata)
adata.obs['AESTETIK_cluster'] = model.predict(adata)

# Modality-only baselines (computed during model.fit) — useful
# for the comparison plot below.
adata.obs['X_pca_transcriptomics_cluster'] = model.transcriptomics_cluster_
adata.obs['X_pca_morphology_cluster'] = model.morphology_cluster_

# %%
adata

# %%
sq.pl.spatial_scatter(adata, color=["ground_truth", 
                                    "X_pca_transcriptomics_cluster",
                                    "X_pca_morphology_cluster",
                                    "AESTETIK_cluster"], size=dot_size)

# %%
visualize(model=model,
          plot_loss=True)
