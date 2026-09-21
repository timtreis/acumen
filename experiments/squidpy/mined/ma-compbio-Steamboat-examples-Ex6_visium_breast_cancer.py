# mined from: https://github.com/ma-compbio/Steamboat/blob/b9c1192f10232147b4b5db54f89d558b85a22367/examples/Ex6_visium_breast_cancer.ipynb
# symbols: squidpy.pl.spatial_scatter, squidpy.read.visium

# %%
import sys
sys.path.append("../") # Append the parent directory to Steamboat package, not needed if installed via pip

import scanpy as sc
import squidpy as sq
import pandas as pd
from tqdm.notebook import tqdm
import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import torch
import gc

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import steamboat as sf

device = "cuda" # this dataset is small and should also work with "cpu"


plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'

# %%
adata = sq.read.visium("../data/Ex6_visium_breast_cancer/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1")
annotation = pd.read_csv("../data/Ex6_visium_breast_cancer/BRCA1/metadata.tsv", sep="\t", index_col=0)
adata.obs = adata.obs.join(annotation)
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata.X = np.array(adata.X.todense())

# %%
# Prepare Steamboat dataset
adata.obs['global'] = 0
adatas = sf.prep_adatas([adata], n_neighs=6, norm=False, log1p=False, scale=False, renorm=False)
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'], mask_var='highly_variable')

# %%
if True: # Instead of training, you can load the trained model by setting this to False
    # Move dataset to GPU
    cuda_dataset = dataset.to(device)

    # Train Steamboat model
    sf.set_random_seed(0)
    model = sf.Steamboat(adatas[0].var_names[adatas[0].var['highly_variable']].tolist(), n_heads=20, n_scales=3)
    model = model.to(device)
    model.fit(cuda_dataset, entry_masking_rate=0.1, feature_masking_rate=0.1,
            max_epoch=10000, 
            loss_fun=torch.nn.MSELoss(reduction='sum'),
            opt=torch.optim.Adam, opt_args=dict(lr=0.01), stop_eps=1e-3, report_per=1000, stop_tol=200, device=device)
    
    # Save the trained model
    torch.save(model.state_dict(), "saved_models/visium_breast_cancer.pth")
else:
    model = sf.Steamboat(adatas[0].var_names[adatas[0].var['highly_variable']].tolist(), n_heads=20, n_scales=3)
    model = model.to(device)
    model.load_state_dict(torch.load("saved_models/visium_breast_cancer.pth"))

# %%
sf.tools.calc_obs(adatas, dataset, model, get_recon=True)

i = 0
sf.tools.neighbors(adatas[i], use_rep='attn')
sf.tools.leiden(adatas[i], resolution=0.6)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sq.pl.spatial_scatter(adatas[i], color=["fine_annot_type", "steamboat_clusters"], shape=None, fig=fig, ax=axes, size=2.5, frameon=False)

labels_pred = adata.obs['steamboat_clusters'].values
labels_true = adata.obs['fine_annot_type'].values
nmi = normalized_mutual_info_score(labels_true, labels_pred)
ari = adjusted_rand_score(labels_true, labels_pred)

n_clusters = adata.obs['steamboat_clusters'].nunique()

print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, n_clusters: {n_clusters}")

axes[0].set_title("Manual annotation", fontsize=11)
axes[1].set_title("Steamboat clusters", fontsize=11)
fig.tight_layout(pad=2.)
