# mined from: https://github.com/ratschlab/st-rep/blob/3246895f76b5d12c65b6dc3828bde98e73929148/workflows/analysis/vizualizeClusters.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
#os.chdir('../../10x_tupro/')
#os.chdir('../../maynard_human_brain_analysis/')
#os.chdir('../../her2_positive_breast_tumors/')

# %%
from sklearn.metrics.cluster import adjusted_rand_score
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np 
import glob
import json
import yaml

# %%
from time import gmtime, strftime
strftime("%Y-%m-%d %H:%M:%S", gmtime())

# %%
with open("info.yaml", "r") as stream:
    INFO = yaml.safe_load(stream)

MODEL_R = INFO["MODEL_R"] if INFO["MODEL_R"] else list()
MODEL_PYTHON = INFO["MODEL_PYTHON"] if INFO["MODEL_PYTHON"] else list()
MODEL_FINE_TUNE = INFO["MODEL_FINE_TUNE"] if INFO["MODEL_FINE_TUNE"] else list()

MODELS = [*MODEL_R, *MODEL_PYTHON, *MODEL_FINE_TUNE]
MODELS

# %%
sample = "151673"
out_folder = "out_benchmark"

# %%
adata_in = f"{out_folder}/data/h5ad/{sample}.h5ad"
json_path = f"{out_folder}/data/meta/{sample}.json"
dot_size = float(json.load(open(json_path))["dot_size"])
dot_size

# %%
best_split_sample = pd.read_csv(f"{out_folder}/summary/summary_best_split_sample.csv")
best_split_sample["sample"] = best_split_sample["sample"].astype(str)
best_split_sample = best_split_sample[best_split_sample.model.isin(MODELS)]
best_split_sample = best_split_sample.query(f"sample == '{sample}'")
best_split_sample = best_split_sample.sort_values("ari")
best_split_sample

# %%
cluster_paths = best_split_sample.path.values
cluster_paths

# %%
adata = sc.read(adata_in)
adata.obs["Barcode"] = adata.obs.index
adata

# %%
def add_model_cluster(cluster_path):
    df = pd.read_csv(cluster_path)
    cluster_label_dict = pd.Series(df[df.columns[1]].values, index=df[df.columns[0]].values).to_dict()
    adata.obs[cluster_path.split("/")[2].replace("_evaluate", "")] = adata.obs.Barcode.apply(lambda x: cluster_label_dict[x]).astype(str)

# %%
for cluster_path in cluster_paths:
    add_model_cluster(cluster_path)

# %%
adata.obs

# %%
color = ["ground_truth", *best_split_sample.model]

# %%
title = ["ground_truth", *[f"{model} ARI: {adjusted_rand_score(adata.obs['ground_truth'].values, adata.obs[model].values):.2f}" for model in best_split_sample.model]]

# %%
sq.pl.spatial_scatter(adata, img_alpha=0.8, color=color, 
                      size=dot_size, ncols=6, title=title,
                      frameon=False, wspace=0, dpi=300)

# %%

