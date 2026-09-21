# mined from: https://github.com/theislab/squidpy_reproducibility/blob/407b151c1b7d657dd46a99cd9d924b13ec2d9afa/notebooks/supp_figures/benchmarks/benchmark_image.ipynb
# symbols: squidpy.datasets.visium_fluo_adata_crop, squidpy.datasets.visium_fluo_image_crop, squidpy.datasets.visium_hne_adata, squidpy.datasets.visium_hne_adata_crop, squidpy.datasets.visium_hne_image, squidpy.datasets.visium_hne_image_crop, squidpy.im.calculate_image_features, squidpy.im.segment

# %%
import squidpy as sq
import scanpy as sc
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

# %%
# load data
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()

# %%
# define workflow
def time_image_workflow(img, adata, n_jobs=1):
    start = time.time()
    #sq.im.process(img, method="smooth", sigma=2)
    sq.im.segment(img, layer="image", method="watershed", thresh=None, n_jobs=n_jobs, size=2000)
    sq.im.calculate_image_features(adata,img,layer="image",key_added='features', 
                               features=['summary', 'histogram'],
                               n_jobs=n_jobs, 
                               spot_scale=1, 
                               scale=1.0, 
                               mask_circle=True)
    end = time.time()
    return end - start

# %%
res = []

# %%
# calculate execution time for different datasets

dataset = 'fluo_crop'
img = sq.datasets.visium_fluo_image_crop()
adata = sq.datasets.visium_fluo_adata_crop()
for run in range(3):
    for n_jobs in [1,2,3,4]:
        duration = time_image_workflow(img, adata, n_jobs=n_jobs)
        res_dict = {
            'dataset': dataset,
            'n_pixels': img['image'].size,
            'shape': str(img['image'].shape),
            'n_jobs': n_jobs,
            'run': run,
            'time': duration,
        }
        res.append(res_dict)

# %%
dataset = 'hne_crop'
img = sq.datasets.visium_hne_image_crop()
adata = sq.datasets.visium_hne_adata_crop()
for run in range(3):
    for n_jobs in [1,2,3,4]:
        duration = time_image_workflow(img, adata, n_jobs=n_jobs)
        res_dict = {
            'dataset': dataset,
            'n_pixels': img['image'].size,
            'shape': str(img['image'].shape),
            'n_jobs': n_jobs,
            'run': run,
            'time': duration,
        }
        res.append(res_dict)

# %%
dataset = 'hne'
img = sq.datasets.visium_hne_image()
adata = sq.datasets.visium_hne_adata()
for run in range(3):
    for n_jobs in [1,2,3,4]:
        duration = time_image_workflow(img, adata, n_jobs=n_jobs)
        res_dict = {
            'dataset': dataset,
            'n_pixels': img['image'].size,
            'shape': str(img['image'].shape),
            'n_jobs': n_jobs,
            'run': run,
            'time': duration,
        }
        res.append(res_dict)

# %%
df = pd.DataFrame(res)

# %%
df

# %%
df.to_csv('figures/feature_extraction_benchmark.csv')

# %%
df = pd.read_csv('figures/feature_extraction_benchmark.csv', index_col=0)
df

# %%
df_grouped = df.groupby(['dataset','n_jobs']).mean().reset_index(drop=False)

# %%
df_grouped

# %%
df_grouped.loc[df_grouped['dataset'] == 'fluo_crop', 'dataset'] = 'fluo small ($16\cdot10^7$ px)'
df_grouped.loc[df_grouped['dataset'] == 'hne', 'dataset'] = 'H&E large ($40\cdot10^7$ px)'
df_grouped.loc[df_grouped['dataset'] == 'hne_crop', 'dataset'] = 'H&E small ($4\cdot10^7$ px)'

# %%
fig, ax = plt.subplots(1,1, figsize=(5,3), dpi=180, tight_layout=True)
sns.lineplot(data=df_grouped, hue='dataset', x='n_jobs', y='time', marker='o', ax=ax)
ax.set_xticks([1,2,3,4])
ax.set_ylabel("runtime [s]")
ax.set_xlabel("#tasks")
ax.get_legend().set_title(None)
plt.grid()
_ = ax.set_title("runtime of feature extraction workflow")
plt.savefig('figures/benchmark_image.png')

# %%

