# mined from: https://github.com/AndyCGraham/thesis/blob/4845be2689b354c6f1991a40d5641c700df8f0f4/scripts/multiome/python/visium.ipynb
# symbols: squidpy.pl.spatial_scatter, squidpy.read.visium

# %%
import glob
import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy

import scanpy as sc
import scanpy.external as sce
import squidpy as sq
import anndata as ad

import cell2location
from cell2location.utils import select_slide
from sklearn.linear_model import ElasticNetCV

# Check if current working directory is named "python" and change if needed
current_dir = os.getcwd()
if os.path.basename(current_dir) == "python":
    os.chdir("../../../")
    print(f"Changed working directory to: {os.getcwd()}")
else:
    print(f"Current working directory: {os.getcwd()}")

random.seed(42)

# %%
# create paths and names to results folders for reference regression and cell2location models
results_folder = 'results/visium/'
ref_run_name = f'{results_folder}/reference_signatures'
run_name = f'{results_folder}/cell2location_map'

# %%
# Find all h5 files in the data/visium directory
h5_files = glob.glob('data/visium/**/filtered_feature_bc_matrix.h5', recursive=True)

# Initialize a list to store AnnData objects
adata_list = []

# Loop through each h5 file
for file in h5_files:
    adata = sq.read.visium(os.path.dirname(file))
    adata.var_names_make_unique()
    # Get annotations from filename
    split = os.path.basename(os.path.dirname(file)).split('_')
    sample_str = '_'.join(split[1:])
    adata.obs['sample'] = sample_str[0].upper() + sample_str[1:]
    adata.obs['batch'] = split[3].split('-')[0]
    # Extract age from the sample name (text before the first underscore)
    adata.obs['age'] = split[0]
    adata_list.append(adata)

# Combine all loaded data if there are any
adata = ad.concat(adata_list,uns_merge='unique')

# Make barcodes unique
adata.obs.index = adata.obs.index.astype(str) + "_" + adata.obs["sample"].astype(str).str.replace("Mouse_brain_", "")

# %%
print(adata.X[:5, :10].toarray())

# You can also check the first few genes and observations
print("\nFirst 5 gene names:")
print(adata.var_names[:5])

print("\nFirst 5 cells with their metadata:")
print(adata.obs.head())

# %%
sq.pl.spatial_scatter(adata, library_key="sample", color="C4b", vmax=2, title=adata.obs["sample"].unique())

# %%
sq.pl.spatial_scatter(adata, library_key="sample", color="Lgals3", vmax=1, title=adata.obs["sample"].unique())

# %%
# find mitochondria-encoded (MT) genes
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# %%
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
sns.histplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, ax=axs[1])
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
sns.histplot(adata.obs["pct_counts_mt"], kde=False, bins=60, ax=axs[3])

# %%
sc.pp.filter_cells(adata, min_counts=2000)
#adata = adata[adata.obs["pct_counts_mt"] < 35].copy()
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=3)

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "sample"], wspace=0.4)

# %%
sce.pp.harmony_integrate(adata, 'batch')
adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
sc.pl.pca_variance_ratio(adata, log=True)

# %%
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "sample"], wspace=0.4)

# %%
res=0.3
sc.tl.leiden(adata, resolution=res)
adata.obs[f"leiden_{res}"] = adata.obs["leiden"].astype("category")
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=[f"leiden_{res}"], wspace=0.4)

# %%
sq.pl.spatial_scatter(adata, library_key="sample", color=f"leiden_{res}", title=adata.obs["sample"].unique())

# %%
hippo = adata[adata.obs[f"leiden_{res}"].isin(["2","8"])].copy()
sq.pl.spatial_scatter(hippo, library_key="sample", color=f"leiden_{res}", title=hippo.obs["sample"].unique())

# %%
adata_ref = sc.read_h5ad("../projects/multiome/analysis/visium/data/snRNA.h5ad")

# %%
# prepare anndata for the regression model
cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                        # cell type, covariate used for constructing signatures
                        labels_key='Clusters'
                       )
# create the regression model
mod = cell2location.models.RegressionModel(adata_ref)

# view anndata_setup as a sanity check
mod.view_anndata_setup()

# %%
mod.train(max_epochs=250)

# %%
mod.plot_history(20)

# %%
# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
)

# Save model
mod.save(f"{ref_run_name}", overwrite=True)

# Save anndata object with results
adata_file = f"{ref_run_name}/sc.h5ad"
adata_ref.write(adata_file)

# %%
mod.plot_QC()

# %%
# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]

# %%
inf_aver.loc["Lgals3",["MG.Hm","MG.ARM"]]

# %%
# find shared genes and subset both anndata and reference signatures
adata.layers["logcounts"] = adata.X.copy()
adata.X = adata.layers["counts"]

# remove MT genes for spatial mapping (keeping their counts in the object)
adata = adata[:, ~adata.var['mt'].values]

intersect = np.intersect1d(adata.var_names, inf_aver.index)
adata = adata[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=adata, batch_key="batch")

# %%
# create and train the model
mod = cell2location.models.Cell2location(
    adata, cell_state_df=inf_aver,
    # the expected average cell abundance: tissue-dependent
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=20,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection:
    detection_alpha=40
)
mod.view_anndata_setup()

# %%
mod.train(max_epochs=2000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1
         )

# %%
# plot ELBO loss history during training, removing first 10 epochs from the plot
mod.plot_history(10)
plt.legend(labels=['full data training']);

# %%
# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata = mod.export_posterior(
    adata, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs}
)

# Save model
mod.save(f"{run_name}", overwrite=True)

# Save anndata object with results
adata_file = f"{run_name}/sp.h5ad"
adata.write(adata_file)

# %%
mod.plot_QC()

# %%
fig = mod.plot_spatial_QC_across_batches()

# %%
# add 5% quantile, representing confident cell abundance, 'at least this amount is present',
# to adata.obs with nice names for plotting
adata.obs[adata.uns['mod']['factor_names']] = adata.obsm['q05_cell_abundance_w_sf']
adata.obs["MG.ARM.Norm"] = adata.obs["MG.ARM"] - adata.obs["MG.Hm"]

sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['MG.ARM.Norm'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=2,title=adata.obs["sample"].unique(),vmax=3
                 )

# %%
slide = select_slide(hippo, 'Mouse_brain_B1-2')
sq.pl.spatial_scatter(slide, cmap='magma',
                  # show first 8 cell types
                  color=['MG.ARM.Norm'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,vmax=2,title="MG.ARM",
                  figsize=(5, 5),
                  crop_coord=(5000, 4000, 11000, 13000)
                 )

# %%
slide.uns['spatial']['Mouse_brain_B1-2']['images']['hires'].shape

# %%
sq.pl.spatial_scatter(slide, cmap='magma',
                  # show first 8 cell types
                  color=['Oligodendrocytes.DAO'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,vmax=3,title="OLG.DAO",
                  figsize=(5, 5),
                  crop_coord=(5000, 4000, 11000, 13000)
                 )

# %%
sq.pl.spatial_scatter(adata, library_key="sample")

# %%
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['MG.ARM'],
                  ncols=4, size=1.3, img_cmap='Blues',
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=3,title=adata.obs["sample"].unique(),vmax=4
                 )

# %%
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['MG.Hm'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,title=adata.obs["sample"].unique()
                 )

# %%
adata.obs["DAO.Norm"] = adata.obs["Oligodendrocytes.DAO"] - adata.obs["Oligodendrocytes"]
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['DAO.Norm'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,title=adata.obs["sample"].unique()
                 )

# %%
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['Oligodendrocytes.DAO'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,title=adata.obs["sample"].unique()
                 )

# %%
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['Oligodendrocytes'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,title=adata.obs["sample"].unique()
                 )

# %%
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['ExNeu.DG.GC'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,title=adata.obs["sample"].unique()
                 )

# %%
sq.pl.spatial_scatter(adata, library_key="sample", cmap='magma',
                  # show first 8 cell types
                  color=['ExNeu.Sub.Cbln4'],
                  ncols=4, size=1.3,
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0,title=adata.obs["sample"].unique()
                 )

# %%
r,p = scipy.stats.pearsonr(hippo.obs["MG.ARM.Norm"], hippo.obs["DAO.Norm"]) 
print(f"r={r}, p={p}")

# %%
r,p = scipy.stats.pearsonr(hippo.obs["MG.ARM"], hippo.obs["ExNeu.DG.GC"]) 
print(f"r={r}, p={p}")

# %%
r,p = scipy.stats.pearsonr(hippo.obs["MG.ARM"], hippo.layers["logcounts"].toarray()[:, hippo.var_names == "Serpina3n"].flatten()) 
print(f"r={r}, p={p}")

# %%
boots=[]
n_boots = 100
boot_size=1
for _ in range(n_boots):
    ridge = ElasticNetCV(cv=5,max_iter=10000)
    bootstrap = hippo[random.choices(hippo.obs.index,k=hippo.n_obs*boot_size),:].copy()
    X = bootstrap.obs.iloc[:, 25:68].drop(columns=["MG.ARM","BAM","NSCs","Neuroblasts","Cajal-Retzus Cells"])
    ridge.fit(X,bootstrap.obs[["MG.ARM"]].to_numpy())
    # Create a DataFrame with coefficients and their corresponding feature names
    coef_df = pd.DataFrame({
        'Feature': X.columns.tolist(),
        'Coefficient': ridge.coef_  # First column of coefficients for each target
    })
    boots.append(coef_df) 

# %%
import numpy as np
from matplotlib.colors import Normalize

# Concatenate all bootstrapped coefficients
coef_df = pd.concat(boots, axis=0)

# Replace long cluster names with shorter ones
coef_df['Feature'] = coef_df['Feature'].replace('Oligodendrocytes.DAO', 'OLG.DAO')

# Calculate the mean coefficient for each cluster
means = coef_df.groupby('Feature')['Coefficient'].mean()

# Filter out clusters with small coefficients (abs < 0.5)
means = means[abs(means) > 0.5].sort_values(ascending=False)
filtered_coef = coef_df[coef_df['Feature'].isin(means.index)].copy()

# Sort by mean cluster coeficients
filtered_coef['Feature'] = pd.Categorical(filtered_coef['Feature'], 
               categories=means.index.tolist(), ordered=True)
filtered_coef = filtered_coef.sort_values('Feature')

# Create bar plot with seaborn
plt.figure(figsize=(12, 8))

# Create a symmetric normalization around 0
vmax = max(abs(filtered_coef['Coefficient'].min()), abs(filtered_coef['Coefficient'].max()))
norm = Normalize(vmin=-vmax, vmax=vmax)

# Create the barplot with colors mapped symmetrically around 0
colors = plt.cm.coolwarm(norm(means))
ax = sns.barplot(x='Coefficient', y='Feature', data=filtered_coef, palette=colors,
                 errorbar='se', capsize=0.1)
# Add vertical line at x=0 
plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)

# Add labels and title
plt.title('ARM Prediction Elastic Net Regression Coefficients (abs > 0.5)', fontsize=20)
plt.xlabel('Coefficient Value', fontsize=17.5)
plt.ylabel('Cluster', fontsize=17.5)
plt.yticks(fontsize=15)

# Improve readability
plt.tight_layout()
plt.grid(axis='x', alpha=0.3)

# Highlight the "OLG.DAO" label in orange
for label in ax.get_yticklabels():
    if label.get_text() == "OLG.DAO":
        label.set_color("#ED7D31")
        label.set_fontweight('bold')

# Show plot
plt.show()

# %%
pd.concat(boots, axis=0).groupby('Feature')['Coefficient'].mean()

# %%
plt.figure(figsize=(12, 8))
sns.barplot(x='MG.ARM', y='sample', data=hippo.obs, palette='coolwarm_r')

# %%
plt.figure(figsize=(12, 8))
sns.barplot(x='Oligodendrocytes.DAO', y='sample', data=hippo.obs, palette='coolwarm_r')
