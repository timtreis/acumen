# mined from: https://github.com/ratschlab/he2st/blob/17087753ce410a9fbd928255679c8c42c088bc58/plotting_notebooks/Figure7AC.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import pandas as pd
import numpy as np
import plotnine as p9
import glob
import yaml
from plotnine_prism import *
from sklearn.linear_model import LinearRegression, LogisticRegression

import scanpy as sc
from tqdm.notebook import tqdm
import yaml
import matplotlib.pyplot as plt
import squidpy as sq
import anndata as ad
import sys
sys.path.append('../')
from src.utils import bootstrapping

# %%
with open("../config.yaml", "r") as stream:
    DATASET_INFO = yaml.safe_load(stream)
DATASET_INFO

# %%
pearson_variation_paths = glob.glob("../Lung_Xenium/out_benchmark/evaluation/pearson_variation.csv")
pearson_variation_paths

# %%
data_var = []
for file in pearson_variation_paths:
    tab = pd.read_csv(file)
    dataset = file.split("/")[1]
    if dataset not in DATASET_INFO["DATASET_NAME"]: continue
    tab["Dataset"] = DATASET_INFO["DATASET_NAME"][dataset]
    data_var.append(tab)
data_var = pd.concat(data_var)
data_var.Dataset = pd.Categorical(data_var.Dataset, DATASET_INFO["DATASET_NAME"].values())
data_var.pearson_median = data_var.pearson_median.astype(float)
data_var.pearson_std = data_var.pearson_std.astype(float)
data_var

# %%
tab = data_var.copy()
position_dodge_width = 0.2
tab["model_rank"] = tab.groupby("Dataset", observed=False).pearson_median.rank(ascending=False)
tab.model = tab.model.apply(lambda x: x.split("_")[0])
tab = tab.query("model in ['DeepCell', 'BLEEP', 'STNet']").copy()
tab["Model"] = pd.Categorical(tab.model, tab.groupby("model").model_rank.agg("median").sort_values().index)
tab.top_n = tab.top_n.astype("category")

g = (p9.ggplot(tab, p9.aes("top_n", "pearson_median", color="Model"))
 + p9.geom_line(p9.aes(color="Model", group="Model"), linetype="dashed", 
                position=p9.position_dodge(width=position_dodge_width))
 + p9.geom_point(p9.aes(color="Model"), position=p9.position_dodge(width=position_dodge_width), size=0.7) 
 + p9.facet_wrap("~Dataset", scales="free_y", ncol=2)
 + p9.geom_errorbar(p9.aes(x="top_n", ymin="pearson_median-pearson_std",
                           ymax="pearson_median+pearson_std", color="Model"), 
                    width=0.4, alpha=1, size=0.5,
                    position=p9.position_dodge(width=position_dodge_width))
 + p9.theme_bw()
 + p9.theme(panel_spacing_y=0, panel_spacing_x=0, figure_size=(7, 5), 
            #axis_text_x = p9.element_blank(), 
            legend_position="right",
            text=p9.element_text(size=17),
            strip_text=p9.element_text(size=17),
            legend_title=p9.element_text(size=17),
            legend_text=p9.element_text(size=16))
 + p9.ylab("Pearson correlation")
 + p9.xlab("Most variable genes")
 + p9.theme(axis_text_x = p9.element_text(angle = 90, hjust = 1))
 + scale_color_prism(palette = "colors")
 + p9.guides(color=p9.guide_legend(nrow=10, override_aes = p9.aes(shape = ".")))
)
g.save("figures/Figure7A-benchmark_xenium.png", dpi=300)
g

# %%


# %%
ref_lung_atlas = sc.read_h5ad("/cluster/customapps/biomed/grlab/users/knonchev/lung_atlas/b351804c-293e-4aeb-9c4c-043db67f4540.h5ad")
ref_lung_atlas.X = ref_lung_atlas.X.toarray()
ref_lung_atlas.var.index = ref_lung_atlas.var.feature_name.values
ref_lung_atlas

# %%
with open("../Lung_Xenium/config_dataset.yaml", "r") as stream:
    config_dataset = yaml.safe_load(stream)
samples = config_dataset["SAMPLE_LQ"]
genes_of_interest = config_dataset["genes_of_interest"]
samples

# %%
adata_genes = pd.read_csv("../Lung_Xenium/out_benchmark/info_highly_variable_genes.csv").query("isPredicted == True").gene_name.values
all_genes = np.array(list((set(genes_of_interest) | set(adata_genes)) & set(ref_lung_atlas.var.feature_name)))
shared_genes = np.array(list(set(ref_lung_atlas.var.feature_name.values) & set(adata_genes)))
len(all_genes), len(shared_genes)

# %%
ref_lung_atlas_genes = ref_lung_atlas[:, all_genes].copy()
sc.pp.filter_cells(ref_lung_atlas_genes, min_genes=10)

# %%
ref_lung_atlas_genes_subset = ref_lung_atlas_genes[:, shared_genes].copy()
ref_lung_atlas_genes_subset

# %%
lr = LinearRegression(n_jobs=-1)
lr.fit(ref_lung_atlas_genes_subset.X, ref_lung_atlas_genes.X)

# %%
scores = []
adatas = {}
for sample in tqdm(samples):
    adata_path_gt = f"../Lung_Xenium/data/h5ad_all_genes/{sample}.h5ad"
    adata_gt = sc.read_h5ad(adata_path_gt)
    sc.pp.normalize_total(adata_gt, target_sum=10000)
    sc.pp.log1p(adata_gt)
    adata_gt_subset = adata_gt[:, shared_genes].copy()

    counts_gt_y_hat = lr.predict(adata_gt_subset.X)
    
    
    adata_gt_y_hat = ad.AnnData(counts_gt_y_hat, obs=adata_gt.obs)
    adata_gt_y_hat.var.index = ref_lung_atlas_genes.var.feature_name.values

    expr_gt_y = pd.DataFrame(adata_gt.X, columns=adata_gt.var.index)[all_genes]
    
    expr_gt_y_hat = pd.DataFrame(adata_gt_y_hat.X, columns=all_genes)[all_genes]

    corr_genes = expr_gt_y.corrwith(expr_gt_y_hat, method="pearson").fillna(0)

    present_genes_upper = corr_genes[~corr_genes.index.to_series().isin(genes_of_interest)].values#.mean()
    absent_genes_upper = corr_genes[corr_genes.index.to_series().isin(genes_of_interest)].values#.mean()
    
    adatas[sample] = {}
    adatas[sample]["10x, Xenium"] = adata_gt[:, all_genes].copy()
    #scores.append([sample, "10x, Xenium", present_genes_upper, absent_genes_upper])

    for model in config_dataset["MODEL"]:
        try:
            adata_path = f"../Lung_Xenium/out_benchmark/prediction/{model}/data/h5ad/{sample}.h5ad"
            adata = sc.read_h5ad(adata_path)            
            
            adata_subset = adata[:, shared_genes].copy()

            counts_y_hat = lr.predict(adata_subset.X)
            
            counts_y_hat[counts_y_hat < 0] = 0

            adata_y_hat = ad.AnnData(counts_y_hat, obs=adata.obs, obsm=adata.obsm, uns=adata.uns)
            adata_y_hat.var.index = ref_lung_atlas_genes.var.feature_name.values
            adatas[sample][model] = adata_y_hat
    
            expr_y_hat = pd.DataFrame(adata_y_hat.X, columns=all_genes)
    
            corr_genes = expr_gt_y.corrwith(expr_y_hat, method="pearson").fillna(0)
    
            present_genes_pred = corr_genes[~corr_genes.index.to_series().isin(genes_of_interest)].values#.mean()
            absent_genes_pred = corr_genes[corr_genes.index.to_series().isin(genes_of_interest)].values#.mean()
            
            scores.append([sample, model, present_genes_pred, absent_genes_pred])
        except Exception as e:
            print(sample, model, e)

     

# %%
results = pd.DataFrame(scores, columns=["sample_id", "method", f"Observed\n(n={len(shared_genes)})", f"Hold-out\n(n={len(genes_of_interest)})"])
results = results.drop({"sample_id"}, axis=1)
results = results.groupby(["method"]).agg("mean").reset_index()
results = results.melt(["method"],
           value_name="correlation", 
           var_name="Genes")
# Assuming bootstrapping returns a tuple like (x_mean, x_std)
results[['correlation_mean', 'correlation_std']] = results['correlation'].apply(
    lambda x: pd.Series(bootstrapping(x))
)
results["Method"] = results.method.apply(lambda x: x.replace("_cell", ""))
results.Method = pd.Categorical(results.Method, ["BLEEP", "STNet", "DeepCell"])
results.Genes = pd.Categorical(results.Genes, ["Observed\n(n=300)", "Hold-out\n(n=16)"])
results
results

# %%
position_dodge_width = 0.5
g = (p9.ggplot(results, p9.aes("Method", "correlation_mean", color="Genes")) 
 + p9.geom_point(position=p9.position_dodge(width=position_dodge_width))
+ p9.geom_errorbar(
        p9.aes(
            ymin="correlation_mean - correlation_std",
            ymax="correlation_mean + correlation_std"
        ),
        width=0.2,  # width of the error bars
    position=p9.position_dodge(width=position_dodge_width)
    )
 + scale_color_prism(palette = "colors")
 + p9.theme_bw()
 + p9.theme(panel_spacing_y=0, panel_spacing_x=0, figure_size=(7, 5), 
            axis_text_x = p9.element_text(angle = 25, hjust = 1),
            legend_position="right",
            text=p9.element_text(size=17),
            strip_text=p9.element_text(size=17),
            legend_title=p9.element_text(size=17),
            legend_text=p9.element_text(size=16))
 + p9.ylab("Pearson correlation")
)
g.save("figures/Figure7B-benchmark_atlas.png", dpi=300)
g

# %%
clf = LogisticRegression(n_jobs=-1)
clf.fit(ref_lung_atlas_genes.X, ref_lung_atlas_genes.obs.ann_level_2.values)
clf.score(ref_lung_atlas_genes.X, ref_lung_atlas_genes.obs.ann_level_2.values)

# %%
samples

# %%
sample = "NCBI867"

# %%


# %%
adatas[sample]['10x, Xenium'].var.index = [f"{g}, 10x Xenium" for g in adatas[sample]['10x, Xenium'].var.index]
adatas[sample]['DeepCell'].var.index = [f"{g}, DeepCell" for g in adatas[sample]['DeepCell'].var.index]

# %%
adata_concat = ad.concat((adatas[sample]['10x, Xenium'], adatas[sample]['DeepCell']), axis=1, merge="same", uns_merge="same")
adata_concat

# %%
genes_to_plot = ["MSLN", "PLIN2", "ACTA2"]
color = [f"{g}, {m}" for m in ["10x Xenium", "DeepCell"] for g in genes_to_plot]
color

# %%
plt.rcParams.update({'font.size': 20})


bounds = (adata_concat.obsm["spatial"][:, 0].min(),
              adata_concat.obsm["spatial"][:, 1].min()+50,
              adata_concat.obsm["spatial"][:, 0].max()-50,
              adata_concat.obsm["spatial"][:, 1].max()-350)
bounds

# %%
sq.pl.spatial_scatter(adata_concat, 
                      color=color, 
                      #img_alpha=0.5,
                      img=False,
                      crop_coord=bounds, 
                      wspace=0.1, 
                      hspace=0.1,
                      size=3,      
                      ncols=len(genes_to_plot), 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure7F_lung_xenium_{sample}_{'_'.join(genes_to_plot)}.png", 
                      dpi=150,
                      frameon=False, 
                      colorbar=False, 
                      legend_fontsize=15,
                      figsize=(7, 7))
plt.show()

# %%


# %%

