# mined from: https://github.com/ratschlab/he2st/blob/17087753ce410a9fbd928255679c8c42c088bc58/plotting_notebooks/Figure3BC.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import pandas as pd
import numpy as np
import plotnine as p9
import glob
import yaml
from tqdm import tqdm
import scanpy as sc
import squidpy as sq
import anndata as ad
import glob

# %%
dataset = "10x_TuPro"

# %%
out_folder = "out_benchmark"
genes = pd.read_csv(f"../{dataset}/out_benchmark/info_highly_variable_genes.csv")
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]
genes_to_predict

# %%
with open(f"../{dataset}/config_dataset.yaml", "r") as stream:
    DATASET_INFO = yaml.safe_load(stream)
models = DATASET_INFO["MODEL"]
all_samples = DATASET_INFO["SAMPLE"]
all_samples

# %%
sample = "MELIPIT-1-1"

# %%
adata_pred_list = {s:{} for s in all_samples}
adata_true_list = {}

for sample in [sample]:#tqdm(all_samples):
    adata_true = sc.read_h5ad(f"../{dataset}/out_benchmark/data/h5ad/{sample}.h5ad")
    sc.pp.normalize_total(adata_true)
    sc.pp.log1p(adata_true)
    adata_true = adata_true[:,adata_true.var.index.isin(genes_to_predict.gene_name)]
    
    adata_true.var["method"] = "Visium, 10x Genomics"
    adata_true.obs["method"] = "Visium, 10x Genomics"
    adata_true.obs["sample_id"] = sample
    adata_true_list[sample] = adata_true
    for model in models:
        #try:
            #adata_pred = sc.read_h5ad(f"../{dataset}/out_benchmark/prediction/{model}/data/h5ad/{sample}.h5ad")
            top_model = pd.read_csv(f"../{dataset}/out_benchmark/evaluation/{model}/top_model_per_test_sample.csv")
            row = top_model[top_model.test_sample.apply(lambda x: sum([s == sample for s in x.split("_")]) == 1)].iloc[0]
            path = f"../{dataset}/out_benchmark/evaluation/{row.test_sample}/*/{model}/prediction/{row.model}_test.pkl"
            path = path.replace('[', '+-+').replace(']', '-+-')
            path = path.replace('+-+', '[[]').replace('-+-', '[]]')    
            expression_predicted_file = glob.glob(path)[0]
            expression_predicted = pd.read_pickle(expression_predicted_file)
            idx = expression_predicted.index.to_series().apply(lambda x: x.split("_")[1]).isin([sample]).values
            expression_predicted = expression_predicted.iloc[idx]
            

            expression_predicted.index = expression_predicted.index.to_series().apply(lambda x: x.split("_")[0])
            expression_predicted = expression_predicted.loc[adata_true.obs.index]
            expression_predicted = expression_predicted[adata_true.var.index]
            adata_pred = adata_true.copy()
            adata_pred.X = expression_predicted.values
            #adata_pred.X = np.exp(adata_pred.X)
            #adata_pred.X[adata_pred.X < 0] = 0
            #adata_pred.X = adata_pred.X
            
            adata_pred.obs["method"] = model
            adata_pred.obs["sample_id"] = sample
            adata_pred.var["method"] = model
            adata_pred_list[sample][model] = adata_pred
        #except:
        #    print(f"Not generated for: {sample}, {model}")
    #adata_pred.var.index = [f"{i}_predicted" for i in adata_pred.var.index]
    

# %%
adata = ad.concat((adata_pred_list[sample]), axis=1, merge="first", uns_merge="first")
adata = ad.concat((adata, adata_true_list[sample]), axis=1, merge="first", uns_merge="first")
adata.var.index = [f"{row.name} {row.method}" for _, row in adata.var.iterrows()]

# %%
adata.obs["ground_truth"].unique()

# %%
translate = {
    'Tumor': 'Tumor', 
    'Stroma': 'Stroma', 
    'Normal lymphoid tissue': 'Normal\nlymphoid', 
    'Blood and necrosis': 'Blood/\nnecrosis',
    'Pigment': np.nan # ignore label
}

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

pad = -10
bounds = (adata.obsm["spatial"][:, 0].min() - pad * 1,
              adata.obsm["spatial"][:, 1].min() - pad * 10,
              adata.obsm["spatial"][:, 0].max() + pad * 1,
              adata.obsm["spatial"][:, 1].max() + pad* 10)


adata.obs["H&E image"] = np.nan
adata.obs["Pathology annotation"] = adata.obs["ground_truth"].apply(lambda x: translate[x])


sq.pl.spatial_scatter(adata, 
                      img_alpha=0.9, 
                      crop_coord=bounds, 
                      wspace=0, 
                      hspace=0.1,
                      color=["H&E image", "Pathology annotation"], 
                      size=15,      
                      ncols=1, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure3B-{sample}_h&e_anno.png", 
                      dpi=300,
                      frameon=False, 
                      colorbar=False, 
                      #legend_loc="lower left",
                      legend_fontsize=15,
                      figsize=(7, 5))

# %%
def significance_level(p_value):
    if p_value < 0.001:
        return '***'  # highly significant
    elif p_value < 0.01:
        return '**'   # significant
    elif p_value < 0.05:
        return '*'    # marginally significant
    else:
        return 'ns'   # not significant

# %%
from scipy.stats import pearsonr

genes = adata.var.index.to_series().apply(lambda x: x.split(" ")[0]).unique()
genes = genes[np.isin(genes, genes_to_predict[genes_to_predict.variances_norm_rank < 50].gene_name)]
corr_score = {m:{} for m in models}


for gene in tqdm(["SOX10"]):

    gene_expr_visium = adata[:, adata.var.index == f'{gene} Visium, 10x Genomics'].X.squeeze()
    
    for model in models:
        try:
            gene_expr_model = adata[:, adata.var.index == f'{gene} {model}'].X.squeeze()
    
            res = pearsonr(gene_expr_model, gene_expr_visium)
            corr_score[model][gene] = {}
            corr_score[model][gene]["r"] = res.statistic
            corr_score[model][gene]["p"] = significance_level(res.pvalue)
        except:
            pass

# %%
corr_score

# %%
# SLC6A15', 'BSG', 'CAPN3', 'TBC1D4', 'MYC', 'COL11A2', 'JCHAIN',
#       'CD74', 'PACSIN3', 'C1R', 'HLA-DPA1', 'HSP90AA1', 'APOD', 'MAFB',
#       'IRF8', 'SELENOM', 'WIPF1', 'MAMDC2', 'SEL1L3', 'JUNB', 'CD53',
#       'RHOBTB3', 'C1QA', 'GPRC5B', 'MPEG1', 'RAC2', 'GYPC', 'CDH11',
#       'C1QC', 'STRADB', 'BIRC7', 'C1S', 'MIF', 'TAGLN', 'XBP1'
#np.array(list(corr_score["DeepSpot"].keys()))[np.argsort(-np.array(list(corr_score["DeepSpot"].values())))][:100]

# %%
adata

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

pad = -10

gene = "SOX10"

bounds = (adata.obsm["spatial"][:, 0].min() - pad * 1,
              adata.obsm["spatial"][:, 1].min() - pad * 10,
              adata.obsm["spatial"][:, 0].max() + pad * 1,
              adata.obsm["spatial"][:, 1].max() + pad* 10)

color = [f"{gene} {m}" for m in ['Visium, 10x Genomics',
                                 'MeanCellType',
                                 "BLEEP",
                                 'STNet',
                                 'MLP',
                                 "DeepSpot"]]


title = [f"{c}\nPearson r={corr_score[c.split(' ')[1]][gene]['r']:.2f}{corr_score[c.split(' ')[1]][gene]['p']}" if c.split(' ')[1] in models else c for c in color]

# %%
sq.pl.spatial_scatter(adata, 
                      img_alpha=0.9, 
                      crop_coord=bounds, 
                      wspace=0.15, 
                      hspace=0.12,
                      color=color, 
                      title=title,
                      size=15,      
                      ncols=3, 
                      cmap="viridis",
                      #title=title, 
                      save=f"figures/Figure3C-{sample}_{gene}.png", 
                      dpi=300,
                      frameon=False, 
                      colorbar=0, 
                      #legend_loc="lower left",
                      figsize=(5.5, 6))

# %%


# %%


# %%

