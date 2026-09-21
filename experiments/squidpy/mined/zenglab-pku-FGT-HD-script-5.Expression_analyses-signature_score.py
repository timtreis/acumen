# mined from: https://github.com/zenglab-pku/FGT-HD/blob/6646b2e38020eeed2ea0650cccd22596baafc4f7/script/5.Expression_analyses/signature_score.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
import warnings
import pandas as pd
import scanpy as sc
import squidpy as sq
import numpy as np
import os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %%
def read_gmt_file(gmt_path):
    gene_sets = {}
    with open(gmt_path, 'r') as file:
        for line in file:
            split_line = line.strip().split('\t')
            pathway_name = split_line[0]
            genes = split_line[2:]
            gene_sets[pathway_name] = genes
    return gene_sets


def score_cells(adata, signature_name, genes):
    adata.obs[signature_name] = np.nan
    for sample_id in adata.obs['sample'].cat.categories:
        adata_sample = adata[adata.obs['sample'] == sample_id].copy()
        sc.tl.score_genes(adata_sample, gene_list=[x for x in genes if x in adata_sample.var_names], score_name=signature_name)
        adata.obs[signature_name][adata.obs['sample'] == sample_id] = adata_sample.obs[signature_name]

# %%
gmt_path = './ref/integrated_geneset.gmt'
gene_sets = read_gmt_file(gmt_path)

# %%
pathways = ["HALLMARK_ANGIOGENESIS'","HALLMARK_APOPTOSIS","HALLMARK_COMPLEMENT","HALLMARK_DNA_REPAIR","HALLMARK_E2F_TARGETS","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION","HALLMARK_G2M_CHECKPOINT",
            "HALLMARK_GLYCOLYSIS","HALLMARK_HYPOXIA","HALLMARK_IL2_STAT5_SIGNALING","HALLMARK_IL6_JAK_STAT3_SIGNALING","HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_KRAS_SIGNALING_UP",
            "HALLMARK_MTORC1_SIGNALING","HALLMARK_MYC_TARGETS_V1","HALLMARK_MYC_TARGETS_V2","HALLMARK_OXIDATIVE_PHOSPHORYLATION","HALLMARK_P53_PATHWAY","HALLMARK_PI3K_AKT_MTOR_SIGNALING",
            "HALLMARK_TGF_BETA_SIGNALING","HALLMARK_TNFA_SIGNALING_VIA_NFKB","KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION"]
filtered_gene_sets = {key: gene_sets[key] for key in gene_sets if key in pathways}

adata_HGSOC = sc.read_h5ad("./clustered_adata_8um.h5ad")
sq.gr.spatial_neighbors(adata_HGSOC, n_rings=1, library_key='sample', coord_type="grid", n_neighs=8)

groups_of_interest = [1, 3, 8, 9, 10, 11, 12, 16, 17]
adata = adata_HGSOC[adata_HGSOC.obs["cluster_cellcharter"].isin(groups_of_interest),]

# %%
for pathway, genes in filtered_gene_sets.items():
    score_cells(adata, pathway, genes)

# %%
for signature_name in pathways:
    adata.obs[f'{signature_name}_smoothed'] = np.nan*np.ones(adata.shape[0])
    adj = adata.obsp['spatial_connectivities']
    neighbor_score = (adj @ adata.obs[f'{signature_name}']) / np.array(np.sum(adj, axis=1)).squeeze()
    neighbor_score[neighbor_score == float('inf')] = 0

    score_smoothed = np.where(neighbor_score != 0, adata.obs[f'{signature_name}'] * 0.5 + neighbor_score * 0.5, adata.obs[f'{signature_name}'])
    low = np.nanpercentile(score_smoothed, 2.5)
    score_smoothed[score_smoothed < low] = low
    high = np.nanpercentile(score_smoothed, 97.5)
    score_smoothed[score_smoothed > high] = high

    adata.obs[f'{signature_name}_smoothed'] = score_smoothed

# %%
scored_obs = adata.obs
scored_obs.to_csv("./signature_score_obs.csv")

# %%
geneset = pd.read_csv("./CAFs_geneset.csv", sep='\t')
filtered_genesets = {
    row['subtype']: [gene.strip() for gene in row['geneset'].split(', ')]
    for _, row in geneset.iterrows()
}

# %%
adata = sc.read_h5ad("./clustered_adata_8um.h5ad")
fibro_adata = adata[adata.obs["annotations"]=="Fibroblast"].copy()
fibro_adata.obs["sample"] = fibro_adata.obs["sample"].astype("category")
sc.pp.normalize_total(fibro_adata, target_sum=10e4)
sc.pp.log1p(fibro_adata)

# %%
for pathway, genes in filtered_genesets.items():
    print(pathway)
    score_cells(fibro_adata, pathway, genes)
    

# %%
fibro_adata.obs.to_csv("/fibro_obs_wt_score.csv")

# %%

