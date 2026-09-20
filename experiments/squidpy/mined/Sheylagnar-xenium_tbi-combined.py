# mined from: https://github.com/Sheylagnar/xenium_tbi/blob/18f2e046dca901bc34d696f4480d5dcc78e8af8e/combined.py
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors


import spatialdata as sd
from spatialdata_io import xenium
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq

xenium_ctrl_path = "../Male_Sham_amygdala"
xenium_tbi_path = "../Male_TBI_amygdala"

sdata_hombre_control = xenium(xenium_ctrl_path)
sdata_hombre_tbi = xenium(xenium_tbi_path)

adata_control = sdata_hombre_control.tables["table"]
adata_tbi = sdata_hombre_tbi.tables["table"]

adata_control.obs.index = [f"control_{i}" for i in range(adata_control.n_obs)]
adata_tbi.obs.index = [f"tbi_{i}" for i in range(adata_tbi.n_obs)]

combined = sc.concat(
    [adata_control, adata_tbi],
    keys=["hombre_control", "hombre_tbi"],
    label="sample",
    merge="unique",
)

#Add metadata
combined.obs['condition'] = combined.obs['sample'].map({
    'hombre_control': 'control',
    'hombre_tbi': 'tbi'
})

#preprocessing
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)
sc.pp.highly_variable_genes(combined, flavor="seurat", n_top_genes=2000)
# Análisis diferencial basado en Wilcoxon (puedes usar otros métodos como t-test)
sc.tl.rank_genes_groups(combined, groupby="condition", method="wilcoxon")
result = combined.uns["rank_genes_groups"]
import pandas as pd
# Convertir a DataFrame
# Crear el DataFrame para la condición 'tbi'
volcano_data = pd.DataFrame({
    "gene": result["names"]["tbi"],  # Nombres de los genes
    "logFC": result["logfoldchanges"]["tbi"],  # Log Fold Change
    "pval": result["pvals"]["tbi"],  # p-values
    "adj_pval": result["pvals_adj"]["tbi"],  # p-values ajustados
})

#analisis
sq.gr.spatial_neighbors(combined, coord_type="generic")
sq.gr.spatial_autocorr(combined, mode="moran")
for condition in combined.obs['condition'].unique():
    sq.tl.aggregate(
        combined[combined.obs['condition'] == condition],
        by="cluster", agg_func="mean"
    )

sc.tl.rank_genes_groups(combined, groupby="condition", method="wilcoxon")
print(combined.uns['rank_genes_groups'])
sc.pl.rank_genes_groups(combined, n_genes=10, sharey=False)
plt.show()

