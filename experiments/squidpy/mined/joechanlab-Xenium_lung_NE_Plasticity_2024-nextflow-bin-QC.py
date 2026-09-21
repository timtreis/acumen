# mined from: https://github.com/joechanlab/Xenium_lung_NE_Plasticity_2024/blob/61c143064d3463afcdd044a7c3efd13eb691f29c/nextflow/bin/QC.ipynb
# symbols: squidpy.pl.spatial_scatter

# %%
import os
import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq
from matplotlib import rcParams

FIGSIZE = (3, 3)
rcParams["figure.figsize"] = FIGSIZE

# %%
adata = sc.read_h5ad(h5ad)
gene_info = pd.read_csv(gene_annotation_file)

# %%
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

axs[0].set_title("Total transcripts per cell")
sns.histplot(
    adata.obs["total_counts"],
    kde=False,
    ax=axs[0],
)

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    adata.obs["n_genes_by_counts"],
    kde=False,
    ax=axs[1],
)

axs[2].set_title("Nucleus area")
sns.histplot(
    adata.obs["nucleus_area"],
    kde=False,
    ax=axs[2],
)

# %%
sc.pl.umap(
    adata,
    color=[
        "total_counts",
        "n_genes_by_counts",
        "leiden",
    ],
    wspace=0.4,
)

# %%
sq.pl.spatial_scatter(
    adata,
    library_id="spatial",
    shape=None,
    color=[
        "leiden",
        "cell_type_predicted"
    ],
    wspace=0.4,
)

# %%
n_genes = 20
SVG = adata.uns["moranI"].head(n_genes).index.values.tolist()

# %%
sq.pl.spatial_scatter(
    adata,
    library_id="spatial",
    shape=None,
    color=SVG,
    wspace=0.4,
)

# %%
n_genes = 10
sc.pl.rank_genes_groups(adata, n_genes = n_genes)

# %%
ranked_genes = adata.uns['rank_genes_groups']
gene_names = ranked_genes['names']
scores = ranked_genes['scores']

genes_dict = {}
for group in gene_names.dtype.names:
    genes_with_score = list(zip(gene_names[group], scores[group]))
    sorted_genes = sorted(genes_with_score, key=lambda x: x[1], reverse=True)  # Sort by fold change in descending order
    top_genes = [gene for gene, score in sorted_genes[:n_genes]]  # Take the top n_genes
    genes_dict[group] = top_genes

# %%
gene_annot_map = dict(zip(gene_info.gene, gene_info.annotation))

def find_keys_for_all_values(gene_annot_map, genes_dict):
    result = []
    for group, genes in genes_dict.items():
        for gene in genes:
            annotation = gene_annot_map.get(gene, None)
            result.append({'group': group, 'gene': gene, 'annotation': annotation})
    return result


table_data = find_keys_for_all_values(gene_annot_map, genes_dict)
df = pd.DataFrame(table_data).drop_duplicates()
pd.set_option('display.max_rows', 500)
df

# %%
n_top_markers = 3
if extra_markers is None:
    top_markers = []
else:
    top_markers = extra_markers.split(",")
for group in gene_names.dtype.names:
    cluster_genes = genes_dict[group]
    top_markers += cluster_genes[:min([len(cluster_genes), n_top_markers])]
top_markers = [x for x in set(top_markers) if (not x.startswith("BLANK") and not x.startswith("NegControl"))]

# %%
sq.pl.spatial_scatter(
    adata,
    library_id="spatial",
    shape=None,
    color=top_markers,
    wspace=0.4
)
