"""
Confirms answers for task `ligrec_top_pair` (train + test).

Runs squidpy's receptor-ligand permutation-test analysis on the seqfish
mouse-embryo dataset, then for a given (source_cluster, target_cluster) pair
finds the ligand-receptor gene pair with the highest mean interaction score.
"""

import squidpy as sq

adata = sq.datasets.seqfish()

res = sq.gr.ligrec(
    adata,
    n_perms=1000,
    cluster_key="celltype_mapped_refined",
    copy=True,
    use_raw=False,
    transmitter_params={"categories": "ligand"},
    receiver_params={"categories": "receptor"},
)

means = res["means"]


def top_pair(source_cluster, target_cluster):
    col = means[(source_cluster, target_cluster)].dropna()
    top = col.sort_values(ascending=False)
    gene_source, gene_target = top.index[0]
    return f"{gene_source}-{gene_target}", top.iloc[0]


train_pair, train_val = top_pair("Erythroid", "Endothelium")
test_pair, test_val = top_pair("Cardiomyocytes", "Endothelium")

print("TRAIN (Erythroid -> Endothelium):", train_pair, train_val)
print("TEST  (Cardiomyocytes -> Endothelium):", test_pair, test_val)
