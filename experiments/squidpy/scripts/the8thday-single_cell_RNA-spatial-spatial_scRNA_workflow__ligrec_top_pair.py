import squidpy as sq

# TRAIN: visium_hne_adata, Hippocampus (source) -> Pyramidal_layer (target)
adata = sq.datasets.visium_hne_adata()
res = sq.gr.ligrec(
    adata,
    cluster_key="cluster",
    clusters=["Hippocampus", "Pyramidal_layer"],
    n_perms=100,
    seed=0,
    copy=True,
    use_raw=False,
    show_progress_bar=False,
)
pv = res["pvalues"]
means = res["means"]
col = ("Hippocampus", "Pyramidal_layer")
sub = pv[col].dropna().sort_values()
tied = sub[sub == sub.iloc[0]].index
m = means[col].loc[tied].sort_values(ascending=False)
print("TRAIN top ligand-receptor pair (Hippocampus -> Pyramidal_layer):", m.index[0])
print(m.head())

# TEST: visium_fluo_adata, Cortex_1 (source) -> Cortex_2 (target)
adata2 = sq.datasets.visium_fluo_adata()
res2 = sq.gr.ligrec(
    adata2,
    cluster_key="cluster",
    clusters=["Cortex_1", "Cortex_2"],
    n_perms=100,
    seed=0,
    copy=True,
    use_raw=False,
    show_progress_bar=False,
)
pv2 = res2["pvalues"]
means2 = res2["means"]
col2 = ("Cortex_1", "Cortex_2")
sub2 = pv2[col2].dropna().sort_values()
tied2 = sub2[sub2 == sub2.iloc[0]].index
m2 = means2[col2].loc[tied2].sort_values(ascending=False)
print("TEST top ligand-receptor pair (Cortex_1 -> Cortex_2):", m2.index[0])
print(m2.head())
