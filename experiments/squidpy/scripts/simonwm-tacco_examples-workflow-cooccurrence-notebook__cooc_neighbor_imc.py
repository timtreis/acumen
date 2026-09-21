import squidpy as sq

# --- train: target = "T cells" ---
adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")
res = adata.uns["cell type_co_occurrence"]
occ = res["occ"]
cats = list(adata.obs["cell type"].cat.categories)

target = "T cells"
idx = cats.index(target)
row = occ[idx, :, 0]  # shortest distance interval
ranked = sorted(zip(cats, row), key=lambda x: -x[1])
ranked_excl = [x for x in ranked if x[0] != target]
print("train answer:", ranked_excl[0])

# --- test: target = "endothelial" ---
target = "endothelial"
idx = cats.index(target)
row = occ[idx, :, 0]
ranked = sorted(zip(cats, row), key=lambda x: -x[1])
ranked_excl = [x for x in ranked if x[0] != target]
print("test answer:", ranked_excl[0])
