import numpy as np
import squidpy as sq

# --- train: imc dataset, reference cluster "T cells" ---
adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type")
res = adata.uns["cell type_co_occurrence"]
occ = res["occ"]
cats = adata.obs["cell type"].cat.categories.tolist()
j = cats.index("T cells")
col = occ[:, j, 0]
order = np.argsort(col)[::-1]
top = [cats[idx] for idx in order if cats[idx] != "T cells"][0]
print("TRAIN answer:", top, col[cats.index(top)])

# --- test: seqfish dataset, reference cluster "Cardiomyocytes" ---
adata2 = sq.datasets.seqfish()
sq.gr.co_occurrence(adata2, cluster_key="celltype_mapped_refined")
res2 = adata2.uns["celltype_mapped_refined_co_occurrence"]
occ2 = res2["occ"]
cats2 = adata2.obs["celltype_mapped_refined"].cat.categories.tolist()
j2 = cats2.index("Cardiomyocytes")
col2 = occ2[:, j2, 0]
order2 = np.argsort(col2)[::-1]
top2 = [cats2[idx] for idx in order2 if cats2[idx] != "Cardiomyocytes"][0]
print("TEST answer:", top2, col2[cats2.index(top2)])
