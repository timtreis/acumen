import os
os.environ.setdefault("TMPDIR", "/tmp")

import squidpy as sq

# ---- TRAIN: imc dataset, 'cell type' annotation ----
adata = sq.datasets.imc()
sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, "cell type", n_jobs=1, seed=0, show_progress_bar=False)

cats = adata.obs["cell type"].cat.categories.tolist()
zscore = adata.uns["cell type_nhood_enrichment"]["zscore"]
n = len(cats)
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((zscore[i, j], cats[i], cats[j]))
pairs.sort(reverse=True)
print("TRAIN top pairs:")
for p in pairs[:5]:
    print(p)

# ---- TEST: mibitof dataset, 'Cluster' annotation, multiple libraries ----
adata2 = sq.datasets.mibitof()
sq.gr.spatial_neighbors(adata2, library_key="library_id")
sq.gr.nhood_enrichment(adata2, "Cluster", n_jobs=1, seed=0, show_progress_bar=False)

cats2 = adata2.obs["Cluster"].cat.categories.tolist()
zscore2 = adata2.uns["Cluster_nhood_enrichment"]["zscore"]
n2 = len(cats2)
pairs2 = []
for i in range(n2):
    for j in range(i + 1, n2):
        pairs2.append((zscore2[i, j], cats2[i], cats2[j]))
pairs2.sort(reverse=True)
print("TEST top pairs:")
for p in pairs2[:5]:
    print(p)
