import squidpy as sq
import pandas as pd

adata = sq.datasets.imc()
sq.gr.co_occurrence(adata, cluster_key="cell type", n_jobs=1, show_progress_bar=False)

occ = adata.uns["cell type_co_occurrence"]["occ"]
cats = adata.obs["cell type"].cat.categories.tolist()


def top_partner_at_shortest_distance(target):
    ti = cats.index(target)
    ranked = pd.Series(occ[ti, :, 0], index=cats).drop(target).sort_values(ascending=False)
    return ranked.index[0], ranked.iloc[0]


# train
print("train (target=T cells):", top_partner_at_shortest_distance("T cells"))
# test
print("test (target=endothelial):", top_partner_at_shortest_distance("endothelial"))
