import squidpy as sq
import pandas as pd

adata = sq.datasets.four_i()
adata.var_names_make_unique()

sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.interaction_matrix(adata, cluster_key="cluster")

cats = adata.obs["cluster"].cat.categories.tolist()
im = adata.uns["cluster_interactions"]
df = pd.DataFrame(im, index=cats, columns=cats)

row = df.loc["ER_mitochondria_1"].drop("ER_mitochondria_1")
print("train (ER_mitochondria_1) top partner by shared edges:", row.idxmax(), row.max())

row2 = df.loc["Endosomes_golgi_2"].drop("Endosomes_golgi_2")
print("test (Endosomes_golgi_2) top partner by shared edges:", row2.idxmax(), row2.max())
