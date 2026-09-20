# mined from: https://github.com/HiDiHlabs/SpatialLeiden-Study/blob/65c182765d2c414d9dee1f9279aa3f93e800625f/2_X_with_SpatialLeiden.ipynb
# symbols: squidpy.gr.spatial_neighbors

# %%
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import squidpy as sq
from spatialleiden import search_resolution_latent, search_resolution_spatial

from utils import get_anndata

# %%
data_dir = Path("./data/LIBD_DLPFC")
result_dir = Path("./results/SpatialLeiden_X")

seed = 42

# %%
metadata = pd.read_table(
    data_dir / "samples.tsv", usecols=["directory", "n_clusters"]
).set_index("directory")

# %%
sample = metadata.loc["Br8100_151673", :]

sample_dir = data_dir / sample.name

n_clusters = sample.n_clusters

# %%
spatial_weights = [0.2, 0.4, 0.6, 0.8, 1.0]

# %%
spicemix_dir = Path("./results/SpiceMix") / "Br8100_151673"

# %%
for f in spicemix_dir.glob("*.h5ad"):
    cols = []
    adata = ad.read_h5ad(f)
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sc.pp.neighbors(adata, use_rep="normalized_X")

    name = "Leiden"
    res = search_resolution_latent(adata, n_clusters, random_state=seed)
    adata.obs[name] = adata.obs["leiden"].copy()
    cols.append(name)

    for w in spatial_weights:
        _ = search_resolution_spatial(
            adata,
            n_clusters,
            directed=(False, False),
            layer_ratio=w,
            resolution=(res, 1),
            seed=seed,
        )

        name = f"SpatialLeiden_w{w:.1f}"
        cols.append(name)
        adata.obs[name] = adata.obs["spatialleiden"].copy()
    adata.obs[cols].to_csv(spicemix_dir / f"{f.stem}_cluster.tsv", sep="\t", index=True)

# %%
banksy_dir = Path("./results/Banksy") / "BR8100_151673"

# %%
adata = get_anndata(sample_dir)
sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

for f in banksy_dir.glob("*.tsv"):
    if "cluster" in f.stem:
        continue
    cols = []
    adata.obsm["Banksy"] = pd.read_table(f, index_col=0).to_numpy()
    sc.pp.neighbors(adata, use_rep="Banksy")

    name = "Leiden"
    res = search_resolution_latent(adata, n_clusters, random_state=seed)
    adata.obs[name] = adata.obs["leiden"].copy()
    cols.append(name)

    for w in spatial_weights:
        _ = search_resolution_spatial(
            adata,
            n_clusters,
            directed=(False, False),
            layer_ratio=w,
            resolution=(res, 1),
            seed=seed,
        )

        name = f"SpatialLeiden_w{w:.1f}"
        cols.append(name)
        adata.obs[name] = adata.obs["spatialleiden"].copy()
    adata.obs[cols].to_csv(banksy_dir / f"{f.stem}_cluster.tsv", sep="\t", index=True)
