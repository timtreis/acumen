# mined from: https://github.com/HiDiHlabs/SpatialLeiden-Study/blob/65c182765d2c414d9dee1f9279aa3f93e800625f/2_Leiden_DLPFC.ipynb
# symbols: squidpy.gr.spatial_autocorr, squidpy.gr.spatial_neighbors

# %%
from pathlib import Path

import pandas as pd
import scanpy as sc
import squidpy as sq
from multispaeti import MultispatiPCA
from spatialleiden import (
    search_resolution,
    search_resolution_latent,
    search_resolution_spatial,
)

from utils import get_anndata, preprocess_anndata

# %%
data_dir = Path("./data/LIBD_DLPFC")
result_dir = Path("./results/LIBD_DLPFC")

seed = 42

# %%
metadata = pd.read_table(
    data_dir / "samples.tsv", usecols=["directory", "n_clusters"]
).set_index("directory")

# %%
sample = metadata.loc["Br8100_151673", :]

# %%
out_dir = result_dir / "weightratio_impact" / sample.name

n_genes = 3_000
n_pcs = 30

# %%
adata = get_anndata(data_dir / sample.name)
preprocess_anndata(adata, genes=n_genes, n_pcs=n_pcs, seed=seed)

sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
sc.pp.neighbors(adata, random_state=seed)

res = search_resolution_latent(adata, sample.n_clusters, start=0.6, random_state=seed)
leiden_df = adata.obs[["leiden"]].copy()
leiden_df.columns = ["label"]

out_dir.mkdir(parents=True, exist_ok=True)
leiden_df.to_csv(out_dir / "leiden.tsv", sep="\t", index_label="")

sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

for weight_ratio in [0, 0.2, 0.4, 0.6, 0.8, 1, 5, 10]:
    res_multi = search_resolution_spatial(
        adata,
        sample.n_clusters,
        resolution=(res, 1),
        directed=(False, False),
        layer_ratio=weight_ratio,
        seed=seed,
    )

    multiplex_df = adata.obs[["spatialleiden"]].copy()
    multiplex_df.columns = ["label"]
    multiplex_df.to_csv(
        out_dir / f"spatial_leiden_w{weight_ratio:.1f}.tsv", sep="\t", index_label=""
    )

# %%
n_pcs = 30
n_genes = 3_000
weight_spatial = 0.7

# %%
for name, sample in metadata.iterrows():
    print("Processing " + name)

    sample_dir = data_dir / name
    out_dir = result_dir / name

    adata = get_anndata(sample_dir)
    preprocess_anndata(adata, genes=n_genes, n_pcs=n_pcs, seed=seed)

    sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(adata, random_state=seed)

    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    _ = search_resolution(
        adata,
        sample.n_clusters,
        latent_kwargs={"random_state": seed},
        spatial_kwargs={
            "directed": (False, False),
            "layer_ratio": weight_spatial,
            "seed": seed,
        },
    )

    label_leiden = adata.obs[["leiden"]].copy()
    label_leiden.columns = ["label"]

    label_leiden_multi = adata.obs[["spatialleiden"]].copy()
    label_leiden_multi.columns = ["label"]

    ## Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    label_leiden.to_csv(out_dir / "leiden.tsv", sep="\t", index_label="")
    label_leiden_multi.to_csv(out_dir / "spatial_leiden.tsv", sep="\t", index_label="")

# %%
for name, sample in metadata.iterrows():
    print("Processing " + name)

    sample_dir = data_dir / name
    out_dir = result_dir / name

    adata = get_anndata(sample_dir)
    preprocess_anndata(adata, genes=n_genes, n_pcs=n_pcs, seed=seed)
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sq.gr.spatial_autocorr(adata, genes=adata.var_names, mode="moran", seed=seed)
    genes = adata.uns["moranI"].nlargest(n_genes, columns="I", keep="all").index
    adata.obsm["X_svg_pca"] = sc.tl.pca(
        adata[:, genes].X, n_comps=n_pcs, random_state=seed
    )
    sc.pp.neighbors(adata, use_rep="X_svg_pca", random_state=seed)

    # Multiplex
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    _ = search_resolution(
        adata,
        sample.n_clusters,
        latent_kwargs={"random_state": seed},
        spatial_kwargs={
            "directed": (False, False),
            "layer_ratio": weight_spatial,
            "seed": seed,
        },
    )

    label_leiden = adata.obs[["leiden"]].copy()
    label_leiden.columns = ["label"]

    label_leiden_multi = adata.obs[["spatialleiden"]].copy()
    label_leiden_multi.columns = ["label"]

    # Multiplex and MULTISPATI-PCA
    adata.obsm["X_mspca"] = MultispatiPCA(
        n_pcs, connectivity=adata.obsp["connectivities"]
    ).fit_transform(adata[:, genes].X.toarray())
    sc.pp.neighbors(adata, use_rep="X_mspca", random_state=seed)

    _ = search_resolution(
        adata,
        sample.n_clusters,
        latent_kwargs={"random_state": seed},
        spatial_kwargs={
            "directed": (False, False),
            "layer_ratio": weight_spatial,
            "seed": seed,
        },
    )

    label_leiden_msPCA = adata.obs[["leiden"]].copy()
    label_leiden_msPCA.columns = ["label"]

    label_leiden_multi_msPCA = adata.obs[["spatialleiden"]].copy()
    label_leiden_multi_msPCA.columns = ["label"]

    ## Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    label_leiden.to_csv(out_dir / "leiden_svg.tsv", sep="\t", index_label="")
    label_leiden_multi.to_csv(
        out_dir / "spatial_leiden_svg.tsv", sep="\t", index_label=""
    )
    label_leiden_msPCA.to_csv(
        out_dir / "leiden_svg_multispati.tsv", sep="\t", index_label=""
    )
    label_leiden_multi_msPCA.to_csv(
        out_dir / "spatial_leiden_svg_multispati.tsv", sep="\t", index_label=""
    )
