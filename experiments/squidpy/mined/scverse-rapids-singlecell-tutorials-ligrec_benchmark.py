# mined from: https://github.com/scverse/rapids-singlecell-tutorials/blob/d24597eff5a528dbb54aa4298730cc3d0e1cd011/ligrec_benchmark.ipynb
# symbols: squidpy.gr.ligrec

# %%
import scanpy as sc
import squidpy as sq
import cupy as cp
import rapids_singlecell as rsc

import warnings

warnings.filterwarnings("ignore")

# %%
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(
    managed_memory=False,  # Keep allocations in device memory
    pool_allocator=False,  # default is False
    devices=0,  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# %%
# %%time
adata = sc.read("h5/adata.raw.h5ad")

# %%
rsc.get.anndata_to_GPU(adata)

# %%
# %%time
adata.var["MT"] = adata.var_names.str.startswith("MT-")

# %%
# %%time
rsc.pp.calculate_qc_metrics(adata, qc_vars=["MT"])

# %%
# %%time
adata = adata[adata.obs["n_genes_by_counts"] < 5000]
adata.shape

# %%
# %%time
adata = adata[adata.obs["pct_counts_MT"] < 20]
adata.shape

# %%
# %%time
rsc.pp.filter_genes(adata, min_cells=3)

# %%
# %%time
rsc.pp.normalize_total(adata, target_sum=1e4)

# %%
# %%time
rsc.pp.log1p(adata)

# %%
# %%time
rsc.get.anndata_to_CPU(adata)
adata.raw = adata

# %%
adata

# %%
interactions = rsc.squidpy_gpu._ligrec._get_interactions()

# %%
# %%time
res_rsc = rsc.gr.ligrec(
    adata,
    n_perms=1000,
    interactions=interactions,
    cluster_key="CellType",
    copy=True,
    use_raw=True,
)

# %%
res_rsc["means"].iloc[:10, :10]

# %%
res_rsc["pvalues"].iloc[:10, :10]

# %%
# %%time
res_sq = sq.gr.ligrec(
    adata,
    n_perms=1000,
    interactions=interactions,
    cluster_key="CellType",
    copy=True,
    use_raw=True,
    n_jobs=32,
)

# %%
res_sq["means"].iloc[:10, :10]

# %%
res_sq["pvalues"].iloc[:10, :10]
