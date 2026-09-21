# mined from: https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/blob/7c4148c59049563554708e01c378352586ab2f4a/notebooks/05_spatial_demo.ipynb
# symbols: squidpy.datasets.visium_hne_adata, squidpy.gr.co_occurrence, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import rapids_singlecell as rsc
import cupy as cp
import squidpy as sq
import numpy as np
import anndata as ad

# %%
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(
    managed_memory=False,  # Allows oversubscription
    pool_allocator=True,  # default is False
    devices=[0,1],  # GPU device IDs to register. By default registers only GPU 0.
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# %%
adata = sq.datasets.visium_hne_adata()
sq.gr.spatial_neighbors(adata)

# %%
genes = adata.var_names

# %%
# %%time
rsc.gr.spatial_autocorr(adata, 
                        mode="moran", 
                        connectivity_key="connectivities", 
                        dtype=np.float32, 
                        genes=genes, 
                        n_perms=100, 
                        use_sparse=True)

# %%
# %%time
rsc.gr.spatial_autocorr(adata, 
                        mode="geary",
                        connectivity_key="connectivities",
                        genes=genes,
                        dtype=np.float32, 
                        n_perms=100, 
                        use_sparse=True)

# %%
# %%time
rsc.gr.co_occurrence(adata, cluster_key="leiden") #calculate time for cooccurrence using rapids singlecell

# %%
# %%time
sq.gr.co_occurrence(adata, cluster_key="leiden") #calculate wall time for cooccurrence using squidpy

# %%
sq.pl.co_occurrence(adata, cluster_key="leiden", clusters="0")

# %%
# These notebooks are very GPU memory intensive!
# In order to free up GPU memory, we'll kill this kernel prior to proceeding.  You will get a message.  This is expected.
# If you have a CUDA or an Out Of Memory (OOM) error, please kill all kernels to free up your GPU memory and try again!
# You can comment this out if you want to continue exploring the notebook.
# Please consult the README for more tips and tricks.

import IPython

IPython.Application.instance().kernel.do_shutdown(True)
