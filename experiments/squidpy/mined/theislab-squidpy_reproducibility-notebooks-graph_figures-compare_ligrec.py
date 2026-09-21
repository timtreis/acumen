# mined from: https://github.com/theislab/squidpy_reproducibility/blob/407b151c1b7d657dd46a99cd9d924b13ec2d9afa/notebooks/graph_figures/compare_ligrec.ipynb
# symbols: squidpy.gr.ligrec

# %%
import sys
import cloudpickle as pickle
from time import process_time
from collections import defaultdict

from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scanpy as sc
import squidpy as sq
import scvelo as scv
import seaborn as sns

# !pip install sqlalchemy fbpca geosketch werkzeug
sys.path.insert(0, './cellphonedb/src/api_endpoints/terminal_api/method_terminal_api_endpoints/')
from method_terminal_commands import statistical_analysis

# %%
def save_data(adata):
    df_expr_matrix = adata.raw.X
    df_expr_matrix = df_expr_matrix.T
    df_expr_matrix = pd.DataFrame(df_expr_matrix)

    df_expr_matrix.columns= adata.obs.index
    df_expr_matrix.set_index(adata.raw.var.index, inplace=True) 
    df_expr_matrix.to_csv('counts.txt', sep='\t')
    
    df_meta = pd.DataFrame(data={'Cell': list(adata.obs_names), 'cell_type': list(adata.obs['Clusters'])})
    df_meta.set_index('Cell', inplace=True)
    df_meta.to_csv('meta.txt', sep='\t')

# %%
n_genes = (1000, 5000, 10000, 15000, 20000, 25000, 30000, 32738)
n_jobs = 32  # used for CellPhoneDB
n_perms = 1000
n_tests = 10
threshold = 0.1
cluster_key = 'Clusters'

# %%
adata = scv.datasets.forebrain()  # human forebrain development
adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
adata.var_names_make_unique()
adata.X = adata.X.A  # densify
adata

# %%
sc.pp.normalize_per_cell(adata)
adata.raw = adata.copy()

# %%
rs = np.random.RandomState(seed=42)
geness = [rs.choice(adata.var_names, size=n_gene, replace=False) for n_gene in n_genes]

# %%
mean_results = defaultdict(lambda: defaultdict(dict))  # method - n_genes - split
pval_results = defaultdict(lambda: defaultdict(dict))
time_results = defaultdict(lambda: defaultdict(dict))

# %%
for genes in geness:
    print(len(genes))
    for i in range(n_tests):
        print(i)
        bdata = adata[:, genes].copy()
        save_data(bdata)

        start_t = process_time()
        statistical_analysis(
            meta_filename='meta.txt', counts_filename='counts.txt', threshold=threshold, threads=n_jobs,
            debug_seed=0, iterations=n_perms, result_precision='4', output_path='out',
            counts_data='gene_name'
        )
        duration = process_time() - start_t
        
        pval_results['cellphonedb'][bdata.n_vars][i] = pd.read_csv('out/pvalues.csv')
        mean_results['cellphonedb'][bdata.n_vars][i] = pd.read_csv('out/means.csv')
        time_results['cellphonedb'][bdata.n_vars][i] = duration

# %%
1

# %%
for genes in geness:
    print(len(genes))
    for i in range(n_tests):
        print(i)
        bdata = adata[:, genes].copy()

        start_t = process_time()
        res = sq.gr.ligrec(
            adata,
            n_perms=n_perms,
            threshold=threshold,
            cluster_key="Clusters",
            copy=True,
            use_raw=False,
            show_progress_bar=False,
            interactions_params={'resources': 'CellPhoneDB'},
            transmitter_params={"categories": "ligand"},
            receiver_params={"categories": "receptor"},
            numba_parallel=False,
            n_jobs=n_jobs,
        )
        duration = process_time() - start_t
        
        pval_results['squidpy'][bdata.n_vars][i] = res['pvalues']
        mean_results['squidpy'][bdata.n_vars][i] = res['means']
        time_results['squidpy'][bdata.n_vars][i] = duration

# %%
dfs = []
for meth, split in time_results.items():
    tmp = pd.DataFrame(split)
    mean, std = tmp.mean(), tmp.std()
    ms = pd.DataFrame([[val for pair in zip(mean, std) for val in pair]],
                      columns=[f"{c} {k}" for c in mean.index for k in ["mean", "std"]],
                      index=[meth])
    dfs.append(ms)
    
df = pd.concat(dfs)
df

# %%
with open("resuls.pickle", "wb") as fout:
    pickle.dump((pval_results, mean_results, time_results, df), fout)

# %%
with open("resuls.pickle", "rb") as fin:
    pvals_results, mean_results, time_results, df = pickle.load(fin)

# %%
dff = df.iloc[:, ::2]
dff

# %%
dff = pd.DataFrame({"runtime": np.r_[dff.values[1], dff.values[0]],
                    "algorithm": ["Squidpy"] * len(dff.T) + ["CellPhoneDB"] * len(dff.T),
                    "#genes": [1000, 5000, 10000, 15000, 20000, 25000, 30000, 32738] * 2})

# %%
fig, ax = plt.subplots(tight_layout=True, dpi=300)
sns.pointplot(x='#genes', y='runtime', hue="algorithm", data=dff,
                linewidth=1,
                ax=ax,
                palette=None)
ax.set_ylabel("runtime (s)")
fig.savefig("ligrec_runtime.png")
