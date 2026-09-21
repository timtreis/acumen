import os

os.environ["TMPDIR"] = "/tmp"

import squidpy as sq

adata = sq.datasets.visium_hne_adata()
res = sq.gr.ligrec(
    adata,
    cluster_key="cluster",
    n_perms=100,
    seed=0,
    n_jobs=1,
    copy=True,
    show_progress_bar=False,
)
means = res["means"]
pvals = res["pvalues"]


def top_significant_pair(source, target, alpha=0.01):
    col = (source, target)
    m = means[col].dropna()
    p = pvals[col]
    sig = m.index[p[m.index] < alpha]
    m_sig = m.loc[sig].sort_values(ascending=False)
    top = m_sig.index[0]
    return f"{top[0]}-{top[1]}", m_sig.iloc[0]


train_answer, train_val = top_significant_pair("Hippocampus", "Pyramidal_layer")
test_answer, test_val = top_significant_pair("Cortex_1", "Cortex_2")

print("train (Hippocampus -> Pyramidal_layer) ->", train_answer, train_val)
print("test (Cortex_1 -> Cortex_2) ->", test_answer, test_val)
