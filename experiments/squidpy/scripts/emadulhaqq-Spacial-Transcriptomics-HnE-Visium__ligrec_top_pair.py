import squidpy as sq


def top_pair(adata, source, target):
    res = sq.gr.ligrec(
        adata,
        n_perms=100,
        cluster_key="cluster",
        seed=0,
        n_jobs=1,
        copy=True,
    )
    means = res["means"]
    sub = means[(source, target)].dropna().sort_values(ascending=False)
    top = sub.index[0]
    return top, round(float(sub.iloc[0]), 2)


def train():
    adata = sq.datasets.visium_hne_adata()
    pair, val = top_pair(adata, "Hippocampus", "Pyramidal_layer")
    print("TRAIN (visium_hne, Hippocampus->Pyramidal_layer) top LR pair:", pair, val)


def test():
    adata = sq.datasets.visium_fluo_adata()
    pair, val = top_pair(adata, "Dentate_gyrus", "Hippocampus")
    print("TEST (visium_fluo, Dentate_gyrus->Hippocampus) top LR pair:", pair, val)


if __name__ == "__main__":
    train()
    test()
