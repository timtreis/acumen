import os
os.environ.setdefault("TMPDIR", "/tmp")

import squidpy as sq
import warnings
warnings.filterwarnings("ignore")


def top_pair(adata, cluster_key):
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_perms=1000, seed=0, n_jobs=1)
    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    cats = adata.obs[cluster_key].cat.categories.tolist()
    pairs = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            pairs.append((zscore[i, j], cats[i], cats[j]))
    pairs.sort(reverse=True)
    return pairs[:5]


def main():
    imc = sq.datasets.imc()
    print("TRAIN (imc):")
    for p in top_pair(imc, "cell type"):
        print("  ", p)

    seqfish = sq.datasets.seqfish()
    print("TEST (seqfish):")
    for p in top_pair(seqfish, "celltype_mapped_refined"):
        print("  ", p)


if __name__ == "__main__":
    main()
