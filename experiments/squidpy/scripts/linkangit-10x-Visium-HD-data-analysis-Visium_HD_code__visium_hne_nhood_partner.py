import squidpy as sq
import pandas as pd


def main():
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sq.gr.nhood_enrichment(adata, cluster_key="cluster", seed=0, n_jobs=1)

    cats = adata.obs["cluster"].cat.categories.tolist()
    z = adata.uns["cluster_nhood_enrichment"]["zscore"]
    df = pd.DataFrame(z, index=cats, columns=cats)

    for target in ["Hippocampus", "Striatum"]:
        row = df.loc[target].drop(target).sort_values(ascending=False)
        print(f"TRAIN/TEST target={target} -> most enriched partner: {row.index[0]} (z={row.iloc[0]:.3f}); "
              f"runner-up: {row.index[1]} (z={row.iloc[1]:.3f})")

    # train answer: Hippocampus -> Pyramidal_layer
    # test answer: Striatum -> Lateral_ventricle


if __name__ == "__main__":
    main()
