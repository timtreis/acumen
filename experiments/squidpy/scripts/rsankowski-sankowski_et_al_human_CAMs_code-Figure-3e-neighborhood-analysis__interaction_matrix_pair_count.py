import squidpy as sq
import pandas as pd


def main():
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    sq.gr.interaction_matrix(adata, cluster_key="cluster")
    mat = adata.uns["cluster_interactions"]
    cats = adata.obs["cluster"].cat.categories.tolist()
    df = pd.DataFrame(mat, index=cats, columns=cats)

    # --- train: Hippocampus <-> Pyramidal_layer spatial-neighbor edge count ---
    print("TRAIN answer (Hippocampus-Pyramidal_layer):", int(df.loc["Hippocampus", "Pyramidal_layer"]))

    # --- test: Fiber_tract <-> Lateral_ventricle spatial-neighbor edge count ---
    print("TEST answer (Fiber_tract-Lateral_ventricle):", int(df.loc["Fiber_tract", "Lateral_ventricle"]))


if __name__ == "__main__":
    main()
