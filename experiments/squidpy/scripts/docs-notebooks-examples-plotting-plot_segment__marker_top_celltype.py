import numpy as np
import pandas as pd
import squidpy as sq

adata = sq.datasets.mibitof()

for marker in ["CD68", "CD31"]:
    expr = adata[:, marker].X
    arr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
    df = pd.DataFrame({"expr": arr, "Cluster": adata.obs["Cluster"].values})
    means = df.groupby("Cluster")["expr"].mean().sort_values(ascending=False)
    print(marker, "->", means.idxmax())
    print(means)
    print()
