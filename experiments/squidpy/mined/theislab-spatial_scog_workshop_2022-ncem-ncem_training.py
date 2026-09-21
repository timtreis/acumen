# mined from: https://github.com/theislab/spatial_scog_workshop_2022/blob/45eca39d4db7a86a48c9e25b5f5ce0eba75f823d/ncem/ncem_training.ipynb
# symbols: squidpy.datasets.mibitof

# %%
# %load_ext autoreload
# %autoreload 2

import ncem
#from ncem.data import get_data_custom, customLoader

import squidpy as sq

# %%
ad = sq.datasets.mibitof()

# %%
trainer = ncem.train.TrainModelInteractions()
trainer.init_estim(log_transform=False)

# %%
trainer.estimator.data = ncem.data.customLoader(
    adata=ad, cluster='Cluster', patient='donor', library_id='library_id', radius=52
)
ncem.data.get_data_custom(interpreter=trainer.estimator)

# %%
trainer.estimator.init_model(n_eval_nodes_per_graph=10)
trainer.estimator.model.training_model.summary()

# %%
trainer.estimator.train(epochs=10)
