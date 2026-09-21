# mined from: https://github.com/estorrs/multiplex-imaging-pipeline/blob/c351da1d41e284eef2f732b3553e5602438ed978/notebooks/batch_neighborhood_analysis_v1.ipynb
# symbols: squidpy.gr.co_occurrence, squidpy.gr.interaction_matrix, squidpy.gr.spatial_neighbors, squidpy.pl.co_occurrence, squidpy.pl.interaction_matrix

# %%
import logging

import scanpy as sc
import numpy as np
import pandas as pd
import scipy
import anndata
import squidpy as sq
import matplotlib.pyplot as plt
import tifffile
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from umap import UMAP

# %%
from mgitools.os_helpers import listfiles

# %%
# %load_ext autoreload

# %%
# %autoreload 2

# %%
from mip.gating import get_ideal_window

# %%
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# %%
fps = sorted(listfiles('/diskmnt/Projects/Users/estorrs/multiplex_data/codex/htan/', regex=r'cell_annotation_full.h5ad$'))
fps

# %%
def cell_to_neighbors(adata, radius=50):
    X = adata.obs[['centroid_row', 'centroid_col']].values
    nbrs = NearestNeighbors(algorithm='ball_tree').fit(X)
    
    g = nbrs.radius_neighbors_graph(X, radius=radius)
    rows, cols, _ = scipy.sparse.find(g)
    
    cell_to_neighbhors = {}
    for r, c in zip(rows, cols):
        cid = adata.obs.index[r]
        if cid not in cell_to_neighbhors:
            cell_to_neighbhors[cid] = []
        else:
            cell_to_neighbhors[cid].append(adata.obs.index[c])
            
    return cell_to_neighbhors


# %%
sample_to_adata = {}
cell_to_nbhrs = {}
for fp in fps:
    sample = fp.split('/')[-3]
    a = sc.read_h5ad(fp)
    print(sample, a.shape)
    
    a.obs['sample_id'] = sample
    a.obs.index = [f'{sample}_{x}' for x in a.obs.index.to_list()]
    cell_to_nbhrs.update(cell_to_neighbors(a, radius=50))
    sample_to_adata[sample] = a

# %%
fps = sorted(listfiles('/diskmnt/Projects/Users/estorrs/multiplex_data/codex/htan/', regex=r'pseudo.tiff$'))
fps

# %%
sample_to_pseudo = {fp.split('/')[-3]:tifffile.imread(fp) for fp in fps}

# %%
sample_to_adata.keys()

# %%
cells = []
docs = []
for s, a in sample_to_adata.items():
    cell_to_cell_type = {c:ct for c, ct in zip(a.obs.index, a.obs['cell_type'])}
    docs += [[cell_to_cell_type[neighbor] for neighbor in cell_to_nbhrs[cell_id]]
            for cell_id in a.obs.index.to_list()]
    print(s, len(docs))
    cells += a.obs.index.to_list()

# %%
dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# %%
len(dictionary), len(corpus), len(cells)

# %%
num_topics = 10
chunksize = len(corpus)
passes = 2
iterations = 100
eval_every = 10 # turn this on to see how well everything is converging. off by default bc is takes time

# %%
temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

# %%
top_topics = model.top_topics(corpus)
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics

# %%
def transformed_corpus_to_emb(tc, n_topics):
    embs = []
    for entity in tc:
        default = [0] * n_topics
        for topic, value in entity:
            default[topic] = value
        embs.append(default)
    return np.asarray(embs)
    

# %%
transformed = model[corpus]
embs = transformed_corpus_to_emb(transformed, num_topics)
embs.shape

# %%
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# %%
kmeans = KMeans(n_clusters=20, random_state=0).fit(embs)
set(kmeans.labels_)

# %%
df = pd.DataFrame(data=embs, columns=np.arange(num_topics), index=cells)
lda_adata = anndata.AnnData(df)
lda_adata

# %%
lda_adata.obs['LDA_kmeans_cluster'] = [str(x) for x in kmeans.labels_]

# %%
sc.pl.matrixplot(lda_adata, var_names=lda_adata.var.index, groupby='LDA_kmeans_cluster', dendrogram=True)

# %%
topic_df = pd.DataFrame(data=model.get_topics(), columns=[dictionary.get(i) for i in range(len(dictionary))],
                        index=np.arange(num_topics))
import seaborn as sns
sns.clustermap(topic_df, cmap='Blues')

# %%
metacluster_to_cluster = {
    'Tumor - Pure': [0, 15],
    'Tumor - Infiltrating TAM': [7],
    'Tumor - Infiltrating Stroma': [3, 17, 2, 11],
    'Tumor - Infiltrating T cell': [10],
    'Myoepithelium': [4, 19],
    'Endothelial': [6],
    'Fibroblast': [13, 16],
    'Immune - T cell': [8],
    'Mixed Stroma': [1],
    'Immune - Mixed': [9, 12],
    'Immune - DC': [18],
    'Noise': [5, 14]
}
cluster_to_metacluster = {str(v):k for k, vs in metacluster_to_cluster.items() for v in vs}
sorted(cluster_to_metacluster.items())

# %%
cell_to_kmeans = {c:str(k) for c, k in zip(cells, kmeans.labels_)}
cell_to_metacluster = {c:cluster_to_metacluster[cell_to_kmeans[c]]
                      for c in cells}
for s, a in sample_to_adata.items():
    f = df.loc[a.obs.index.to_list()]
    a.obsm['X_lda'] = f.values
    a.obs['LDA_kmeans_cluster'] = [cell_to_kmeans[c] for c in a.obs.index.to_list()]
    a.obs['metacluster'] = [cell_to_metacluster[c] for c in a.obs.index.to_list()]

# %%
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 120

# %%
a.uns['metacluster_colors']

# %%
def show_cluster(adata, cluster, cluster_col='metacluster', radius=300):
    r1, r2, c1, c2 = get_ideal_window(
        adata, radius=radius, cell_type=cluster, cell_type_col=cluster_col,
        return_filtered=False)
    sc.pl.spatial(adata, color=cluster_col, crop_coord=[c1, c2, r1, r2], size=1.)
    return r1, r2, c1, c2
    
def display_on_img(adata, img, cluster, cluster_col='metacluster', radius=300, show_all=False,
                  pallete=sns.color_palette('tab20'), s=5, edgecolors='black', ax=None, legend=True,
                  pallete_map=None):
    f, (r1, r2, c1, c2) = get_ideal_window(
        adata, radius=radius, cell_type=cluster, cell_type_col=cluster_col,
        return_filtered=True)
    
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(img[r1:r2, c1:c2])
    
    if show_all:
        f = adata[((adata.obs['centroid_row']>r1)&(adata.obs['centroid_row']<r2))]
        f = f[((f.obs['centroid_col']>c1)&(f.obs['centroid_col']<c2))]
        
        for ct, color in zip(sorted(set(f.obs[cluster_col])), pallete):
            fx = f[f.obs[cluster_col]==ct]
            if pallete_map is not None:
                color = pallete_map[ct]
            ax.scatter(fx.obs['centroid_col'] - c1, fx.obs['centroid_row'] - r1, c=color, label=ct, s=s,
                      edgecolors=edgecolors)
            
        if legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.scatter(f.obs['centroid_col'] - c1, f.obs['centroid_row'] - r1, c='red', s=s,
                  edgecolors=edgecolors)
    return r1, r2, c1, c2

# %%
s = 'HT206B1-H1'
a = sample_to_adata[s]
a.shape

# %%
display_on_img(a, sample_to_pseudo[s], 'Tumor - Infiltrating T cell', cluster_col='metacluster', radius=1000,
               show_all=True, s=1, edgecolors=None)

# %%
for s, a in sample_to_adata.items():
    print(s)
    display_on_img(a, sample_to_pseudo[s], metacluster, cluster_col='metacluster', radius=300)
    
    plt.show()

# %%
metacluster = 'Tumor - Infiltrating T cell'

fig, axs = plt.subplots(ncols=len(sample_to_adata), figsize=(20, 5))
m = {}
for s, a in sample_to_adata.items():
    m.update({ct:c for ct, c in zip(sorted(set(a.obs['metacluster'])), sns.color_palette('tab20'))})
for i, (s, a) in enumerate(sample_to_adata.items()):
    print(s)
    ax = axs[i]
    display_on_img(a, sample_to_pseudo[s], metacluster, cluster_col='metacluster', radius=300, show_all=False,
                   s=5, ax=ax, edgecolors=None)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %%
metacluster = 'Tumor - Pure'

fig, axs = plt.subplots(ncols=len(sample_to_adata), figsize=(20, 5))
m = {}
for s, a in sample_to_adata.items():
    m.update({ct:c for ct, c in zip(sorted(set(a.obs['metacluster'])), sns.color_palette('tab20'))})
for i, (s, a) in enumerate(sample_to_adata.items()):
    print(s)
    ax = axs[i]
    display_on_img(a, sample_to_pseudo[s], metacluster, cluster_col='metacluster', radius=500, show_all=False,
                   s=5, ax=ax, edgecolors=None)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %%
metacluster = 'Tumor - Infiltrating T cell'
for s, a in sample_to_adata.items():
    print(s)
    display_on_img(a, sample_to_pseudo[s], metacluster, cluster_col='metacluster', radius=300, show_all=True,
                   s=20, pallete_map=m)
    plt.show()

# %%
metacluster = 'Noise'
for s, a in sample_to_adata.items():
    print(s)
    display_on_img(a, sample_to_pseudo[s], metacluster, cluster_col='metacluster', radius=300, show_all=True,
                   s=20)
    plt.show()

# %%


# %%
for s, a in sample_to_adata.items():
    fp = f'/diskmnt/Projects/Users/estorrs/multiplex_data/codex/htan/{s}/level_4/metacluster_lda.h5ad'
    a.write_h5ad(fp)

# %%
from collections import Counter
data, idxs = [], []
cols = sorted(set(cell_to_metacluster.values()))
for s, a in sample_to_adata.items():
    counts = Counter(a.obs['metacluster'])
    data.append([counts.get(c, 0) for c in cols])
    idxs.append(s)
df = pd.DataFrame(data=data, index=idxs, columns=cols)
df

# %%
ax = df.plot(kind='bar', stacked=True, color=sns.color_palette('tab20'))
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# %%
ax = (df / df.sum(axis=1).values.reshape(-1, 1)).plot(kind='bar', stacked=True, color=sns.color_palette('tab20'))
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# %%
data, idxs = [], []
cols = sorted(set(a.obs['cell_type']))
for s, a in sample_to_adata.items():
    counts = Counter(a.obs['cell_type'])
    data.append([counts.get(c, 0) for c in cols])
    idxs.append(s)
df = pd.DataFrame(data=data, index=idxs, columns=cols)
df

# %%
ax = df.plot(kind='bar', stacked=True, color=sns.color_palette('tab20'))
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# %%
ax = (df / df.sum(axis=1).values.reshape(-1, 1)).plot(kind='bar', stacked=True, color=sns.color_palette('tab20'))
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# %%


# %%
sq.gr.spatial_neighbors(a, key_added='spatial')

# %%
sq.gr.interaction_matrix(a, cluster_key="metacluster")

# %%
sq.pl.interaction_matrix(a, cluster_key="metacluster", vmax=10000)

# %%
sq.gr.co_occurrence(a, cluster_key="metacluster", n_splits=1, n_jobs=40,
                    interval=[32, 64, 128, 256, 512, 1028])

# %%
sq.pl.co_occurrence(
    a,
    cluster_key="metacluster",
    clusters=["Myoepithelium"],
    figsize=(15, 4),
)

# %%
sq.pl.co_occurrence(
    a,
    cluster_key="metacluster",
    clusters=["Immune - T cell"],
    figsize=(15, 4),
)

# %%
for c in sorted(set(a.obs['metacluster'])):
    sq.pl.co_occurrence(
        a,
        cluster_key="metacluster",
        clusters=[c],
#         dpi=220
        figsize=(15, 4),
    )

# %%
fps = sorted(listfiles('/diskmnt/Projects/Users/estorrs/multiplex_data/codex/htan/', regex=r'metacluster_lda.h5ad$'))
fps

# %%
sample_to_adata = {fp.split('/')[-3]:sc.read_h5ad(fp) for fp in fps}

# %%
for s, a in sample_to_adata.items():
    print(s)
    sq.gr.spatial_neighbors(a, key_added='spatial')
    sq.gr.interaction_matrix(a, cluster_key="metacluster")
    sq.gr.co_occurrence(a, cluster_key="metacluster", n_splits=1, n_jobs=40, interval=[50, 100, 200, 500, 1000])
    sq.gr.interaction_matrix(a, cluster_key="cell_type")
    sq.gr.co_occurrence(a, cluster_key="cell_type", n_splits=1, n_jobs=40, interval=[50, 100, 200, 500, 1000])
    a.write_h5ad(f'/diskmnt/Projects/Users/estorrs/multiplex_data/codex/htan/{s}/level_4/metacluster_spatial_analysis.h5ad')
    

# %%


# %%


# %%


# %%

