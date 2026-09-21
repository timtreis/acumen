# mined from: https://github.com/beelabhmc/spatial_cancer_thesis/blob/f556f995245451e0feab09b0e2854214e9473022/image_extraction_tester.py
# symbols: squidpy.im.ImageContainer

# Developement ground for the image extraction pipeline
# relies on image_extracter.py

# %% Load in libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt
import numpy as np



from image_extracter import image_extracter

# %% Load in data
# s8t2_adata = ad.read_h5ad("./intermediate_data/33D_S8T2.h5ad")
s8t2_adata = ad.read_h5ad("./intermediate_data/s8t2_all_at_once_enrichments.h5ad")

# %% Set up image extracter
# extracter = image_extracter()
# extracter.prep_model()
#%% Testing
# import cv2

# test_tile = cv2.imread(R"intermediate_data/patched_data/V10F03-033_A/patches/V10F03-033_A_AAACACCAATAACTGC-1_59_19.jpg")
# extracter.extract_one(test_tile)

# # %% Set up image container with image
# im_container = sq.im.ImageContainer()
# im_container = im_container.from_adata(s8t2_adata)
# This initializes with the hires image
# We want an image container full of each spot image
# Use the spot crop generater for this

# %%
# big_im_container = sq.im.ImageContainer()
# big_im_container.add_img(R".\original_data\High-resolution_tissue_images\V10F03-033\201210_BC_V10F03-033_S8C-T_RJ.D1-Spot000001.jpg")
# # Seems to use the adata.uns and the spatial coords in adata to get image?
# generator = big_im_container.generate_spot_crops(s8t2_adata, 
#                                                  return_obs=True)
# # Gets tuple of spot image and image container
# spot_image = generator.__next__()
# # Generates to size 377 here, based on math for the slide

# # Reshape out an extra dimension
# plt.imshow(spot_image[0]['image_0'].to_numpy().reshape((377,377,3)))
# spot_image[0].show() # Also works

# new_extracter = image_extracter(image_size= (377,377,3))
# new_extracter.prep_model()

# # Works!
# new_extracter.extract_one(
#     spot_image[0]['image_0']
# )
# # %% Test if underlying functions for feature extraction work
# spot_image[0].features_custom(func = new_extracter.extract_one, layer = 'image_0')
# %%[markdown]
# This shows that we gnerate the features just fine, but as a single entry for each spot which is a single tensor
# %% 
# Does this work?
# sq.im.calculate_image_features(
#     s8t2_adata, 
#     big_im_container,
#     features= 'summary'
# )

# %% 
# Untested
# The wrapper should set adata.obsm
# Doesn't seem to work yet
# Crashes repeatedly
# sq.im.calculate_image_features(
#     adata = s8t2_adata,
#     img = spot_image[0],
#     features = 'custom',
#     key_added = 'features',
#     # n_jobs = 4,
#     features_kwargs={"custom": {"func":  new_extracter.extract_one}})


#   Look at this guide for help? https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html
# %%[markdown]
# ## For loop method
# %% Attempt to manually run through and set obsm
# Initialize the output array
# s8t2_adata.obsm['im_features'] = np.empty(shape = (s8t2_adata.shape[0], new_extracter.num_features), 
# 
from time import process_time
import logging

logging.basicConfig(filename="log_image_extraction_tester.log", format='%(asctime)s - %(message)s', level=logging.DEBUG)

logging.debug(f"Started: {process_time()}\n")

big_im_container = sq.im.ImageContainer()
big_im_container.add_img(R".\original_data\High-resolution_tissue_images\V10F03-033\201210_BC_V10F03-033_S8C-T_RJ.D1-Spot000001.jpg")
# Seems to use the adata.uns and the spatial coords in adata to get image?
generator = big_im_container.generate_spot_crops(s8t2_adata, 
                                                 return_obs=True)


new_extracter = image_extracter(image_size= (377,377,3))
new_extracter.prep_model()

# Initialize image feature array as empty
# Renaming the columns here is necessary to avoid un readable adata
empty_array = np.full(fill_value= np.nan,
                        shape = (s8t2_adata.shape[0], new_extracter.num_features), 
                        dtype=np.float32)
# Set empty array as .obsm['im_features']
s8t2_adata.obsm['im_features'] = pd.DataFrame(empty_array, 
                                                index = s8t2_adata.obs_names,
                                                columns = [f"feature_{x}" for x in range(new_extracter.num_features)])

for i, (image_container, spot_index) in enumerate(generator):
    s8t2_adata.obsm['im_features'].loc[spot_index,:] = new_extracter.extract_one(
        image_container['image_0'], 
        as_np=True)
    if i % 10 == 0:
        # print(f"Spot number:{i} at time {process_time()}")
        logging.debug(f"Spot number {i}")


# Seems to fail if adata does not have correct column names in .obsm['im_features']
s8t2_adata.write_h5ad("./intermediate_data/with_image_features_33D_S8T2_2.h5ad")
s8t2_adata.obsm['im_features'].to_csv('./intermediate_data/im_features_s8t2.csv')

logging.debug(f"End: {process_time()}\n")

# s8t2_adata_with_imfeatures = ad.read_h5ad("./intermediate_data/with_image_features_33D_S8T2_2.h5ad")
# %%[markdown]
# Tensorflow training syntax method. See here [https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_tf.html] for inspiration.
# %%
# # put some of below into image extracter
# import tensorflow as tf
# big_im_container = sq.im.ImageContainer()
# big_im_container.add_img(R".\original_data\High-resolution_tissue_images\V10F03-033\201210_BC_V10F03-033_S8C-T_RJ.D1-Spot000001.jpg")
# # Seems to use the adata.uns and the spatial coords in adata to get image?
# second_generator = big_im_container.generate_spot_crops(s8t2_adata, 
#                                                  return_obs=True)
# dataset = tf.data.Dataset.from_tensor_slices ([x for x in second_generator])
# # # tf.data.Dataset.from_generator # seems to be not parallelizable according to those online
# # spot_names = tf.data.Dataset.from_tensor_slices(s8t2_adata.obs_names) # takes a while but may end up faster?


# # make tf dataset from slice
#   using generate_image_crops
# zip with labels (here just names?)
# # embedding = model.predict [still use image_extracter]
# # construct new anndata from embeddings
# # Then merge in old anndata columns with new?

# %%
# See here for advice: https://stackoverflow.com/questions/74056685/create-tensorflow-dataset-from-list-files
import tensorflow as tf
# filenames = tf.data.Dataset.list_files("intermediate_data/patched_data/V10F03-033_A/patches/*", shuffle=False)
filenames = tf.data.Dataset.list_files("intermediate_data/patched_data/V10F03-033_D/patches/*", shuffle=False)

def read_image (path):
    image_tensor =  tf.image.decode_image(tf.io.read_file(path))
    return tf.expand_dims(image_tensor, axis=0)
    

def read_spot_name (path):
    pathname = path.numpy().decode('ascii')
    filename = pathname.split("\\")[-1]
    split_file_name = filename.split("_")
    barcode = split_file_name[2]
    slide_id = split_file_name[0][-2:] + split_file_name[1]

    return barcode + '_' + slide_id

image_set = filenames.map(read_image)
names = [read_spot_name(x) for x in filenames]
# %%
new_extracter = image_extracter(image_size=(380, 380, 3))
new_extracter.prep_model()

image_results = new_extracter.model.predict(image_set, use_multiprocessing=True)
# %% [markdown]
# This runs! Very quickly
# This does not run the same as using the spot cropped images, but does run consistent
# with self. It seems like they are pretty different, event though all that this added
# is 3 pixels in each direction
# Minor issues
#   Names still not identical between when patches generated and way of referring to spots
#   Image features may not be the same across each? Or are spot names just wrong
# TODO: if possible integrate the predict call into image_extracter

image_feature_df = pd.DataFrame(
    data = image_results,
    index = names,
    columns = [f"feature_{i}" for i in range(image_results.shape[1])]
)

with_image_features = ad.read_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")
print(with_image_features.to_df().filter(regex = "AAACAAGTATCTCCCA", axis=0))
print(image_feature_df.filter(regex = "AAACAAGTATCTCCCA", axis=0))

# %% Pipeline for extraction
# This works

# %% Map spots

# %% 
prev_imfeatures = ad.read_h5ad('intermediate_data/with_image_features_33D_S8T2_2.h5ad')
# %% Image feature extraction in batch
image_feature_df.to_csv("intermediate_data/bigbatch_im_features.csv")

image_feature_df_1 = pd.read_csv("intermediate_data/bigbatch_im_features.csv", index_col=0)
image_feature_df_1 = image_feature_df_1.astype(dtype='float32')
 
image_feature_df == image_feature_df_1
# %% [markdown]
# It seems like it worked!

# %% Fix the file contents
with_image_features = ad.read_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")
# with_enrichments = ad.read_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")

just_image_features = ad.AnnData(
    X = with_image_features.obsm['im_features'],
    obs=with_image_features.obs,
    obsm=with_image_features.obsm,
    uns = with_image_features.uns
)
del just_image_features.obsm['im_features']
just_image_features.write_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")

with_enrichments = ad.read_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")
with_enrichments.layers['nonnormalized'] = with_enrichments.X
with_enrichments.X = with_enrichments.layers['nes']
del with_enrichments.layers['nes']
with_enrichments.write_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")
# %% [markdown]
# ## Combining with our gene set data
# Let's get everything in one adata
# %% 
with_enrichments = ad.read_h5ad("intermediate_data/s8t2_all_at_once_enrichments.h5ad")
just_image_features =  ad.read_h5ad("intermediate_data/with_image_features_33D_S8T2_2.h5ad")

# Check what would be removed with 'inner' join
assert with_enrichments.shape[0] == just_image_features.shape[0]

combined_enr_im_ad = ad.concat(
    [with_enrichments, just_image_features],
    axis = 1,
    join = 'inner', # better keep what they both have
    merge = 'same', # better keep what they both have!
    uns_merge = 'first' # keep enrichments uns
)

combined_enr_im_ad.write("intermediate_data/s8t2_combined_enr_im.h5ad")




# %% [markdown]
# ## Big Run
# Run through each set of presliced images and make image feature adatas.
import os
import logging

logging.basicConfig(filename="log_image_extraction_tester.log", format='%(asctime)s - %(message)s', level=logging.DEBUG)

logging.debug("Starting batch extraction")
# Converter from slide to biopsy name
config = pd.read_csv('classify/main_config.csv')

config['simple_biopsy_sample_id'] = config['biopsy_sample_id'].map(
    lambda x: x[:-4] + x[-3] + x[-1]
)
slide_to_biopsy_converter = {}
for _, (sample_id, biopsy_sample_id) in config[['sample_id', 'simple_biopsy_sample_id']].iterrows():
    slide_to_biopsy_converter[sample_id] = biopsy_sample_id

# Load in model
new_extracter = image_extracter(image_size=(380, 380, 3))
new_extracter.prep_model()

# 
INPUTS_DIR = "intermediate_data/patched_data/"
OUTPUTS_DIR = "intermediate_data/batch_extracted_image_adatas"
logging.debug("Starting extraction loop")
for folder in os.listdir(INPUTS_DIR):
    logging.debug(f"Starting {folder}")
    # print(folder)
    path = os.path.join(INPUTS_DIR, folder, "patches")
    simple_biopsy_id = slide_to_biopsy_converter[folder]
    new_extracter.image_set_from_path(path, 
                                      in_place = True)
    adata_result = new_extracter.extract_from_imageset_adata(
        name_append = f"_{simple_biopsy_id}" 
    )
    adata_result.write_h5ad(
        os.path.join(OUTPUTS_DIR, f"{simple_biopsy_id}_im.h5ad"))
    logging.debug(f"{folder} written")

logging.debug("Batch extractions done")

# TODO: integrate uns, obsm, and other obs values to allow for plotting and comparison
# %%
