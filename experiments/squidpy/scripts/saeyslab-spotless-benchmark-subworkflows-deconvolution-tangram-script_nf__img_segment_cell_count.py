import warnings

import squidpy as sq

warnings.filterwarnings("ignore")


def count_cells(adata):
    img = sq.im.ImageContainer.from_adata(adata)
    layer = list(img)[0]
    sq.im.process(img=img, layer=layer, method="smooth")
    sq.im.segment(img=img, layer=f"{layer}_smooth", method="watershed", channel=0)
    seg = img["segmented_watershed"].values
    return int(seg.max())


# train: cropped H&E Visium mouse brain dataset
train_adata = sq.datasets.visium_hne_adata_crop()
train_answer = count_cells(train_adata)
print("train (visium_hne_adata_crop) number of segmented cells:", train_answer)

# test: cropped fluorescence Visium mouse brain dataset
test_adata = sq.datasets.visium_fluo_adata_crop()
test_answer = count_cells(test_adata)
print("test (visium_fluo_adata_crop) number of segmented cells:", test_answer)
