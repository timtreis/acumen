# Ground-truth script for task "visium_channel_mean_intensity".
# Reproduces, for a bundled Visium image dataset, the mean pixel intensity
# of a given channel across the whole tissue image (the same "summary"
# image-feature statistic squidpy's calculate_image_features/features_summary
# computes in the mined analysis, applied here to the full image container).
import squidpy as sq

# --- train variant: visium_fluo, channel index 0 ---
img = sq.datasets.visium_fluo_image_crop()
feats = img.features_summary(layer="image", channels=[0])
train_answer = round(feats["summary_ch-0_mean"], 2)
print("train (visium_fluo, channel 0) mean intensity:", train_answer)

# --- test variant: visium_hne, channel index 0 ---
img2 = sq.datasets.visium_hne_image_crop()
feats2 = img2.features_summary(layer="image", channels=[0])
test_answer = round(feats2["summary_ch-0_mean"], 2)
print("test (visium_hne, channel 0) mean intensity:", test_answer)
