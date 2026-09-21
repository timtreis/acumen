import numpy as np
import squidpy as sq

img = sq.datasets.visium_fluo_image_crop()
sq.im.process(img=img, layer="image", method="smooth")

counts = {}
for ch in [0, 1, 2]:
    sq.im.segment(
        img=img, layer="image_smooth", method="watershed", channel=ch,
        layer_added=f"seg_{ch}",
    )
    arr = img[f"seg_{ch}"].values
    u = np.unique(arr)
    n = len(u) - 1 if 0 in u else len(u)
    counts[ch] = n

print("segment counts per channel:", counts)
ranked = sorted(counts.items(), key=lambda kv: kv[1])
print("fewest segments channel (test answer):", ranked[0][0])
print("most segments channel (train answer):", ranked[-1][0])
