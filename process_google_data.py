# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = "data/google-goes-contrail"

# %%


def load_image(record_id):
    t11_bound = (243, 303)
    cloud_top_tdiff_bound = (-4, 5)
    tdiff_bound = (-4, 2)

    band11 = np.load(f"{BASE_DIR}/{record_id}/band_11.npy")
    band13 = np.load(f"{BASE_DIR}/{record_id}/band_13.npy")
    band14 = np.load(f"{BASE_DIR}/{record_id}/band_14.npy")
    band15 = np.load(f"{BASE_DIR}/{record_id}/band_15.npy")
    pixel_mask = np.load(f"{BASE_DIR}/{record_id}/human_pixel_masks.npy")
    individual_mask = np.load(f"{BASE_DIR}/{record_id}/human_individual_masks.npy")

    def normalize(data, bounds):
        return (data - bounds[0]) / (bounds[1] - bounds[0])

    r = normalize(band15 - band14, tdiff_bound)
    g = normalize(band14 - band11, cloud_top_tdiff_bound)
    b = normalize(band14, t11_bound)
    ash = np.clip(np.stack([r, g, b], axis=2), 0, 1)

    btd = band13 - band15

    return dict(
        ash=ash[..., 4],
        btd=btd[..., 4],
        pixel_mask=pixel_mask,
        individual_mask=individual_mask,
    )


# %%
train_dirs = sorted(glob.glob(f"{BASE_DIR}/*/*"))
records = [
    "/".join(s.split("/")[-2:])
    for s in train_dirs
    if ("train" in s or "validation" in s)
]

# %%

f_mask_stats = f"{BASE_DIR}/mask_stats.csv"

if os.path.exists(f_mask_stats):
    mask_stats = pd.read_csv(f_mask_stats)
else:
    buffer = []

    for record in tqdm(records):
        pixel_mask = np.load(f"{BASE_DIR}/{record}/human_pixel_masks.npy")
        buffer.append(
            dict(
                record_id=record,
                mask_pixels=pixel_mask.sum(),
                tag=record.split("/")[0],
            )
        )

    mask_stats = pd.DataFrame.from_dict(buffer)

    mask_stats = mask_stats.assign(contrail=lambda d: d.mask_pixels > 0)

    mask_stats.to_csv(f_mask_stats, index=False)

mask_stats

# %%

for record in mask_stats.query("500<mask_pixels<800").head(20).record_id:
    img_dict = load_image(record)

    print(record)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ash = img_dict["ash"]
    btd = img_dict["btd"]
    mask = img_dict["pixel_mask"]
    ax1.imshow(ash)
    ax1.imshow(mask, alpha=0.3)
    ax2.imshow(btd, cmap="Greys_r")
    # ax2.imshow(mask, alpha=0.3)
    plt.show()

# %%
