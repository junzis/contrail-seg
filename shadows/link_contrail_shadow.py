# %%
import cv2
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler


# %%
def hough_transform(img, threshold=2000):
    height, width = img.shape

    thetas = torch.arange(0, 180, 0.1)
    d = np.sqrt(np.square(height) + np.square(width))
    rhos = torch.arange(-d, d, 3)

    cos_thetas = torch.cos(torch.deg2rad(thetas))
    sin_thetas = torch.sin(torch.deg2rad(thetas))

    points = torch.argwhere(img > 0.5).type_as(cos_thetas)
    rho_values = torch.matmul(points, torch.stack((sin_thetas, cos_thetas)))

    accumulator, (theta_vals, rho_vals) = torch.histogramdd(
        torch.stack(
            (
                torch.tile(thetas, (rho_values.shape[0],)),
                rho_values.ravel(),
            )
        ).T,
        bins=[thetas, rhos],
    )

    accumulator = torch.transpose(accumulator, 0, 1)
    hough_lines = torch.argwhere(accumulator > threshold)
    rho_idxs, theta_idxs = hough_lines[:, 0], hough_lines[:, 1]
    hough_rhos, hough_thetas = rhos[rho_idxs], thetas[theta_idxs]

    return hough_thetas, hough_rhos


def cluster_hough_lines(hough_rhos, hough_thetas, min_cluster_size=10):
    hough_rhos, hough_thetas = np.array(hough_rhos), np.array(hough_thetas)

    # Stack rhos and thetas into a single array
    hough_data = np.column_stack((hough_rhos, hough_thetas))

    # Normalize the data
    scaler = StandardScaler()
    hough_data_normalized = scaler.fit_transform(hough_data)

    # Apply HDBSCAN clustering
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clustering.fit(hough_data_normalized)
    labels = clustering.labels_

    # Initialize lists to store the centroids
    centroid_rhos = []
    centroid_thetas = []

    # Iterate through unique labels to find the centroid of each cluster
    for label in np.unique(labels):
        if label == -1:
            # Skip the noise points, which have a label of -1
            continue
        idxs = np.where(labels == label)
        centroid_rho = np.mean(hough_rhos[idxs])
        centroid_theta = np.mean(hough_thetas[idxs])
        centroid_rhos.append(centroid_rho)
        centroid_thetas.append(centroid_theta)

    return np.array(centroid_rhos), np.array(centroid_thetas)


def viz(image, hough_thetas, hough_rhos, centroid_rhos, centroid_thetas):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.imshow(image)

    colors = plt.cm.Set1(np.linspace(0, 1, len(centroid_thetas)))

    for idx, (theta, rho) in enumerate(zip(centroid_thetas, centroid_rhos)):
        color = colors[idx]

        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        ax2.plot((x1, x2), (y1, y2), color=color, lw=2)
        ax3.scatter(theta, rho, s=30, c=[color], zorder=100)

    ax2.set_xlim(0, image.shape[0])
    ax2.set_ylim(0, image.shape[1])

    Y, X = np.where(image > 0.5)
    ax2.scatter(X, Y, s=1, zorder=2, color="gray")

    ax2.invert_yaxis()

    ax3.scatter(hough_thetas, hough_rhos, s=1, label="Original Hough Lines", c="grey")

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_xlabel("$\\theta$", rotation=0)
    ax3.set_ylabel("$\\rho$")
    ax3.set_xlim([0, 180])

    plt.tight_layout()
    plt.show()


# %%

number = 60

f_contrail = glob.glob(f"data/landsat/lowres/1_{number}/label/*_contrail.png")[0]
f_shadow = glob.glob(f"data/landsat/lowres/1_{number}/label/*_shadow.png")[0]

img_contrail = np.amax(cv2.imread(f_contrail, cv2.IMREAD_UNCHANGED), axis=2)
img_shadow = np.amax(cv2.imread(f_shadow, cv2.IMREAD_UNCHANGED), axis=2)
img_contrail = img_contrail / img_contrail.max()
img_shadow = img_shadow / img_shadow.max()

# %% process contrails
contrail_thetas = []
contrail_rhos = []
hough_thetas, hough_rhos = hough_transform(
    torch.from_numpy(img_contrail), threshold=1500
)
centroid_rhos, centroid_thetas = cluster_hough_lines(hough_rhos, hough_thetas)
contrail_rhos.append(centroid_rhos)
contrail_thetas.append(centroid_thetas)

viz(img_contrail, hough_thetas, hough_rhos, centroid_rhos, centroid_thetas)


# %% process images
shadow_thetas = []
shadow_rhos = []
hough_thetas, hough_rhos = hough_transform(torch.from_numpy(img_shadow), threshold=1500)
centroid_rhos, centroid_thetas = cluster_hough_lines(hough_rhos, hough_thetas)
shadow_rhos.append(centroid_rhos)
shadow_thetas.append(centroid_thetas)

viz(img_shadow, hough_thetas, hough_rhos, centroid_rhos, centroid_thetas)


# %%
pairs = (
    pd.merge(
        pd.DataFrame(
            np.array([shadow_thetas, shadow_rhos]).squeeze().T,
            columns=["shadow_theta", "shadow_rho"],
        ).assign(shadow_id=lambda d: d.index + 1),
        pd.DataFrame(
            np.array([contrail_thetas, contrail_rhos]).squeeze().T,
            columns=["contrail_theta", "contrail_rho"],
        ).assign(contrail_id=lambda d: d.index + 1),
        how="cross",
    )
    .assign(d_rho=lambda d: abs(d.shadow_rho - d.contrail_rho))
    .sort_values(["shadow_id", "d_rho"])
    .drop_duplicates("shadow_id")
    .sort_values(["contrail_id", "d_rho"])
    .drop_duplicates("contrail_id")
    .assign(d_theta=lambda d: abs(d.shadow_theta - d.contrail_theta))
)
pairs

# %%
for i, r in pairs.iterrows():
    lines = plt.plot([r.shadow_theta, r.contrail_theta], [r.shadow_rho, r.contrail_rho])
    color = lines[0].get_color()
    plt.scatter(r.shadow_theta, r.shadow_rho, color=color)
    plt.scatter(
        r.contrail_theta,
        r.contrail_rho,
        color=color,
        marker="o",
        facecolor="w",
        zorder=100,
    )
    plt.text(r.shadow_theta - 1, r.shadow_rho, int(r.shadow_id), ha="left")
    plt.text(r.contrail_theta + 1, r.contrail_rho, int(r.contrail_id), ha="right")

# %%

f_mtl = open(sorted(glob.glob(f"data/landsat/highres/1_{number}/*_MTL.txt"))[0])
file_lines = f_mtl.readlines()

sun_azimuth = float(file_lines[73].split("= ")[1].split("\n")[0])
print("sun_azimuth", sun_azimuth)
sun_elevation = float(file_lines[74].split("= ")[1].split("\n")[0])
print("sun_elevation", sun_elevation)
origin_width = float(file_lines[90].split("= ")[1].split("\n")[0])
print("origin_width", origin_width)


# %%

grid_cell_size = 30
lowres_width = 4096

pairs = pairs.assign(
    distance_m=lambda x: origin_width
    / lowres_width
    * 30
    * x.d_rho
    / np.sin(np.radians(sun_azimuth - x.contrail_theta))
).assign(altitude_m=lambda x: x.distance_m * np.tan(np.radians(sun_elevation)))

pairs
