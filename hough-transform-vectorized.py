import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def line_detection_vectorized(image, num_rhos=180, num_thetas=180, t_count=200):
    edge_height, edge_width = image.shape[:2]

    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    #
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    #
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    #
    accumulator = np.zeros((len(rhos), len(rhos)))
    #
    figure = plt.figure(figsize=(12, 12))

    subplot4 = figure.add_subplot(1, 1, 1)
    # subplot4.imshow(image)
    #
    edge_points = np.argwhere(image != 0)
    # edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
    #
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    #
    accumulator, theta_vals, rho_vals = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]), rho_values.ravel(), bins=[thetas, rhos]
    )
    accumulator = np.transpose(accumulator)
    lines = np.argwhere(accumulator > t_count)
    rho_idxs, theta_idxs = lines[:, 0], lines[:, 1]
    r, t = rhos[rho_idxs], thetas[theta_idxs]

    for line in lines:
        y, x = line
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        # x0 = (a * rho) + edge_width_half
        # y0 = (b * rho) + edge_height_half
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot4.plot([x1, x2], [y1, y2], lw=1, color="k", alpha=0.5)

    subplot4.set_xlim(0, image.shape[0])
    subplot4.set_ylim(0, image.shape[1])
    subplot4.title.set_text("Detected Lines")
    plt.show()
    return accumulator, rhos, thetas


if __name__ == "__main__":
    image_path = f"data/test_florida/mask/florida_2020_03_17_0921.png"
    image = np.amax(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), axis=2)
    image = image / image.max()
    line_detection_vectorized(image)
