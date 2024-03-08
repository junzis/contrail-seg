# %%
import os
import glob
from PIL import Image
from scipy.ndimage import zoom
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

WIDTH = 4096

#%%

def resize_img(img_pil):
    aspect_ratio = img_pil.height / img_pil.width

    # Convert the image to a NumPy array
    img_np = np.array(img_pil)

    # Calculate scaling factors
    scale_x = WIDTH / img_np.shape[1]
    scale_y = aspect_ratio * scale_x

    # Use scipy's zoom function to resize
    if len(img_np.shape) == 2:
        # Grayscale or single channel image
        img_resize = zoom(img_np, (scale_y, scale_x))
    else:
        # RGB or RGBA image
        img_resize = zoom(img_np, (scale_y, scale_x, 1))

    # normalize image
    img_norm = (
        (img_resize - img_resize.min())
        / (img_resize.max() - img_resize.min())
        * 255
    )

    # Convert the NumPy array back to a PIL Image
    img_output = Image.fromarray(img_norm.astype("uint8")).convert("L")

    return img_output


def process_b7(f_b7):
    f_qa = f_b7.replace("_B7", "_QA_PIXEL")

    img_qa_pil = Image.open(f_qa)
    img_b7_pil = Image.open(f_b7)

    # Convert PIL images to NumPy arrays
    img_qa = np.array(img_qa_pil)
    img_b7 = np.array(img_b7_pil)

    # Vectorized function to get mask
    def get_mask_vectorized(img_qa_array, bit_index):
        return ((img_qa_array >> bit_index) & 1) ^ 1

    # Create a water mask
    bit_index_for_water = 7
    water_mask = get_mask_vectorized(img_qa, bit_index_for_water)

    # Identify water pixels
    water_pixels = np.argwhere(water_mask == 0)

    # Identify non-water pixels and their intensities
    non_water_pixels = np.argwhere(water_mask == 1)
    non_water_intensities = img_b7[non_water_pixels[:, 0], non_water_pixels[:, 1]]

    # Compute 85th percentile of the brightest non-water pixels
    p85_value = np.percentile(non_water_intensities, 85)

    # Replace water pixels with the 85th percentile value
    img_b7[water_pixels[:, 0], water_pixels[:, 1]] = p85_value

    # Convert the modified NumPy array back to a PIL image
    img_b7_pil = Image.fromarray(img_b7.astype('uint16'))

    return img_b7_pil


#%%
for input_path in sorted(glob.glob(f"data/landsat/highres/*/*.TIF")):
    if any(map(input_path.__contains__, ["BTD", "RADSAT", "RGB"])):
        continue

    output_path = input_path.replace(".TIF", ".png").replace("highres", "lowres")

    # if Path(output_path).exists():
    #     continue

    # if os.path.exists(output_path) and os.path.getmtime(output_path) > 1694426195:
    #     continue

    # if "1_11/" not in input_path:
    #     continue

    if "_B7" not in input_path:
        continue

    print(output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if "_B7" in input_path:
        img = process_b7(input_path)
    else:
        img = Image.open(input_path)

    img_output = resize_img(img)
    img_output.save(output_path, format="PNG")


# %%
def normalize_and_scale(band_array):
    # Calculate percentiles for scaling
    cmin = np.percentile(band_array, 5)
    cmax = np.percentile(band_array, 99.5)

    # Clip and scale the array based on the calculated percentiles
    band_array = np.clip(band_array, cmin, cmax)
    return ((band_array - cmin) / (cmax - cmin) * 255).astype("uint8")


def create_rgb_image(b2_path, b3_path, b4_path, output_path):
    # Read the individual bands as grayscale images
    b2 = Image.open(b2_path)
    b3 = Image.open(b3_path)
    b4 = Image.open(b4_path)

    # Convert PIL images to NumPy arrays, normalize and scale
    b2_array = normalize_and_scale(np.array(b2))
    b3_array = normalize_and_scale(np.array(b3))
    b4_array = normalize_and_scale(np.array(b4))

    # Stack the bands to make an RGB image
    rgb_array = np.dstack((b4_array, b3_array, b2_array))

    # Convert back to PIL image in RGB mode
    rgb_image = Image.fromarray(rgb_array, "RGB")

    # Save the RGB image
    rgb_image.save(output_path)


# Sample code to get image paths and call the function
image_paths = sorted(glob.glob(f"data/landsat/lowres/1_{img_number}/*.png"))

b2_path = image_paths[3]
b3_path = image_paths[4]
b4_path = image_paths[5]
output_path = b2_path.replace("B2", "RGB")

create_rgb_image(b2_path, b3_path, b4_path, output_path)

# %%

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Read images with PIL
f_qa = "data/landsat/highres/1_7/LC08_L1TP_023039_20230119_20230131_02_T1_QA_PIXEL.TIF"
f_b7 = "data/landsat/highres/1_7/LC08_L1TP_023039_20230119_20230131_02_T1_B7.TIF"

