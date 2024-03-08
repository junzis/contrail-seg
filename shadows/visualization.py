# %%
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from pyopensky.trino import Trino
from traffic.core import Traffic, Flight
from traffic.data.adsb.opensky import format_history


# %%
def load_metadata(filename):
    metadata = {}
    with open(filename) as MTL:
        for line in MTL:
            if "=" not in line:
                continue

            name, var = line.split("=")
            var = var.strip().replace('"', "")

            try:
                var = float(var)
            except:
                pass

            metadata[name.strip()] = var
    return metadata


# %%
number = 7

directory = f"data/landsat/highres/1_{number}"
metafile = glob.glob(f"{directory}/*MTL*")[0]
metadata = load_metadata(metafile)


# %%
# Define date and time
utm_zone = metadata["UTM_ZONE"]
date_str = metadata["DATE_ACQUIRED"]
time_str = metadata["SCENE_CENTER_TIME"].split(".")[0]
time_image = date_str + "T" + time_str
time_image = datetime.strptime(time_image, "%Y-%m-%dT%H:%M:%S")
print(time_image)

# Define bounds
west = min(metadata["CORNER_UL_LON_PRODUCT"], metadata["CORNER_LL_LON_PRODUCT"])
east = max(metadata["CORNER_UR_LON_PRODUCT"], metadata["CORNER_LR_LON_PRODUCT"])
north = max(metadata["CORNER_UL_LAT_PRODUCT"], metadata["CORNER_UR_LAT_PRODUCT"])
south = min(metadata["CORNER_LL_LAT_PRODUCT"], metadata["CORNER_LR_LAT_PRODUCT"])
bounds_box = [west, south, east, north]
print("bounding box", bounds_box)


def visualize(directory, data):
    import cartopy.crs as ccrs
    from osgeo import gdal

    # generate RGB image
    ds = gdal.Open(glob.glob(f"{directory}/*_RGB.TIF")[0])
    band1 = ds.GetRasterBand(1).ReadAsArray()
    band2 = ds.GetRasterBand(2).ReadAsArray()
    band3 = ds.GetRasterBand(3).ReadAsArray()
    rgb_array = np.dstack([band1, band2, band3])

    # get projection from band 10
    ds = gdal.Open(glob.glob(f"{directory}/*_B10.TIF")[0])
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()

    # %%
    zone = int(utm_zone)
    projection = ccrs.UTM(zone, southern_hemisphere=False)

    # create an Axes object with this projection
    subplot_kw = dict(projection=projection)
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=subplot_kw)

    # set the extents of the Axes object to match the image
    # extent is x,y origin to x,y + pixel size in (x,y) direction * image size in (x,y) direction
    extent = (
        gt[0],
        gt[0] + ds.RasterXSize * gt[1],
        gt[3] + ds.RasterYSize * gt[5],
        gt[3],
    )

    # ax.set_extent(extent, projection)

    # show the image (specifying the image origin as being the top of the image),
    #   and using a blue color map
    img = ax.imshow(rgb_array, extent=extent, origin="upper")

    for i in range(len(unique_ic)):
        ax.scatter(
            data.get_group(unique_ic[i])["lon"].values,
            data.get_group(unique_ic[i])["lat"].values,
            s=1,
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            data.get_group(unique_ic[i])["lon"].values[-1],
            data.get_group(unique_ic[i])["lat"].values[-1],
            data.get_group(unique_ic[i])["icao24"].values[-1],
            transform=ccrs.PlateCarree(),
            color="b",
        )

    # draw coastlines, and gridlines  (grid line labels not supported for hand crafted UTM projection)
    ax.coastlines(resolution="10m", zorder=4, alpha=0.4)
    ax.gridlines()
    ax.axis("on")
    plt.show()
