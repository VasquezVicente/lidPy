import os
import laspy
import numpy as np
from typing import List, Tuple
from pathlib import Path
from tools import cloud_check
#the first functionality has to be reading a las or laz, right not this is possible through laspy
#assume new user just got here and wants to read a point cloud, he or she is trying to read a TLS/MLS
# ALS point cloud and get its attributes. 

dir=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\plot1\tiles"
os.makedirs(dir, exist_ok=True)
path= r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\panama_BCI_plot1 0.010 m_clip.laz"


def generate_bounds_grid(xmin: float, ymin: float, xmax: float, ymax: float, 
                         tile_size: float, buffer_size: float) -> List[Tuple[float, float, float, float]]:
    sub_bounds = []

    num_tiles_x = int((xmax - xmin) / tile_size)
    num_tiles_y = int((ymax - ymin) / tile_size)

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_min = xmin + i * tile_size - buffer_size
            y_min = ymin + j * tile_size - buffer_size
            x_max = xmin + (i + 1) * tile_size + buffer_size
            y_max = ymin + (j + 1) * tile_size + buffer_size

            sub_bounds.append((np.float64(x_min), np.float64(y_min), np.float64(x_max), np.float64(y_max)))

    return sub_bounds
grid=generate_bounds_grid(0,0,100,100,20,0)

def cloud_retile(input_path: str,
                 output_folder: str,
                 sub_bounds: List[Tuple[float, float, float, float]],
                 chunk_size= 10**6):
    with laspy.open(input_path) as file:
        writers = [None] * len(sub_bounds)
        try:
            count = 0
            for points in file.chunk_iterator(chunk_size):
                print(f"{count / file.header.point_count * 100:.2f}%")
                x, y = points.x.copy(), points.y.copy()
                point_piped = 0
                for i, (x_min, y_min, x_max, y_max) in enumerate(sub_bounds):
                    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

                    if np.any(mask):
                        if writers[i] is None:
                            output_filename = f"retile_{int(x_min)}_{int(y_min)}.laz"
                            output_path = Path(output_folder) / output_filename
                            writers[i] = laspy.open(output_path, mode="w", header=file.header)

                        sub_points = points[mask]
                        writers[i].write_points(sub_points)

                    point_piped += np.sum(mask)
                    if point_piped == len(points):
                        break

                count += len(points)

            print(f"{count / file.header.point_count * 100:.2f}%")
        finally:
            for writer in writers:
                if writer is not None:
                    writer.close()

cloud_retile(path, dir, grid,10**6)

# I think to read a catalog we need spatial indexing of every tile. every tile has buffer so there is repeated points
# with same position, so when handling a catalog it must be filtering repeated points out of the whole dataset
# the catalog function of the lidR package in r must be the most powerful tool.


## probably one of the most challenging utilities of this package will be dealing with MLS. Lets assume every MLS
# scan has reference points and we will require the user to provide reference points for every MLS scan it inputs
# we already have a function to retile any cloud, now we need one to open a cloud


