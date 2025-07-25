import os
from joblib import Parallel, delayed
from itertools import combinations
import open3d as o3d
import pandas as pd
import numpy as np
import shutil
import laspy
import copy
import time
import os
import json

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


###FUNCTIONS###

def get_file_path(plot_num, path=r'\\stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'): ##adding a default
    check = False  ## a check inside of a fucntion? without a parameter? 
    for filename in os.listdir(path):    #okay now listing a path variable which is also not a parameter
        if filename.startswith(plot_num) and not filename.endswith('alt'):
            folder = filename
            check = True
    if not check:
        print(plot_num, 'FILE NOT FOUND')
    return folder   ###this function need a test


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def get_xy_convex_polygon(pcd):
    """Return a shapely Polygon representing the 2D convex hull (XY) of a point cloud."""
    pcd= pcd.voxel_down_sample(voxel_size=0.5)  # Optional: downsample for performance
    points = np.asarray(pcd.points)[:, :2]  # XY projection
    if len(points) < 3:
        return None  # Not enough points for a polygon
    hull = ConvexHull(points)
    polygon = Polygon(points[hull.vertices])
    return polygon

def get_overlap_xy_polygon(pcd1, pcd2):
    poly1 = get_xy_convex_polygon(pcd1)
    poly2 = get_xy_convex_polygon(pcd2)
    if poly1 is None or poly2 is None:
        return None
    overlap = poly1.intersection(poly2)
    if overlap.is_empty:
        return None
    return overlap

def polygon_to_selection_volume(polygon: Polygon, z_min=100, z_max=200):
    """
    Converts a 2D shapely Polygon to an Open3D SelectionPolygonVolume.
    
    Args:
        polygon (shapely.geometry.Polygon): The overlap polygon in XY.
        z_min (float): Minimum Z value for the cropping prism.
        z_max (float): Maximum Z value for the cropping prism.

    Returns:
        o3d.visualization.SelectionPolygonVolume: The cropping volume.
    """
    if polygon.is_empty or not polygon.is_valid:
        return None

    # Convert 2D polygon to Nx3 array with Z=0 (or any constant)
    xy_coords = np.array(polygon.exterior.coords)
    xyz_coords = np.column_stack((xy_coords, np.zeros(len(xy_coords))))  # Z=0

    # Convert to Vector3dVector
    bounding_polygon = o3d.utility.Vector3dVector(xyz_coords.astype("float64"))

    # Create and configure SelectionPolygonVolume
    volume = o3d.visualization.SelectionPolygonVolume()
    volume.orthogonal_axis = "Z"
    volume.bounding_polygon = bounding_polygon
    volume.axis_min = z_min
    volume.axis_max = z_max

    return volume

def lazO3d_crop_transform(file_path, json_data, plot):
    """
    Reads a .laz file, crops using bounding box, and applies transformation from JSON dict. Order of operations matters.

    Args:
        file_path (str): Path to the .laz file.
        json_data (dict): Dictionary loaded from transformations.json.
        plot (str): Plot key used to access the transformation and bounding box.

    Returns:
        open3d.geometry.PointCloud: Cropped and transformed point cloud.
    """

    # Read .laz using laspy
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Get bounds and transformation from the JSON dict
    info = json_data[plot]
    min_bound = np.array(info["min_bound"])
    max_bound = np.array(info["max_bound"])
    transformation = np.array(info["transformation"])
    # Crop and transform
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_cloud = pcd.crop(bbox)
    cropped_cloud.transform(transformation)

    return cropped_cloud

def evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_overlap, target_overlap, threshold, init_matrix)
    return eval_result.fitness, eval_result


def calculate_overlap(reference, target):
    bbr = reference.get_axis_aligned_bounding_box()
    bbt = target.get_axis_aligned_bounding_box()

    bbox_min = [np.maximum(bbr.min_bound[0], bbt.min_bound[0]), np.maximum(bbr.min_bound[1], bbt.min_bound[1]),
                np.maximum(bbr.min_bound[2], bbt.min_bound[2])]
    bbox_max = [np.minimum(bbr.max_bound[0], bbt.max_bound[0]), np.minimum(bbr.max_bound[1], bbt.max_bound[1]),
                np.minimum(bbr.max_bound[2], bbt.max_bound[2])]

    crop_box = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)

    return crop_box

###LOADS MAIN DICTIONARY###
with open(os.path.join(r"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/global_transformations.json")) as f:
    global_transformations = json.load(f)
path = r'\\stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'

## we will start by defining the plots we want to process
## we will process 0500 to 0609
# I dont know why Isaac wants to do left and bottom  insted of right and top but sure. 
## i mean it breaks if i start with 0500 as my reference plot

plots = (
    "0500", "0501", "0502", "0503", "0504", "0505", "0506", "0507", "0508", "0509",
    "0600", "0601", "0602", "0603", "0604", "0605", "0606", "0607", "0608", "0609"
)

plots_05 = [p for p in plots if p.startswith("05")]

plot1= lazO3d_crop_transform(os.path.join(path,get_file_path(plots_05[0], path),"results.laz"), global_transformations, plots_05[0])  ##we are reading the cloud for the first plot
plot2= lazO3d_crop_transform(os.path.join(path,get_file_path(plots_05[1], path),"results.laz"), global_transformations, plots_05[1])  ##we are reading the cloud for the second plot


overlap= get_overlap_xy_polygon(plot1, plot2)  ##we are calculating the overlap between the first two plots
vol= polygon_to_selection_volume(overlap, z_min=100, z_max=200)  ##we are converting the overlap to a selection volume

plot1_overlap= vol.crop_point_cloud(plot1)  ##we are cropping the first plot with the overlap
plot2_overlap= vol.crop_point_cloud(plot2)  ##we are cropping the second plot with the overlap

##visualization of the first two plots
draw_registration_result(plot1_overlap, plot2_overlap, np.eye(4))  ##we are visualizing the first two plots

threshold = 0.1  ##this is the threshold for the registration evaluation
init_matrix = np.eye(4)  ##this is the initial matrix for the registration evaluation

fitness, eval_result = evaluate_registration(plot1_overlap, plot2_overlap, threshold, init_matrix)  ##we are evaluating the registration between the first two plots
print(f"Fitness: {fitness}")  ##we are printing the fitness of the registration
##fitness is crazy high when i do the convex hull, but it is not the same as the one in the original code

