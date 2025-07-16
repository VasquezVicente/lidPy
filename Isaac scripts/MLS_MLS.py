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

#Link to Vicente's folder: https://drive.google.com/drive/folders/1vqcBVej_-oHQ5_PJoLsepj3iTGa3lDMa

columns = {}
plot_transformations = {}
pcds = {}
adj_pcds = {}

save_path = os.path.join(r'D:\MLS_alignment\temporary')
os.makedirs(save_path, exist_ok=True)

elev_raw = np.loadtxt(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\aux_files\elev.TXT", skiprows=1) #change the path to the server location
elev = {}
min_elev = 1000   ##1000 asl? 
max_elev = 1
for plot in elev_raw:
    elev[str(int(plot[0])) + str(int(plot[1]))] = plot[2]   # this is trying to create a a dict with the q20 and their elevations but i think it has a an error 
    if min_elev > plot[2]:
        min_elev = plot[2]
    if max_elev < plot[2] + 10:
        max_elev = plot[2] + 10

def calc_plot_target(plot, elev):
    """
    Computes the 4 corner coordinates (x, y, z) of a plot using the elevation dict.
    Args:
        plot (str): Plot name, e.g. "0500"
        elev (dict): Elevation dictionary with keys like '100020' and values as z.

    Returns:
        np.ndarray: 4x3 array of target positions.
    """
    col = int(plot[:2])
    row = int(plot[2:])
    return np.array([
        [col * 20, row * 20, elev[str(col*20) + str(row*20)]],
        [col * 20, (row + 1) * 20, elev[str(col*20) + str((row+1)*20)]],
        [(col + 1) * 20, row * 20, elev[str((col+1)*20) + str(row*20)]],
        [(col + 1) * 20, (row + 1) * 20, elev[str((col+1)*20) + str((row+1)*20)]]
    ])

column_target_positions = {}

def calc_col_target(column):   # more annotations about what stuff do, i am guessing this determines the column/strip 
    row = 20
    col = int(column)
    column_target_positions[column] = np.array([
        [col * 20, row * 20, elev[str(col*20) + str(row*20)]],   ## so this is the bottom left corned 
        [col * 20, (row + 5) * 20, elev[str(col*20) + str((row+5)*20)]],   ## this is the top left corner assuming from the +5
        [(col + 1) * 20, row * 20, elev[str((col+1)*20) + str(row*20)]],  ## this looks like the the bottom left corner maybe?
        [(col + 1) * 20, (row + 5) * 20, elev[str((col+1)*20) + str((row+5)*20)]]  ## aah top right 
    ])


def get_file_path(plot_num, path=r'\\stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'): ##adding a default
    check = False  ## a check inside of a fucntion? without a parameter? 
    for filename in os.listdir(path):    #okay now listing a path variable which is also not a parameter
        if filename.startswith(plot_num) and not filename.endswith('alt'):
            folder = filename
            check = True
    if not check:
        print(plot_num, 'FILE NOT FOUND')
    return folder   ###this function need a test


def lazO3d(file_path): #Reads laz files  ## the header is missing from this function, its a flaw that will need changing eventually 
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def mem_eff_loading(file_path):  #again what doest this function does? 
    cloud = laspy.read(file_path)
    points = np.empty((len(cloud.x), 3), dtype=np.float32)   ## seeems to be creating an empty np cloud using the dimensions of the original
    points[:, 0] = cloud.x
    points[:, 1] = cloud.y
    points[:, 2] = cloud.z
    return points

def sort_ref(ref):
    dist_zero = [np.linalg.norm(ref[0]), np.linalg.norm(ref[1]), np.linalg.norm(ref[2]), np.linalg.norm(ref[3])]

def get_ref(ref, path):
    if len(ref) == 4:  ## so ref is already a loaded txt, bad
        return ref
    elif len(ref) < 4: # Open the trajectory, get bounds, and return the bounds.
        traj = o3d.io.read_point_cloud(path)  ## but the traj is a path and not a loaded one, consistency
        b_box = traj.get_minimal_oriented_bounding_box()
        E = np.asarray(b_box.extent.copy())
        C = np.asarray(b_box.center.copy())
        R = np.asarray(b_box.R.copy())
        ref = np.array([
            [E[0] / -2, E[1] / -2, E[2] / -2],
            [E[0] / -2, E[1] / 2, E[2] / -2],
            [E[0] / 2, E[1] / -2, E[2] / -2],
            [E[0] / 2, E[1] / 2, E[2] / -2]
        ])
        ref = np.dot(ref, R)
        ref += C
        return ref

    elif len(ref) > 4:   ## so 5 points, we generally drop the last oen which is repeated
        best_combo = 'x'
        best_score = 10000
        for combo in combinations(ref, 4):
            dists = [
                np.linalg.norm(combo[0] - combo[1]), np.linalg.norm(combo[0] - combo[2]),   ###what?
                np.linalg.norm(combo[0] - combo[3]), np.linalg.norm(combo[1] - combo[2]),
                np.linalg.norm(combo[1] - combo[3]), np.linalg.norm(combo[2] - combo[3])
            ]
            dists.sort()
            score = np.linalg.norm(dists[0] - 10) + np.linalg.norm(dists[1] - 10) + np.linalg.norm(dists[2] - 10) + np.linalg.norm(dists[3] - 10) + np.linalg.norm(dists[4] - 20) + np.linalg.norm(dists[5] - 20)
            if score < best_score:

                best_combo, best_score = combo, score

        ref_xy = ref[:, :2]
        dist_matrix = np.zeros((len(ref_xy), len(ref_xy)))
        for row, _ in enumerate(dist_matrix):
            for col, _ in enumerate(dist_matrix[row]):
                dist_matrix[row][col] = np.linalg.norm(ref_xy[row] - ref_xy[col])

        diagonals = []
        for row, _ in enumerate(dist_matrix[1:]):
            for col, _ in enumerate(dist_matrix[row]):
                if dist_matrix[row][col] - 20 < min(diagonals) and not dist_matrix[row][col] in diagonals:
                    diagonals[diagonals.index(min(diagonals))] = dist_matrix[row][col]


def angle_between(p1, p2, p3): #what? more annotations
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def is_square_rotated(quad, target_side=20, tol=2.0, angle_tol=10.0):
    """
    Checks if a 4-point shape is a square, regardless of orientation.
    """
    from scipy.spatial import distance_matrix # a module loaded inside a function

    dists = distance_matrix(quad, quad)
    idx = np.argsort(np.sum(dists, axis=1))  # reorder by closeness to centroid
    quad = quad[idx]

    # Check pairwise distances
    side_lengths = [
        np.linalg.norm(quad[0] - quad[1]),
        np.linalg.norm(quad[1] - quad[2]),
        np.linalg.norm(quad[2] - quad[3]),
        np.linalg.norm(quad[3] - quad[0])
    ]
    diagonals = [
        np.linalg.norm(quad[0] - quad[2]),
        np.linalg.norm(quad[1] - quad[3])
    ]

    side_ok = np.all(np.abs(np.array(side_lengths) - target_side) < tol)
    diag_ok = np.abs(diagonals[0] - diagonals[1]) < tol and np.all(np.abs(np.array(diagonals) - target_side * np.sqrt(2)) < tol * 1.5)

    # Check right angles
    angles = [
        angle_between(quad[3], quad[0], quad[1]),
        angle_between(quad[0], quad[1], quad[2]),
        angle_between(quad[1], quad[2], quad[3]),
        angle_between(quad[2], quad[3], quad[0])
    ]
    angles_ok = np.all(np.abs(np.array(angles) - 90) < angle_tol)

    error = (np.std(side_lengths) + np.std(diagonals) + np.std(angles))

    return side_ok and diag_ok and angles_ok, error

def find_best_rotated_square(points, target_side=20, tol=2.0, angle_tol=10.0):
    best_error = float('inf')
    best_quad = None

    for combo in combinations(points, 4):
        quad = np.array(combo)
        is_sq, error = is_square_rotated(quad, target_side, tol, angle_tol)
        if is_sq and error < best_error:
            best_error = error
            best_quad = quad

    return best_quad


def get_b_box(traj, min_bound, z_extent): #Step 1: Applies initial crop
    b_box = traj.get_minimal_oriented_bounding_box()
    extent = b_box.extent.copy()
    extent[0] += 5
    extent[1] += 5
    extent[2] = z_extent
    center = b_box.center.copy()
    center[2] = min_bound + extent[2] * 0.5
    b_box = o3d.geometry.OrientedBoundingBox(center, b_box.R, extent)
    return b_box

timer = [[],[],[],[],[],[],[]]  ##whattt? you can use packages to get timers
def shorten(points, resolution, height):  #so this is a ground classification and normalization algorithm. we can work on it and is novel in python
    t1 = time.time()
    # Create a grid (e.g., 1m resolution)
    xy_indices = np.floor(points[:, :2] / resolution).astype(int) # i understand the concept of florr but we need a classfier option which should be more robust
    t2 = time.time()

    # Step 2: Estimate local ground using pandas groupby (fast!)
    df = pd.DataFrame({'gx': xy_indices[:, 0], 'gy': xy_indices[:, 1], 'z': points[:, 2]}) # clever but we need a more robust approach
    df = df.groupby(['gx', 'gy']).agg(['min', 'count'])['z'].reset_index()
    t3 = time.time()

    # Step 3: Apply density threshold
    threshold = (float(len(points)) / (625 / (resolution * resolution))) / 40
    df = df[df['count'] > threshold]
    t4 = time.time()

    # Build fast lookup for ground z values
    ground_dict = {(row['gx'], row['gy']): row['min'] for _, row in df.iterrows()}
    del df
    t5 = time.time()

    surrounding_points = [ #this will eventually break
        (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
        (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2)
    ]

    # Removes outliers
    updated_dict = {}
    med = np.median(list(ground_dict.values()))
    for key in ground_dict:
        adj_vals = []
        for point in surrounding_points:
            if (key[0] + point[0], key[1] + point[1]) in ground_dict:
                adj_vals.append(ground_dict[key[0] + point[0], key[1] + point[1]])
        adj_med = np.median(adj_vals)
        if ground_dict[key] > med + 2.0:
            continue
        if not adj_vals:
            updated_dict[key] = ground_dict[key]
        elif adj_med + resolution * 0.5 > ground_dict[key] > adj_med - resolution * 0.5:
            updated_dict[key] = ground_dict[key]
        else:
            updated_dict[key] = adj_med
    t6 = time.time()

    # Step 4: Lookup ground height for each point
    ground_z = np.array([updated_dict.get(tuple(k), np.nan) for k in xy_indices])
    del xy_indices
    t7 = time.time()

    # Step 5: Keep points within height of ground
    keep_mask = ~np.isnan(ground_z) & (points[:, 2] - ground_z < height)
    t8 = time.time()

    timer[0].append(t2 - t1)
    timer[1].append(t3 - t2)
    timer[2].append(t4 - t3)
    timer[3].append(t5 - t4)
    timer[4].append(t6 - t5)
    timer[5].append(t7 - t6)
    timer[6].append(t8 - t7)

    # Return filtered points
    return points[keep_mask]

# Compute transformation using Procrustes analysis (rigid alignment)
def compute_transformation(source, target):
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute optimal rotation using SVD
    U, _, Vt = np.linalg.svd(target_centered.T @ source_centered)
    R = U @ Vt  # Rotation matrix

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute translation
    t = target_mean - R @ source_mean

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return t, R, transformation


def add_transformation(plot_num, transformation):
    plot_transformations[plot_num] = transformation @ plot_transformations[plot_num]  #dot product might not need to be a function


global_transformations = {}
target_positions = {}
plots = ("0500", "0501","0502", "0503", "0504","0505", "0506", "0507", "0508", "0509","0600", "0601", "0602", "0603", "0604","0605", "0606", "0607", "0608", "0609")
path=r'\\stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'

for plot in plots:
    target_positions[plot] = calc_plot_target(plot, elev)

for plot in plots:
    # Get file paths
    folder = get_file_path(plot)
    ref_path = os.path.join(path, folder, 'results_trajref.txt')
    traj_path = os.path.join(path, folder, 'results_traj_time.ply')
    pcd_path = os.path.join(path, folder, 'results.laz')  # not used now, but might be later

    # Load and reorder reference points
    ref_unordered = np.loadtxt(ref_path, skiprows=1)[:, :3][-4:]
    ref = np.array([ref_unordered[3], ref_unordered[0], ref_unordered[1], ref_unordered[2]])

    # Compute transformation
    _, _, transformation = compute_transformation(ref, target_positions[plot])

    # Get bounding box from trajectory point cloud
    traj = o3d.io.read_point_cloud(traj_path)
    min_bound = traj.get_min_bound()
    max_bound = traj.get_max_bound()

    # Optional buffer — remove if not needed
    min_bound[:2] -= 5
    max_bound[:2] += 5
    min_bound[2] = -20  ###hard coded for now 
    max_bound[2] = 100  #hard coded for now 

    # Store data in dictionary
    global_transformations[plot] = {
        "transformation": transformation.tolist(),
        "min_bound": min_bound.tolist(),
        "max_bound": max_bound.tolist()
    }


with open(os.path.join(r"//stri-sm01/ForestLandscapes/UAVSHARE/BCNM Lidar Raw Data/global_transformations.json"), "w") as f:
    json.dump(global_transformations, f, indent=2)


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


with open(os.path.join(save_path, "global_transformations.json")) as f:
    global_transformations = json.load(f)

plot_0500= lazO3d_crop_transform(file_path = os.path.join(path, get_file_path("0500"), 'results.laz') ,  #this is a test
                                  json_data=global_transformations,
                                  plot="0500")

plot_0501= lazO3d_crop_transform(file_path = os.path.join(path, get_file_path("0501"), 'results.laz') ,  ##this is a test
                                  json_data=global_transformations,
                                  plot="0501")

o3d.visualization.draw_geometries([plot_0500,plot_0501])    #sanity check 


def evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_overlap, target_overlap, threshold, init_matrix)
    return eval_result.fitness, eval_result


def calculate_overlap(reference, target):
    bbr = reference.get_axis_aligned_bounding_box()
    bbt = target.get_axis_aligned_bounding_box()

    bbox_min = [np.maximum(bbr.min_bound[0], bbt.min_bound[0]), np.maximum(bbr.min_bound[1], bbt.min_bound[1]), np.maximum(bbr.min_bound[2], bbt.min_bound[2])]
    bbox_max = [np.minimum(bbr.max_bound[0], bbt.max_bound[0]), np.minimum(bbr.max_bound[1], bbt.max_bound[1]), np.minimum(bbr.max_bound[2], bbt.max_bound[2])]

    crop_box = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)

    return reference.crop(crop_box), target.crop(crop_box)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp(reference_cloud, target_cloud, threshold, transformation, radius, max_nn, k):
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    loss = o3d.pipelines.registration.TukeyLoss(k=k)
    p2p_loss = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    # initial evaluation
    initial_eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_cloud, target_cloud, threshold, transformation)
    initial_fitness_match = initial_eval_result.fitness
    final_fitness_match = initial_fitness_match + 0.001
    output = transformation
    while final_fitness_match < 1 and final_fitness_match > initial_fitness_match:
        initial_fitness_match = final_fitness_match
        # icp iteration
        reg_result = o3d.pipelines.registration.registration_icp(
            reference_cloud, target_cloud, threshold, transformation, p2p_loss)
        final_fitness_match = reg_result.fitness
        #draw_registration_result(reference_cloud, target_cloud, reg_result.transformation)
        if final_fitness_match <= 1 and final_fitness_match > initial_fitness_match:
            print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
            output = reg_result.transformation @ output
            reference_cloud.transform(reg_result.transformation)
        else:
            print('fitness did not improve. final fitness is', initial_fitness_match)
    return output


def extract(plot):
    if not plot in broken_plots:
        broken_plots.append(plot)

# ISSUES WITH THIS FUNCTION:
# 1. MOST IMPORTANT: fails to recognize when point cloud already exists, always loads next point cloud
# 2. Loaded point cloud is not shortened (I think this is fixed)
# 3. Loaded point cloud is way out of sync from the point cloud we are checking
# I suspect issues 1 and 3 have to do with the way I am finding the adjacent plots (str(int(plot) +/- 100))
def checker(plot, threshold, transformation, radius, max_nn, k):
    if not plot in checked_plots and not plot in broken_plots:
        left_plot, right_plot = str(int(plot) - 100), str(int(plot) + 100)
        if left_plot in broken_plots or not left_plot in pcds:
            print('No valid comparison plot found, creating new plot.')
            folder = get_file_path(right_plot)
            ref_path = os.path.join(path, folder, 'results_trajref.txt')
            pcd_path = os.path.join(path, folder, 'results.laz')
            calc_plot_target(right_plot)

            if os.path.isfile(pcd_path) and os.path.isfile(ref_path):
                pcd = lazO3d(pcd_path)
                ref_unordered = np.loadtxt(ref_path, skiprows=1)[:, :3][-4:]
                ref = np.array([ref_unordered[3], ref_unordered[0], ref_unordered[1], ref_unordered[2]])

                _, _, transformation = compute_transformation(ref, target_positions[right_plot])
                pcd.transform(transformation)

                box_min = [target_positions[right_plot][0][0] - 2.5,
                           target_positions[right_plot][0][1] - 2.5,
                           pcd.get_min_bound()[2]]
                box_max = [target_positions[right_plot][3][0] + 2.5,
                           target_positions[right_plot][3][1] + 2.5,
                           pcd.get_max_bound()[2]]
                b_box = o3d.geometry.AxisAlignedBoundingBox(box_min, box_max)
                pcd = pcd.crop(b_box)

                points = shorten(np.asarray(pcd.points), 0.5, height)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                reference_cloud, target_cloud = calculate_overlap(adj_pcds[plot], pcd)
            else:
                print(f'{str(int(plot) + 100)} not found')
                broken_plots.append(plot)
                return
        else:
            print('Loading plots')
            reference_cloud, target_cloud = calculate_overlap(adj_pcds[plot], pcds[left_plot])

        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        loss = o3d.pipelines.registration.TukeyLoss(k=k)
        p2p_loss = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        # initial evaluation
        initial_eval_result = o3d.pipelines.registration.evaluate_registration(
            reference_cloud, target_cloud, threshold, transformation)
        initial_fitness_match = initial_eval_result.fitness
        final_fitness_match = initial_fitness_match + 0.001

        while 0.9 > final_fitness_match > initial_fitness_match:
            initial_fitness_match = final_fitness_match
            # icp iteration
            reg_result = o3d.pipelines.registration.registration_icp(
                reference_cloud, target_cloud, threshold, transformation, p2p_loss)
            final_fitness_match = reg_result.fitness
            #draw_registration_result(reference_cloud, target_cloud, reg_result.transformation)
            if final_fitness_match >= 0.9:
                print(f'{plot} is not broken')
                print(initial_fitness_match, final_fitness_match)
                checked_plots.append(plot)
                break
            elif final_fitness_match <= initial_fitness_match:
                print(f'{plot} is broken')
                print(initial_fitness_match, final_fitness_match)
                broken_plots.append(plot)
                break
            else:
                print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
                reference_cloud.transform(reg_result.transformation)


def compiler(column):
    joined_pcd = o3d.geometry.PointCloud()
    joined_adj_pcd = o3d.geometry.PointCloud()
    for plot in columns[column]:
        if plot in checked_plots and str(int(plot) + 100) in checked_plots:
            joined_pcd += pcds[plot]
            joined_adj_pcd += adj_pcds[str(int(plot) + 100)]

    return joined_pcd, joined_adj_pcd

plot = '3020'
path = r'\\stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'
identity = np.eye(4)
fitness_scores = {}
column_fitness_scores = {}
accumulated_column_tf = identity
broken_plots = []
checked_plots = []
r = 0.1
max_nn = 80
k =  0.05
height = 2
threshold = 0.1
stats = {}

next_plot = ''
starting_plots = []
while next_plot != 'run':
    next_plot = input(f'Plots entered: {starting_plots}\nTo run, type "run". Otherwise, enter the next plot number: ')
    starting_plots.append(next_plot)

for stp in starting_plots:
    # Creating a dictionary of all the columns I will be aligning
    stc = int(stp[:2])
    stl = int(stp[2:])
    columns = {
        str(stc): [stp, str(stc) + str(stl + 1), str(stc) + str(stl + 2), str(stc) + str(stl + 3), str(stc) + str(stl + 4)],
        str(stc + 1): [str(stc + 1) + str(stl), str(stc + 1) + str(stl + 1), str(stc + 1) + str(stl + 2), str(stc + 1) + str(stl + 3), str(stc + 1) + str(stl + 4)],
        str(stc + 2): [str(stc + 2) + str(stl), str(stc + 2) + str(stl + 1), str(stc + 2) + str(stl + 2), str(stc + 2) + str(stl + 3), str(stc + 2) + str(stl + 4)],
        str(stc + 3): [str(stc + 3) + str(stl), str(stc + 3) + str(stl + 1), str(stc + 3) + str(stl + 2), str(stc + 3) + str(stl + 3), str(stc + 3) + str(stl + 4)],
        str(stc + 4): [str(stc + 4) + str(stl), str(stc + 4) + str(stl + 1), str(stc + 4) + str(stl + 2), str(stc + 4) + str(stl + 3), str(stc + 4) + str(stl + 4)],
    }

    plots = [
        stp, str(stc) + str(stl + 1), str(stc) + str(stl + 2), str(stc) + str(stl + 3), str(stc) + str(stl + 4),
        str(stc + 1) + str(stl), str(stc + 1) + str(stl + 1), str(stc + 1) + str(stl + 2), str(stc + 1) + str(stl + 3), str(stc + 1) + str(stl + 4),
        str(stc + 2) + str(stl), str(stc + 2) + str(stl + 1), str(stc + 2) + str(stl + 2), str(stc + 2) + str(stl + 3), str(stc + 2) + str(stl + 4),
        str(stc + 3) + str(stl), str(stc + 3) + str(stl + 1), str(stc + 3) + str(stl + 2), str(stc + 3) + str(stl + 3), str(stc + 3) + str(stl + 4),
        str(stc + 4) + str(stl), str(stc + 4) + str(stl + 1), str(stc + 4) + str(stl + 2), str(stc + 4) + str(stl + 3), str(stc + 4) + str(stl + 4)
    ]

    for plot in plots:
        plot_transformations[plot] = identity

    t1 = time.time()
    pre_process = Parallel(n_jobs=-1)(delayed(create_aligned_pcd)(plot) for plot in plots)

    for item in pre_process:
        key = list(item.keys())[0]
        if not item[key]:
            print(key, 'failed')
            broken_plots.append(key)
            columns[key[:2]].remove(key)
    t2 = time.time()
    print(f'Time to pre-process: \n===> {t2 - t1}')

    for column in columns:
        adj_column = str(int(column) + 1)
        if len(pcds) == 0:
            calc_col_target(column)
            accumulated_transformation = identity
            try:
                t1 = time.time()
                pcds = {}

                for plot in columns[column]:
                    pcds[plot] = o3d.io.read_point_cloud(os.path.join(save_path, f'{plot}TEMP.ply'))

                t2 = time.time()
                print(f'time to load column is\n===> {t2 - t1}')

            except Exception as e:
                print(f'An error ocurred loading {column}')
                print(f'Error type: {type(e).__name__}')
                print(f'Error message: {e}')

            # Loop through plots in the column to align them
            for plot in columns[column]:
                t1 = time.time()
                adj_plot = str(int(plot) + 1)
                prev_plot = str(int(plot) - 1)

                # Checks if there is another plot to be aligned
                if adj_plot in columns[column]:
                    try:
                        # Adds transformations from previous plots' alignments
                        add_transformation(adj_plot, accumulated_transformation)
                        pcds[adj_plot].transform(accumulated_transformation)

                        # Calculate overlap and run icp
                        adj_pcd_overlap, pcd_overlap = calculate_overlap(pcds[adj_plot], pcds[plot])
                        precise_transformation = icp(adj_pcd_overlap, pcd_overlap, threshold, identity, r, max_nn, k)

                        # Calculate and store fitness values
                        temp_pcd = copy.deepcopy(pcds[adj_plot])
                        temp_pcd.transform(precise_transformation)
                        adj_pcd_overlap, pcd_overlap = calculate_overlap(temp_pcd, pcds[plot])
                        fitness_scores[(plot, adj_plot)] = o3d.pipelines.registration.evaluate_registration(
                            adj_pcd_overlap, pcd_overlap, threshold, identity)

                        # Checking if the previous plot was broken, restarting the chain if it was.
                        if fitness_scores[(plot, adj_plot)].fitness < 0.9:
                            if (prev_plot, plot) in fitness_scores:
                                print('Checking adjacent pcd')
                                checker(adj_plot, threshold, identity, r, max_nn, k)
                            else:
                                print('Checking current and adjacent pcd')
                                checker(plot, threshold, identity, r, max_nn, k)
                                checker(adj_plot, threshold, identity, r, max_nn, k)
                        else:
                            print('Alignment successful')
                            if (prev_plot, plot) in fitness_scores:
                                checked_plots.append(adj_plot)
                            else:
                                checked_plots.append(plot)
                                checked_plots.append(adj_plot)

                        if plot in broken_plots or adj_plot in broken_plots:
                            continue

                        accumulated_transformation = precise_transformation @ accumulated_transformation
                        add_transformation(adj_plot, precise_transformation)
                        pcds[adj_plot].transform(precise_transformation)

                        del pcd_overlap, adj_pcd_overlap, temp_pcd

                        t2 = time.time()
                        print('time for alignment is\n===>', t2-t1)
                    except Exception as e:
                        print(f"An error occurred aligning {plot} and {adj_plot}")
                        print(f'Error type: {type(e).__name__}')
                        print(f'Error message: {e}')

            print('Column pcd done')

        if adj_column in columns:
            calc_col_target(adj_column)
            accumulated_transformation = identity
            try:
                adj_pcds = {}
                t1 = time.time()

                for plot in columns[adj_column]:
                    adj_pcds[plot] = o3d.io.read_point_cloud(os.path.join(save_path, f'{plot}TEMP.ply'))

                t2 = time.time()
                print(f'time to load column is\n===> {t2 - t1}')

            except Exception as e:
                print(f'An error occurred loading {adj_column}')
                print(f'Error type: {type(e).__name__}')
                print(f'Error message: {e}')

            for plot in columns[adj_column]:
                t1 = time.time()
                adj_plot = str(int(plot) + 1)
                prev_plot = str(int(plot) - 1)

                if adj_plot in columns[adj_column]:
                    # Create aligned reference point cloud
                    try:
                        add_transformation(adj_plot, accumulated_transformation)
                        adj_pcds[adj_plot].transform(accumulated_transformation)

                        # Calculate overlap and run icp
                        adj_pcd_overlap, pcd_overlap = calculate_overlap(adj_pcds[adj_plot], adj_pcds[plot])
                        precise_transformation = icp(adj_pcd_overlap, pcd_overlap, threshold, identity, r, max_nn, k)

                        # Calculate and store fitness values
                        temp_pcd = copy.deepcopy(adj_pcds[adj_plot])
                        temp_pcd.transform(precise_transformation)
                        adj_pcd_overlap, pcd_overlap = calculate_overlap(temp_pcd, adj_pcds[plot])
                        fitness_scores[(plot, adj_plot)] = o3d.pipelines.registration.evaluate_registration(
                            adj_pcd_overlap, pcd_overlap, threshold, identity)

                        if fitness_scores[(plot, adj_plot)].fitness < 0.9:
                            if (prev_plot, plot) in fitness_scores:
                                print('Checking adjacent pcd')
                                checker(adj_plot, threshold, identity, r, max_nn, k)
                            else:
                                print('Checking current and adjacent pcd')
                                checker(plot, threshold, identity, r, max_nn, k)
                                checker(adj_plot, threshold, identity, r, max_nn, k)
                        else:
                            print('Alignment successful')
                            if (prev_plot, plot) in fitness_scores:
                                checked_plots.append(adj_plot)
                            else:
                                checked_plots.append(plot)
                                checked_plots.append(adj_plot)

                        if plot in broken_plots or adj_plot in broken_plots:
                            continue

                        accumulated_transformation = precise_transformation @ accumulated_transformation
                        add_transformation(adj_plot, precise_transformation)
                        adj_pcds[adj_plot].transform(precise_transformation)

                        del pcd_overlap, adj_pcd_overlap, temp_pcd

                        t2 = time.time()
                        print('time for alignment is\n===>', t2 - t1)
                    except Exception as e:
                        print(f"An error occurred aligning {plot} and {adj_plot}")
                        print(f'Error type: {type(e).__name__}')
                        print(f'Error message: {e}')
            print('adjacent column pcd done')
            try:
                t1 = time.time()
                # Creating point clouds from dictionaries, re-assigning dictionaries
                column_pcd, adj_column_pcd = compiler(column)
                pcds = adj_pcds
                adj_pcds = {}

                # Adding initial transformation
                adj_column_pcd.transform(accumulated_column_tf)

                # Running ICP
                adj_pcd_overlap, pcd_overlap = calculate_overlap(adj_column_pcd, column_pcd)
                precise_transformation = icp(adj_pcd_overlap, pcd_overlap, threshold, identity, r, max_nn, k)

                # Calculating fitness
                adj_column_pcd.transform(precise_transformation)
                adj_pcd_overlap, pcd_overlap = calculate_overlap(adj_column_pcd, column_pcd)
                column_fitness_scores[column + '->' + adj_column] = o3d.pipelines.registration.evaluate_registration(
                    adj_pcd_overlap, column_pcd, threshold, identity
                )

                # Storing transformations, updating dictionaries
                accumulated_column_tf = precise_transformation @ accumulated_column_tf
                for plot in columns[adj_column]:
                    add_transformation(plot, accumulated_column_tf)
                for plot in pcds:
                    pcds[plot].transform(accumulated_column_tf)

                # Clearing memory
                del column_pcd, adj_column_pcd, pcd_overlap, adj_pcd_overlap
                t2 = time.time()
                print('time for alignment is\n===>', t2 - t1)
            except Exception as e:
                print(f"An error occurred aligning {column} and {adj_column}")
                print(f'Error type: {type(e).__name__}')
                print(f'Error message: {e}')
                pcds = {}
        else:
            for key in fitness_scores:
                print(key[0] + '->' + key[1] + ':', fitness_scores[key].fitness)
            for key in column_fitness_scores:
                print(key + ':', column_fitness_scores[key].fitness)
            pcds, adj_pcds = {}, {}


#    shutil.rmtree(r'D:\MLS_alignment\temporary')


    complete_pcd = o3d.geometry.PointCloud()
    for column in columns:
        for plot in columns[column]:
            # Reading the point cloud
            folder = get_file_path(plot)
            cloud = laspy.read(os.path.join(path, folder, 'results.laz'))
            points = np.vstack((cloud.x, cloud.y, cloud.z))

            #Decomposing the transformation
            T = plot_transformations[plot]
            translate = T[:3, 3]
            rotate = T[:3, :3]

            #Applying the transformation
            transformed_points = (rotate @ points).T + translate

            new_hdr = laspy.LasHeader(version="1.4", point_format=6)
            new_cloud = laspy.LasData(new_hdr)
            new_cloud.x = transformed_points[:, 0]
            new_cloud.y = transformed_points[:, 1]
            new_cloud.z = transformed_points[:, 2]

            save_path = os.path.join(r'D:\MLS_alignment', plot)
            if not plot in broken_plots:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                new_cloud.write(os.path.join(save_path, 'processed.laz'))
            else:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                new_cloud.write(os.path.join(save_path, 'BAD_processed.laz'))

    print(timer)
    means = [np.mean(timer[0]), np.mean(timer[1]), np.mean(timer[2]), np.mean(timer[3]), np.mean(timer[4]), np.mean(timer[5]), np.mean(timer[6]), np.mean(timer[7])]
    for m in means:
        print(m)