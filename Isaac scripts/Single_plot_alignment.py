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

#################################### Defining variables ############################################

path = r'\\Stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'
identity = np.eye(4)
fitness_scores = {}

# plots and columns will not be modified, they will be iterated through in for loops.
columns = {}
plots = []

# I used a set of input statements so that the user can define what plots they want to align
# In later stages this will not be necessary
r = int(input('enter number of rows: '))
r1 = int(input('enter starting row: '))
c = int(input('enter number of columns: '))
c1 = int(input('enter starting column: '))

for col in range(c):
    row_list = []
    col = str(col + c1)
    if len(col) == 1:
        col = '0' + col
    for row in range(r):
        row = str(row + r1)
        if len(row) == 1:
            row = '0' + row
        plot = col + row
        row_list.append(plot)
        plots.append(plot)
    columns[col] = row_list

# This variable will be used to store transformations from both the global and precise alignment.
plot_transformations = {}
for plot in plots:
    plot_transformations[plot] = identity
acc_tfs = {}

# These dictionaries will hold the point clouds I am aligning, so that they are easily accessible
pcds = {}
prev_pcds = {}

# plots will be added to these lists after alignment. If alignment was successful, they will be added to checked.
# If alignment was unsuccessful, they will be added to broken. If there were no plots to align them to, they will be added to hanging.
broken_plots = ['1900']
checked_plots = ['1901']
hanging_plots = []

#Parameters for icp
r = 0.1
max_nn = 80
k = 0.05
threshold = 0.1

# Creating temporary directory to store pre-processed files
temp_path = os.path.join(r'D:\MLS_alignment\temporary')
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

# Calculating elevation values
elev_raw = np.loadtxt(r"C:\Users\Wright-MullerI\Downloads\elev.TXT", skiprows=1)
elev = {}
min_elev = 1000
max_elev = 1
for plot in elev_raw:
    elev[str(int(plot[0])) + str(int(plot[1]))] = plot[2]
    if min_elev > plot[2]:
        min_elev = plot[2]
    if max_elev < plot[2] + 10:
        max_elev = plot[2] + 10

# Calculating the target positions for the plots
target_positions = {}
for plot in plots:
    col = int(plot[:2])
    row = int(plot[2:])
    target_positions[plot] = np.array([
        [col * 20, row * 20, elev[str(col * 20) + str(row * 20)]],
        [col * 20, (row + 1) * 20, elev[str(col * 20) + str((row + 1) * 20)]],
        [(col + 1) * 20, row * 20, elev[str((col + 1) * 20) + str(row * 20)]],
        [(col + 1) * 20, (row + 1) * 20, elev[str((col + 1) * 20) + str((row + 1) * 20)]]
    ])

############################################## DEFINING FUNCTIONS #######################################

# Getting neighboring plots. I created this function because I got errors when doing str(int(plot)), when "plot" began with a 0.
def get_plot(plot, side):
    if side == 'N':
        plot = str(int(plot) + 1)
    elif side == 'E':
        plot = str(int(plot) + 100)
    elif side == 'S':
        plot = str(int(plot) - 1)
    elif side == 'W':
        plot = str(int(plot) - 100)
    while len(plot) < 4: # Ensuring plot is the correct length
        plot = '0' + plot
    return plot


def get_file_path(plot_num):
    for filename in os.listdir(path):
        if filename.startswith(plot_num) and not filename.endswith('alt'):
            return filename
    print(plot_num, 'FILE NOT FOUND')
    return 'FILE NOT FOUND'


def lazO3d(file_path):  # Reads laz files
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def mem_eff_loading(file_path):
    cloud = laspy.read(file_path)
    points = np.empty((len(cloud.x), 3), dtype=np.float32)
    points[:, 0] = cloud.x
    points[:, 1] = cloud.y
    points[:, 2] = cloud.z
    return points

# These two functions, sort_ref and get_ref, ensure that the reference points are in the right order.
# I added them because I found a plot that had six reference positions, and I figured it could help.
def sort_ref(ref):
    index_list = [0, 1, 2, 3]
    dist_zero = [np.linalg.norm(ref[0][:2]), np.linalg.norm(ref[1][:2]), np.linalg.norm(ref[2][:2]),
                 np.linalg.norm(ref[3][:2])]
    ref_final = np.zeros((4, 3))
    ref_final[0] = ref[dist_zero.index(min(dist_zero))]
    ref_final[3] = ref[dist_zero.index(max(dist_zero))]
    index_list.remove(dist_zero.index(min(dist_zero)))
    index_list.remove(dist_zero.index(max(dist_zero)))
    if ref_final[3][1] > 0:
        if ref[index_list[0]][0] < ref[index_list[1]][0]:
            ref_final[1] = ref[index_list[0]]
            ref_final[2] = ref[index_list[1]]
        elif ref[index_list[0]][0] > ref[index_list[1]][0]:
            ref_final[1] = ref[index_list[1]]
            ref_final[2] = ref[index_list[0]]
    elif ref_final[3][1] < 0:
        if ref[index_list[0]][0] < ref[index_list[1]][0]:
            ref_final[1] = ref[index_list[1]]
            ref_final[2] = ref[index_list[0]]
        elif ref[index_list[0]][0] > ref[index_list[1]][0]:
            ref_final[1] = ref[index_list[0]]
            ref_final[2] = ref[index_list[1]]
    return ref_final


def get_ref(plot, path):
    ref = np.loadtxt(path, skiprows=1)[:, :3][:-1]
    if len(ref) == 4:
        return sort_ref(list(ref))
    elif len(ref) < 4:  # Open the trajectory, get bounds, and return the bounds.
        print(f'{plot} is missing a reference point')
        path = path[:-7] + '_time.ply'
        traj = o3d.io.read_point_cloud(path)
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
        return sort_ref(ref)

    elif len(ref) > 4:
        print(f'{plot} has an extra reference point')
        best_combo = np.empty(4)
        best_score = 10000
        for combo in combinations(ref, 4):
            dists = [
                np.linalg.norm(combo[0] - combo[1]), np.linalg.norm(combo[0] - combo[2]),
                np.linalg.norm(combo[0] - combo[3]), np.linalg.norm(combo[1] - combo[2]),
                np.linalg.norm(combo[1] - combo[3]), np.linalg.norm(combo[2] - combo[3])
            ]
            dists.sort()
            score = float(np.linalg.norm(dists[0] - 20)) + float(np.linalg.norm(dists[1] - 20)) + float(
                np.linalg.norm(dists[2] - 20)) + float(np.linalg.norm(dists[3] - 20)) + float(
                np.linalg.norm(dists[4] - (20 * np.sqrt(2)))) + float(np.linalg.norm(dists[5] - 20 * (20 * np.sqrt(2))))
            if score < best_score:
                best_combo, best_score = sort_ref(list(combo)), score
        return sort_ref(best_combo)

# This code shortens the point clouds. I have had some issues with this, in some files it removes nearly every point.
# I believe the issue is with the outlier removal part. The print statements are temporary, for troubleshooting.
def shorten(points, resolution, height):
    # Create a grid (e.g., 1m resolution)

    xy_indices = np.floor(points[:, :2] / resolution).astype(int)

    # Step 2: Estimate local ground using pandas groupby
    df = pd.DataFrame({'gx': xy_indices[:, 0], 'gy': xy_indices[:, 1], 'z': points[:, 2]})
    df = df.groupby(['gx', 'gy']).agg(['min', 'count'])['z'].reset_index()

    # Step 3: Apply density threshold
    threshold = (float(len(points)) / (625 / (resolution * resolution))) / 40
    df = df[df['count'] > threshold]

    # Build fast lookup for ground z values
    ground_dict = {(row['gx'], row['gy']): row['min'] for _, row in df.iterrows()}
    del df

    surrounding_points = [
        (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
        (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2)
    ]

    # Removes outliers
    updated_dict = {}
    med = float(np.median(list(ground_dict.values()))) + 2.0
    for key in ground_dict:
        if ground_dict[key] > med:
            continue

        adj_vals = []
        for point in surrounding_points:
            if (key[0] + point[0], key[1] + point[1]) in ground_dict:
                adj_vals.append(ground_dict[key[0] + point[0], key[1] + point[1]])
        if len(adj_vals) > 0:
            adj_med = np.median(adj_vals)
            if adj_med + resolution * 0.5 > ground_dict[key] > adj_med - resolution * 0.5:
                updated_dict[key] = ground_dict[key]
            else:
                updated_dict[key] = adj_med


    # Step 4: Lookup ground height for each point
    ground_z = np.array([updated_dict.get(tuple(k), np.nan) for k in xy_indices])
    del xy_indices

    # Step 5: Keep points within height of ground
    keep_mask = ~np.isnan(ground_z) & (points[:, 2] - ground_z < height)

    # Return filtered points
    return points[keep_mask]


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


def create_aligned_pcd(plot):
    # Getting pcd and ref paths. The conditionals are because the naming conventions on some files in the ForestLandscapes folder vary.
    try:
        folder = get_file_path(plot)
        if os.path.isfile(os.path.join(path, folder, 'results_trajref.txt')):
            ref_path = os.path.join(path, folder, 'results_trajref.txt')
        else:
            ref_path = ''
            print(f'{plot} REF FILE NOT FOUND')
        if os.path.isfile(os.path.join(path, folder, 'results.laz')):
            pcd_path = os.path.join(path, folder, 'results.laz')
        else:
            pcd_path = ''
            print(f'{plot} PCD FILE NOT FOUND')
        if os.path.isfile(os.path.join(path, folder, 'results_traj_time.ply')):
            traj_path = os.path.join(path, folder, 'results_traj_time.ply')
        else:
            traj_path = ''
            print(f'{plot} TRAJ FILE NOT FOUND')

        # Checking if file has been pre-processed
        if os.path.exists(os.path.join(temp_path, f'{plot}TEMP.ply')):
            ref = get_ref(plot, ref_path)
            _, _, transformation = compute_transformation(ref, target_positions[plot])
            plot_transformations[plot] = transformation
            return {plot: True}

        # Checking if pcd and ref exist
        if os.path.isfile(pcd_path) and os.path.isfile(ref_path) and os.path.isfile(traj_path):
            # Loading ref, making sure it is in the proper order
            ref = get_ref(plot, ref_path)

            # Computing and storing transformation
            t, R, transformation = compute_transformation(ref, target_positions[plot])
            plot_transformations[plot] = transformation

            # Loading points, applying transformation
            points = mem_eff_loading(pcd_path)
            points = np.dot(points, R.T)
            points += t

            # Loading traj, Calculating bounding box and cropping point cloud
            traj = o3d.io.read_point_cloud(traj_path)
            traj.transform(transformation)
            box_min = traj.get_min_bound()
            box_max = traj.get_max_bound()
            if 17.5 > box_max[0] - box_min[0] > 25 or 17.5 > box_max[1] - box_min[1] > 25: # Checks dimensions of bounding box
                print(f'{plot} trajectory is broken')
                box_min = [target_positions[plot][0][0] - 2.5, target_positions[plot][0][1] - 2.5, min_elev]
                box_max = [target_positions[plot][3][0] + 2.5, target_positions[plot][3][1] + 2.5, max_elev]
            else:
                box_min[0] -= 2.5
                box_min[1] -= 2.5
                box_max[0] += 2.5
                box_max[1] += 2.5
            mask = np.all((points >= box_min) & (points <= box_max), axis=1)
            points = points[mask]

            # Shortening point cloud
            points = shorten(points, 0.5, 2)

            # Store as pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(temp_path, f'{plot}TEMP.ply'), pcd)

            print(plot, 'done')

            return {plot: True}
        else:
            print(f'{plot} not found')
            return {plot: False}
    except Exception as e:
        print(f"An error occurred loading {plot}")
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {e}')
        return {plot: False}


# Checks which of the neighboring plots have been checked and can be aligned to.
def check_alignment(column, vert_plot, hori_plot):
    if vert_plot in checked_plots and hori_plot in checked_plots:
        return 'L'
    elif vert_plot in checked_plots:
        return 'V'
    elif hori_plot in checked_plots:
        return 'H'
    else:
        return 'B'


def evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_overlap, target_overlap, threshold, init_matrix)
    return eval_result.fitness, eval_result


def calculate_overlap(reference, target):
    bbr = reference.get_axis_aligned_bounding_box()
    bbt = target.get_axis_aligned_bounding_box()

    bbox_min = [np.maximum(bbr.min_bound[0], bbt.min_bound[0]), np.maximum(bbr.min_bound[1], bbt.min_bound[1]), min_elev]
    bbox_max = [np.minimum(bbr.max_bound[0], bbt.max_bound[0]), np.minimum(bbr.max_bound[1], bbt.max_bound[1]), max_elev]

    crop_box = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)

    return reference.crop(crop_box), target.crop(crop_box)

# Performs different crops based on the output of the check_alignment function
def new_calculate_overlap(plot, vert_plot, hori_plot, transformation, typ):
    # I use a temporary point cloud to avoid transforming the original multiple times
    temp = copy.deepcopy(pcds[plot])
    temp.transform(transformation)
    if typ == 'L':
        # I essentially perform calculate_overlap twice, once vertically, once horizontally, and add the results into a single reference and a single target pcd.
        reference = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        ref_temp, tar_temp = calculate_overlap(temp, pcds[plot])
        reference += ref_temp
        target += tar_temp
        ref_temp, tar_temp = calculate_overlap(temp, prev_pcds[vert_plot])
        reference += ref_temp
        target += tar_temp

        return (reference, target)

    elif typ == 'V':
        return calculate_overlap(temp, pcds[vert_plot])

    elif typ == 'H':
        return calculate_overlap(temp, prev_pcds[hori_plot])


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

    # initial evaluation (REMOVE FOR FINAL VERSION)
    initial_eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_cloud, target_cloud, threshold, transformation)
    initial_fitness_match = initial_eval_result.fitness
    final_fitness_match = initial_fitness_match + 0.002
    output = transformation
    while final_fitness_match < 1 and final_fitness_match > initial_fitness_match + 0.001: # ADD 0.001 to stop endless insignificant improvements
        initial_fitness_match = final_fitness_match
        # icp iteration
        reg_result = o3d.pipelines.registration.registration_icp(
            reference_cloud, target_cloud, threshold, transformation, p2p_loss)
        final_fitness_match = reg_result.fitness
#        draw_registration_result(reference_cloud, target_cloud, reg_result.transformation)
        if final_fitness_match <= 1 and final_fitness_match > initial_fitness_match:
            print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
            output = reg_result.transformation @ output
            reference_cloud.transform(reg_result.transformation)
        else:
            print('fitness did not improve. final fitness is', initial_fitness_match)
    return output

# This function is unnecessary, I haven't really gotten around to modifying it yet.
def writer(plot):
    try:
        # Reading the point cloud
        folder = get_file_path(plot)
        points = mem_eff_loading(os.path.join(path, folder, 'results.laz'))

        # Decomposing the transformation
        T = plot_transformations[plot]
        t = T[:3, 3]
        R = T[:3, :3]

        # Applying the transformation
        points = np.dot(points, R.T)
        points += t

        new_hdr = laspy.LasHeader(version="1.4", point_format=6)
        new_cloud = laspy.LasData(new_hdr)
        new_cloud.x = points[:, 0]
        new_cloud.y = points[:, 1]
        new_cloud.z = points[:, 2]
        del points

        save_path = os.path.join(r'D:\MLS_alignment', plot)
        if not plot in broken_plots:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            new_cloud.write(os.path.join(save_path, 'processed.laz'))
            print(f'{plot} written successfully')
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            new_cloud.write(os.path.join(save_path, 'BAD_processed.laz'))
            print(f'{plot} written successfully')
    except Exception as e:
        print(f'An error ocurred saving {plot}')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {e}')

############################################### PRE-PROCESSING #####################################################
### I insist proesecutes analysis is not heavy, time parallelization is not necessary. only slow because the time it takes to load and write temp clouds.
t1 = time.time()
# Parallelized pre-processing. I have found this to be significantly faster, although the code still has some issues with the shorten function
pre_process = Parallel(n_jobs=-1)(delayed(create_aligned_pcd)(plot) for plot in plots)

# Pre-processing returns a list of dictionaries in the format {str(plot_num): bool}
# This loops through that output and determines which point clouds were not able to be loaded.
for item in pre_process:
    key = list(item.keys())[0]
    if not item[key]:
        print(key, 'failed')
        broken_plots.append(key)
t2 = time.time()
print(f'Time to pre-process: \n===> {t2 - t1}')

################################################## MAIN LOOP ########################################################
columns = ['05','06']  #temporary definition of columns, this should be removed in the future.
for column in columns:   #columns is not defined
    # Resetting variables
    accumulated_transformation = identity
    prev_pcds = pcds
    pcds = {}

    # Loop through plots in the column to align them
    for plot in columns[column]:
        try:
            t1 = time.time()

            # First, checks if plot was loaded properly
            if plot in broken_plots:   #where is broken_plots defined?
                print(f'{plot} was given as broken')
                continue

            # Loads plot into pcds
            pcds[plot] = o3d.io.read_point_cloud(os.path.join(temp_path, f'{plot}TEMP.ply'))  #PCDS arent defined and its is unclear where do they come from

            # Checks if this is the first plot to be loaded. In the future this should be removed.
            if len(checked_plots) == 0:
                checked_plots.append(plot)
                acc_tfs[plot] = accumulated_transformation
                continue

            # Finds adjacent plot numbers
            prev_plot = get_plot(plot, 'S')
            left_plot = get_plot(plot, 'W')

            # Checks which of the adjacent plots can be used for alignment
            alignment_type = check_alignment(column, prev_plot, left_plot)

            if alignment_type == 'L':
                print('L crop')
            elif alignment_type == 'V':
                print('V crop')
            elif alignment_type == 'H': # If the crop is horizontal, accumulated_transformations is reset to the transformation of the plot that it will be aligned to
                print('H crop')
                accumulated_transformation = acc_tfs[left_plot]
            else: # If no valid point cloud is found, then the plot is marked as hanging.
                print(plot, 'is hanging')
                acc_tfs[plot] = accumulated_transformation
                hanging_plots.append(plot)
                continue

            # Transforms point cloud
            pcds[plot].transform(accumulated_transformation)  ###where is the accumulated_transformation coming from? 

            # Calculates overlap and stores point cloud
            overlap = new_calculate_overlap(plot, prev_plot, left_plot, identity, alignment_type)
            precise_transformation = icp(overlap[0], overlap[1], threshold, identity, r, max_nn, k)

            # Calculate and store fitness values
            overlap = new_calculate_overlap(plot, prev_plot, left_plot, precise_transformation, alignment_type)
            fitness_scores[plot] = o3d.pipelines.registration.evaluate_registration(overlap[0], overlap[1],
                                                                                    threshold, identity)
            print(fitness_scores[plot])

            # Checks fitness scores to verify if plot is broken or not
            if fitness_scores[plot].fitness < 0.9:
                broken_plots.append(plot)
                print(plot, 'is broken')
                t2 = time.time()
                print('time for alignment is\n===>', t2 - t1)
                continue

            # If plot is not broken, marks plot as checked
            checked_plots.append(plot)
            # Applies transformation to pcd so that the next plot can be aligned properly
            pcds[plot].transform(precise_transformation)
            # Accumulates transformation
            accumulated_transformation = precise_transformation @ accumulated_transformation
            # Saves local alignment for plot
            plot_transformations[plot] = accumulated_transformation @ plot_transformations[plot]
            # Stores accumulated transformation
            acc_tfs[plot] = accumulated_transformation

            print(plot, 'is good')
            # Clears memory
            del overlap
            if left_plot in prev_pcds:
                del prev_pcds[left_plot]

            t2 = time.time()
            print('time for alignment is\n===>', t2 - t1)
        except Exception as e:
            broken_plots.append(plot)
            print(f"An error occurred aligning {plot}")
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {e}')

    print(f'Column {column} done')

########################################## CHECK HANGING PLOTS #######################################

# The goal of this loop is to align all plots listed as hanging, only stopping if all plots are aligned
# or if all unaligned plots are entirely surrounded by broken plots.
isolated = False
while len(hanging_plots) > 0 and not isolated:
    # Reverse order of hanging_plots for efficiency, set isolated to true.
    plots_to_check = hanging_plots[::-1]
    isolated = True

    for plot in plots_to_check:
        # Gets all neighboring plots
        neighbors = [get_plot(plot, 'S'), get_plot(plot, 'W'), get_plot(plot, 'N'), get_plot(plot, 'E')]
        checked_neighbors = list(set(neighbors) & set(checked_plots))
        if len(checked_neighbors) > 0: # If any of the neighbors can be used for alignment, runs alignment
            try:
                # Loads pcd and neighbors, transforms by the stored accumulated transformations, calculates overlap.
                pcds = {}
                pcd = o3d.io.read_point_cloud(os.path.join(temp_path, f'{plot}TEMP.ply'))
                pcd.transform(acc_tfs[plot])
                overlap = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
                for n in checked_neighbors:
                    pcds[n] = o3d.io.read_point_cloud(os.path.join(temp_path, f'{n}TEMP.ply'))
                    pcds[n].transform(acc_tfs[plot])
                    ref_overlap, tar_overlap = calculate_overlap(pcd, pcds[n])
                    overlap[0] += ref_overlap
                    overlap[1] += tar_overlap

                # Runs ICP
                precise_transformation = icp(overlap[0], overlap[1], threshold, identity, r, max_nn, k)

                # Transforms point cloud, calculates overlap again, and calculates fitness
                pcd.transform(precise_transformation)
                overlap = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
                for n in pcds:
                    ref_overlap, tar_overlap = calculate_overlap(pcd, pcds[n])
                    overlap[0] += ref_overlap
                    overlap[1] += tar_overlap
                fitness_scores[plot] = o3d.pipelines.registration.evaluate_registration(overlap[0], overlap[1], threshold,
                                                                                        identity)

                # Determines if plot is broken
                if fitness_scores[plot].fitness < 0.9:
                    print(plot, 'is broken')
                    broken_plots.append(plot)
                    hanging_plots.remove(plot)
                    continue

                # Stores transformation if plot is good, resets isolated to False so that loop continues.
                checked_plots.append(plot)
                hanging_plots.remove(plot)
                plot_transformations[plot] = precise_transformation @ acc_tfs[plot] @ plot_transformations[plot]
                isolated = False
            except Exception as e:
                print(f"An error occurred reviewing {plot}")
                print(f'Error type: {type(e).__name__}')
                print(f'Error message: {e}')

# Displays fitness values
for key in fitness_scores:
    print(key + ':', fitness_scores[key].fitness)

# Stores point clouds
for plot in plots:
    writer(plot)

# Removes temporary plots. Not running for efficiency purposes.
#shutil.rmtree(r'D:\MLS_alignment\temporary')