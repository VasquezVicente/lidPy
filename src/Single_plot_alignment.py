from joblib import Parallel, delayed
from itertools import combinations
import open3d as o3d
import pandas as pd
import pdal
import shutil
import laspy
import copy
import time
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import points
import json

# SUGGESTION: Try using 3dfin / forest structural complexities tool to verify alignment.
#################################### Defining variables ############################################

path = r'\\Stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_data_processed'
identity = np.eye(4)
fitness_scores = {}

## spatial indexing of the plots
shp_file=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCI_50ha_timeseries\arcgis\crownmap_50haplot_2025\50haplot_crownmap_2025\BCI_50ha_20x20_Grid\20x20m.shp"
shp = gpd.read_file(shp_file)
shp = shp.rename(columns={"Label": "plot"})

folders=os.listdir(path)
df_plots = pd.DataFrame({'folders': folders})
df_plots['plot'] = df_plots['folders'].str.extract('(\d{4})')

merged= shp.merge(df_plots, on='plot', how='left')
merged['fitness'] = None  ##adding a column for the fitness of the registration
merged['x_local']= merged['X_IDX']*20
merged['y_local']= merged['Y_IDX']*20
merged[merged['folders'].notnull()].plot(figsize=(10, 10))
plt.show()
####################################################################################

columns = {}
plots = []
r = 25
r1 =0 
c = 40
c1 = 6
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
acc_tfs = {}
plot = plots[0]

# These dictionaries will hold the point clouds I am aligning, so that they are easily accessible
pcds = {}
prev_pcds = {}

# plots will be added to these lists after alignment. If alignment was successful, they will be added to checked.
# If alignment was unsuccessful, they will be added to broken. If there were no plots to align them to, they will be added to hanging.
broken_plots = []
checked_plots = []
hanging_plots = []

#Parameters for icp
r = 0.1
max_nn = 80
k = 0.05
threshold = 0.1  #10 CM

# Creating temporary directory to store pre-processed files
temp_path = os.path.join(r'D:\MLS_alignment\temporary2')
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

# Calculating elevation values
elev_raw = np.loadtxt(r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\aux_files\elev.TXT", skiprows=1)
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

get_plot('0600', 'N') # test 

def get_file_path(plot_num):
    for filename in os.listdir(path):
        if filename.startswith(plot_num) and not filename.endswith('alt'):
            return filename
    print(plot_num, 'FILE NOT FOUND')
    return 'FILE NOT FOUND'

get_file_path('0600') # test

def lazO3d(file_path):  # Reads laz files
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def load_las_points(file_path):
    cloud = laspy.read(file_path)
    points = np.empty((len(cloud.x), 3), dtype=np.float32)
    points[:, 0] = cloud.x
    points[:, 1] = cloud.y
    points[:, 2] = cloud.z
    return points

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


side=20
def square_test(plot, ref):
    if len(ref)==4:
        print(f'{plot} REF has 4 reference points')
        dists = [np.linalg.norm(ref[a]-ref[b]) for a,b in combinations(range(4),2)]
        dists.sort()
        sides= dists[:4]
        if any(abs(d - side) > 10 for d in sides): #  10 meter threshold
            print(f'{plot}: Anomalous side detected, removing bad point...')
            scores = []
            for i in range(4):
                err = sum(abs(np.linalg.norm(ref[i]-ref[j]) - side) for j in range(4) if j != i)
                scores.append(err)
            # Remove the point with highest error
            bad_idx = np.argmax(scores)
            print(f'{plot}: Removing point {bad_idx} -> {ref[bad_idx]}')
            best_combo = np.delete(ref, bad_idx, axis=0)
            A, B, C = best_combo[0], best_combo[1], best_combo[2]  # pick 3 corners
            P1 = A + B - C
            P2 = A + C - B
            P3 = B + C - A

            candidates = [P1, P2, P3]
            best = float("inf")
            best_quad = None

            for missing_corner in candidates:
                quad = np.vstack([best_combo, missing_corner])
                # Compute all pairwise distances
                dists = [np.linalg.norm(quad[a] - quad[b]) for a, b in combinations(range(4), 2)]
                dists.sort()
                side_score = sum(abs(d - side) for d in dists[:4])
                diag_score = sum(abs(d - (side * np.sqrt(2))) for d in dists[4:])
                total_score = side_score + diag_score
                if total_score < best:
                    best = total_score
                    best_quad = quad

            print("Best score with missing corner:", best)
            best_combo = sort_ref(best_quad)
            return best_combo, None
        else:
            print(f'{plot} REF has all points are below threshold')
            best_combo = None
            best_score = float("inf")
            for combo in combinations(ref, 4):
                dists = [np.linalg.norm(combo[a]-combo[b]) for a,b in combinations(range(4),2)]
                dists.sort()
                score = sum(abs(d - side) for d in dists[:4]) + \
                        sum(abs(d - (side*np.sqrt(2))) for d in dists[4:])
                if score < best_score:
                    best_combo, best_score = sort_ref(np.array(combo)), score
            print("best score",best_score)
            return best_combo, best_score
    elif len(ref) >= 5:
        best_combo = None
        best_score = float("inf")
        for combo in combinations(ref, 4):
            dists = [np.linalg.norm(combo[a]-combo[b]) for a,b in combinations(range(4),2)]
            dists.sort()
            score = sum(abs(d - side) for d in dists[:4]) + \
                    sum(abs(d - (side*np.sqrt(2))) for d in dists[4:])
            if score < best_score:
                best_combo, best_score = sort_ref(np.array(combo)), score
        print("best score",best_score)
        return best_combo, best_score
    elif len(ref)<=3:
        print(f'{plot} REF has 3 reference points')   ## 3 is really bad considering 2 are equal for a close loop, then collinearity problem arises
        return None, None

def get_tf_from_traj(plot, traj):  ##this should be used when 4 points and detect if 2 points are very very close,a  trycycle
    b_box = traj.get_minimal_oriented_bounding_box()
    E = np.asarray(b_box.extent.copy())  # extent
    C = np.asarray(b_box.center.copy())  # center
    R = np.asarray(b_box.R.copy())  # rotation
    ref = np.array([
        [E[0] / -2, E[1] / -2, E[2] / -2],
        [E[0] / -2, E[1] / 2, E[2] / -2],
        [E[0] / 2, E[1] / -2, E[2] / -2],
        [E[0] / 2, E[1] / 2, E[2] / -2]
    ])
    ref = np.dot(ref, R)
    ref += C
    ref = sort_ref(ref)
    return compute_transformation(ref, target_positions[plot])

def get_crop_from_ref(plot, ref):
    box_min = [target_positions[plot][0][0] - 2.5, target_positions[plot][0][1] - 2.5, min_elev]
    box_max = [target_positions[plot][3][0] + 2.5, target_positions[plot][3][1] + 2.5, max_elev]
    return box_min, box_max

def get_crop_from_traj(plot, traj):
    box_min = traj.get_min_bound()
    box_max = traj.get_max_bound()
    box_min[0] -= 2.5
    box_min[1] -= 2.5
    box_min[2] -= 2.5  # buffer it by 2.5 meters 
    box_max[0] += 2.5
    box_max[1] += 2.5
    box_max[2] += 5  # buffer it by 5 meters to avoid to much weight
    return box_min, box_max

main_dict = {}
#main dict should have an space for prcrustes_transformation

for plot in plots: #testing with first 2 columns
    folder = get_file_path(plot)
    ref_path = os.path.join(path, folder, 'results_trajref.txt')
    traj_path = os.path.join(path, folder, 'results_traj_time.ply')
    pcd_path = os.path.join(path, folder, 'results.laz')
    # Checking if file has been pre-processed
    if not os.path.exists(os.path.join(temp_path, f'{plot}TEMP.ply')):
        if os.path.exists(ref_path):  #if it exists
            ref = np.loadtxt(ref_path, skiprows=1)[:, :3]
            best_combo, best_score = square_test(plot, ref)
            if best_combo is None:
                print(f"{plot}: Not enough reference points, skipping.")
                continue
            t,R,transformation = compute_transformation(best_combo, target_positions[plot])
            if plot not in main_dict:
                main_dict[plot] = {}
            main_dict[plot]['ref_transformation'] = transformation
            main_dict[plot]['square_score'] = best_score
        if os.path.exists(traj_path):
            print(f"Attempting to read trajectory for plot: {plot} at {traj_path}") 
            traj = o3d.io.read_point_cloud(traj_path)
            box_min, box_max = get_crop_from_traj(plot, traj.transform(transformation))
            if plot not in main_dict:
                main_dict[plot] = {}
            main_dict[plot]['box_min'] = box_min
            main_dict[plot]['box_max'] = box_max
            b_box = traj.get_minimal_oriented_bounding_box()
            E = np.asarray(b_box.extent.copy())
            E = [max(E), np.median(E), min(E)]
            if 20 < E[0] < 30 and 20 < E[1] < 30:
                _, _, transformation2 = get_tf_from_traj(plot, traj)
                if plot not in main_dict:
                    main_dict[plot] = {}
                main_dict[plot]['traj_transformation'] = transformation2
            else:
                print("trajectory doesnt meet size requirements")

for plot in plots:
    try:
        temp_laz = os.path.join(temp_path, f'{plot}TEMP.laz')
        laz_path = os.path.join(path, get_file_path(plot), 'results.laz')
        if not os.path.exists(temp_laz) and plot in main_dict:
            if os.path.exists(laz_path):
                print(f'Processing {plot} with PDAL...')
                pipeline = {
                    "pipeline": [
                        laz_path,
                        {
                            "type": "filters.transformation",
                            "matrix": " ".join(map(str, main_dict[plot]['ref_transformation'].flatten()))
                        },
                        {
                            "type": "filters.crop",
                            "bounds": f"([{main_dict[plot]['box_min'][0]},{main_dict[plot]['box_max'][0]}],"
                                      f"[{main_dict[plot]['box_min'][1]},{main_dict[plot]['box_max'][1]}],"
                                      f"[{main_dict[plot]['box_min'][2]},{main_dict[plot]['box_max'][2]}])"
                        },
                        {
                            "type": "filters.smrf",
                            "window": 12.0,
                            "slope": 0.15,
                            "threshold": 0.5,
                            "scalar": 1.25
                        },
                        {
                            "type": "filters.hag_nn"
                        },
                        {
                            "type": "filters.range",
                            "limits": "HeightAboveGround[0:2]"
                        },
                        temp_laz
                    ]
                }
                time1 = time.time()
                try:
                    pipeline_obj = pdal.Pipeline(json.dumps(pipeline))
                    pipeline_obj.execute()
                    time2 = time.time()
                    print(f"PDAL processing time for {plot}: {time2 - time1:.2f} seconds")
                except Exception as e:
                    print(f"PDAL processing failed for plot {plot}: {e}")
            else:
                print(f"results.laz for {plot} does not exist, skipping.")
        else:
            print(f"Temporary file for {plot} already exists or plot not in main_dict, skipping processing.")
    except Exception as e:
        print(f"Error processing plot {plot}: {e}")
        continue


##################################cut off the calculations right here##################################
checked_plots = []
broken_plots = []
hanging_plots = []
for column in columns:   #columns is not defined
    accumulated_transformation = identity
    prev_pcds = pcds
    pcds = {}

    # Loop through plots in the column to align them
    for plot in columns[column]:
        try:
            t1 = time.time()

            # First, checks if plot was loaded properly
            if plot in broken_plots:   #where is broken_plots defined? A: Broken_plots is defined as an empty list, and added to as broken plots are found.
                print(f'{plot} was given as broken')
                continue

            # Loads plot into pcds
            pcds[plot] = o3d.io.read_point_cloud(os.path.join(temp_path, f'{plot}TEMP.ply'))  #PCDS arent defined and its is unclear where do they come from
            #A: This is the only segment where I load pcds in this loop. I am loading the cropped files I created earlier.
             
            # Checks if this is the first plot to be loaded. In the future this should be removed.
            if len(checked_plots) == 0:  ## if the length of the tuple checked plots is 0, then it means that no plots have been aligned yet.
                print('First plot loaded, this is the reference')
                checked_plots.append(plot)
                acc_tfs[plot] = accumulated_transformation
                continue

            # Finds adjacent plot numbers
            prev_plot = get_plot(plot, 'S')
            print(prev_plot)
            left_plot = get_plot(plot, 'W')
            print(left_plot)
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
            #A: The accumulated transformation is, as the name implies, an accumulation of all the previous transformations in that column/row.
            # I do this so that I can maintain the same relative positions between the plots I am aligning; otherwise the target plot would be in an un-ideal position relative to the reference.

            # Calculates overlap and stores point cloud
            overlap = new_calculate_overlap(plot, prev_plot, left_plot, identity, alignment_type)
            precise_transformation = icp(overlap[0], overlap[1], threshold, identity, r, max_nn, k)

            # Calculate and store fitness values
            overlap = new_calculate_overlap(plot, prev_plot, left_plot, precise_transformation, alignment_type)
            fitness_scores[plot] = o3d.pipelines.registration.evaluate_registration(overlap[0], overlap[1],
                                                                                    threshold, identity)

            # Checks fitness scores to verify if plot is broken or not
            print(fitness_scores[plot].fitness)
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




    # get the bounding box from the transformed trajectory
    if transformation is not None:
        traj = o3d.io.read_point_cloud(traj_path)
        box_min, box_max = get_crop_from_traj(plot, traj.transform(transformation))


    ##LOAD THE POINT CLOUD
    pcd=load_las_points(pcd_path)  #read the pcd eats time 
    print(f'{plot} point cloud loaded, {len(pcd)} points')
    points = np.dot(pcd, R.T)
    points += t

    mask = np.all((points >= box_min) & (points <= box_max), axis=1)
    points = points[mask]

    
    points_shorten = shorten(points, 0.5, 2)  #then shorten function eats the rest
    print(f'{plot} point cloud shortened, {len(points_shorten)} points')

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points_shorten)
    o3d.io.write_point_cloud(os.path.join(temp_path, f'{plot}TEMP.ply'), pcd_out)
    print(plot, 'done')

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

        ref_temp, tar_temp = calculate_overlap(temp, pcds[vert_plot])
        reference += ref_temp
        target += tar_temp
        ref_temp, tar_temp = calculate_overlap(temp, prev_pcds[hori_plot])
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
        #draw_registration_result(reference_cloud, target_cloud, reg_result.transformation)
        if final_fitness_match <= 1 and final_fitness_match > initial_fitness_match:
            print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
            output = reg_result.transformation @ output
            reference_cloud.transform(reg_result.transformation)
        else:
            print('fitness did not improve. final fitness is', initial_fitness_match)
    return output


############################################### PRE-PROCESSING #####################################################
### I insist proesecutes analysis is not heavy, time parallelization is not necessary. only slow because the time it takes to load and write temp clouds.
# The proscutes analysis isn't the heavy part, it's the DSM calculation and cropping the point cloud based off that.
t1 = time.time()
# Parallelized pre-processing. I have found this to be significantly faster, although the code still has some issues with the shorten function
pre_process = Parallel(n_jobs=-1)(delayed(create_aligned_pcd)(plot) for plot in plots)

# Pre-processing returns a list of dictionaries in the format {str(plot_num): bool}
# This loops through that output and determines which point clouds were not able to be loaded.
for item in pre_process:
    key = list(item.keys())[0]
    print(key)
    if not item[key][0]:
        print(key, 'failed')
        broken_plots.append(key)
    else:
        plot_transformations[key] = item[key][1]
t2 = time.time()
print(f'Time to pre-process: \n===> {t2 - t1}')

################################################## MAIN LOOP ########################################################
# columns is a dictionary of the form {'Column_number': ['plot_number', 'plot_number', ...], ...}

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
                pcd.transform(acc_tfs[checked_neighbors[0]])
                overlap = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
                for n in checked_neighbors:
                    pcds[n] = o3d.io.read_point_cloud(os.path.join(temp_path, f'{n}TEMP.ply'))
                    pcds[n].transform(acc_tfs[n])
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
                acc_tfs[plot] = precise_transformation @ acc_tfs[checked_neighbors[0]]
                plot_transformations[plot] = acc_tfs[plot] @ plot_transformations[plot]
                isolated = False
            except Exception as e:
                print(f"An error occurred reviewing {plot}")
                print(f'Error type: {type(e).__name__}')
                print(f'Error message: {e}')

try:
    # Displays fitness values
    new_fitness_scores = []
    for key in fitness_scores:
        print(key + ':', fitness_scores[key].fitness)
        new_fitness_scores.append(key + ': ' + str(fitness_scores[key].fitness))

    results_file = open(os.path.join('D:\MLS_alignment', 'fitness_scores.txt'), 'w')
    results_file.writelines(new_fitness_scores)
    results_file.close()
except Exception as e:
    print(f"An error occurred storing fitness scores")
    print(f'Error type: {type(e).__name__}')
    print(f'Error message: {e}')

# Stores point clouds
for plot in plots:
    try:
        # Reading the point cloud
        folder = get_file_path(plot)
        traj = o3d.io.read_point_cloud(os.path.join(path, folder, 'results_traj_time.ply'))
        traj.transform(plot_transformations[plot])
        pcd = lazO3d(os.path.join(path, folder, 'results.laz'))
        pcd.transform(plot_transformations[plot])

        save_path = os.path.join(r'\\Stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_aligned', plot)
        if plot in checked_plots:
            save_path = os.path.join(r'\\Stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_aligned', plot)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            o3d.io.write_point_cloud(os.path.join(save_path, 'results_traj_time_aligned.ply'), traj)
            o3d.io.write_point_cloud(os.path.join(save_path, 'results_aligned.ply'), pcd)
            print(f'{plot} written successfully')
        else:
            save_path = os.path.join(r'\\Stri-sm01\ForestLandscapes\LandscapeProducts\MLS\2023\BCI_50ha_aligned', plot + '_BAD')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            o3d.io.write_point_cloud(os.path.join(save_path, 'results_traj_time_aligned.ply'), traj)
            o3d.io.write_point_cloud(os.path.join(save_path, 'results_aligned.ply'), pcd)
            print(f'{plot} written successfully')
    except Exception as e:
        print(f'An error ocurred saving {plot}')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {e}')

# Removes temporary plots. Not running for efficiency purposes.
shutil.rmtree(r'D:\MLS_alignment\temporary')