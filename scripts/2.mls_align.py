import laspy
import open3d as o3d
import numpy as np
import copy
import os
import re

target_positions = {
    "3020": np.array([
        [0, 0, 0], [0, 20, 0], [20, 0, 0], [20, 20, 0]
    ]),
    "3021": np.array([
        [0, 20, 0], [0, 40, 0], [20, 20, 0], [20, 40, 0]
    ]),
    "3022": np.array([
        [0, 40, 0], [0, 60, 0], [20, 40, 0], [20, 60, 0]
    ]),
    "3023": np.array([
        [0, 60, 0], [0, 80, 0], [20, 60, 0], [20, 80, 0]
    ]),
    "3024": np.array([
        [0, 80, 0], [0, 100, 0], [20, 80, 0], [20, 100, 0]
    ]),
    "3120": np.array([
        [20, 0, 0], [20, 20, 0], [40, 0, 0], [40, 20, 0]
    ]),
    "3121": np.array([
        [20, 20, 0], [20, 40, 0], [40, 20, 0], [40, 40, 0]
    ]),
    "3122": np.array([
        [20, 40, 0], [20, 60, 0], [40, 40, 0], [40, 60, 0]
    ]),
    "3123": np.array([
        [20, 60, 0], [20, 80, 0], [40, 60, 0], [40, 80, 0]
    ]),
    "3124": np.array([
        [20, 80, 0], [20, 100, 0], [40, 80, 0], [40, 100, 0]
    ]),
    "3220": np.array([
        [40, 0, 0], [40, 20, 0], [60, 0, 0], [60, 20, 0]
    ]),
    "3221": np.array([
        [40, 20, 0], [40, 40, 0], [60, 20, 0], [60, 40, 0]
    ]),
    "3222": np.array([
        [40, 40, 0], [40, 60, 0], [60, 40, 0], [60, 60, 0]
    ]),
    "3223": np.array([
        [40, 60, 0], [40, 80, 0], [60, 60, 0], [60, 80, 0]
    ]),
    "3224": np.array([
        [40, 80, 0], [40, 100, 0], [60, 80, 0], [60, 100, 0]
    ]),
    "3320": np.array([
        [60, 0, 0], [60, 20, 0], [80, 0, 0], [80, 20, 0]
    ]),
    "3321": np.array([
        [60, 20, 0], [60, 40, 0], [80, 20, 0], [80, 40, 0]
    ]),
    "3322": np.array([
        [60, 40, 0], [60, 60, 0], [80, 40, 0], [80, 60, 0]
    ]),
    "3323": np.array([
        [60, 60, 0], [60, 80, 0], [80, 60, 0], [80, 80, 0]
    ]),
    "3324": np.array([
        [60, 80, 0], [60, 100, 0], [80, 80, 0], [80, 100, 0]
    ]),
    "3420": np.array([
        [80, 0, 0], [80, 20, 0], [100, 0, 0], [100, 20, 0]
    ]),
    "3421": np.array([
        [80, 20, 0], [80, 40, 0], [100, 20, 0], [100, 40, 0]
    ]),
    "3422": np.array([
        [80, 40, 0], [80, 60, 0], [100, 40, 0], [100, 60, 0]
    ]),
    "3423": np.array([
        [80, 60, 0], [80, 80, 0], [100, 60, 0], [100, 80, 0]
    ]),
    "3424": np.array([
        [80, 80, 0], [80, 100, 0], [100, 80, 0], [100, 100, 0]
    ]),
}
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

    return transformation

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def icp(photo_cloud, lidar_cloud, threshold, identity_matrix,radius, max_nn, k):

    lidar_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    initial_eval_result = o3d.pipelines.registration.evaluate_registration(
        photo_cloud, lidar_cloud, threshold, identity_matrix)
    
    initial= str(initial_eval_result)
    initial_fitness_match = re.search(r'fitness=(.*?),', initial)
    initial_fitness_match = float(initial_fitness_match.group(1))
    final_fitness_match=initial_fitness_match+0.001
    while initial_fitness_match<1 and final_fitness_match>initial_fitness_match:
        #initial evaluation
        initial_eval_result = o3d.pipelines.registration.evaluate_registration(
            photo_cloud, lidar_cloud, threshold, identity_matrix)
        initial= str(initial_eval_result)
        initial_fitness_match = re.search(r'fitness=(.*?),', initial)
        initial_fitness_match = float(initial_fitness_match.group(1))
        #icp iteration
        loss = o3d.pipelines.registration.TukeyLoss(k=k)
        p2p_loss = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_result = o3d.pipelines.registration.registration_icp(
            photo_cloud, lidar_cloud, threshold, identity_matrix, p2p_loss)
        reg_result_str = str(reg_result)
        final_fitness_match = re.search(r'fitness=(.*?),', reg_result_str)
        final_fitness_match = float(final_fitness_match.group(1))
        if final_fitness_match<1 and final_fitness_match>initial_fitness_match:
            print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
            photo_cloud=photo_cloud.transform(reg_result.transformation)
            #save the reg result
            higuest_reg_result=reg_result
            higuest_reg_result_matrix=reg_result.transformation
            print('photo_cloud has been transformed')
        else:
            print('fitness did not improve, while loop will stop')
            print('final fitness is', final_fitness_match)
            print('the last photo cloud was not transformed,it is the correct one')
    return photo_cloud, lidar_cloud, higuest_reg_result, higuest_reg_result_matrix
def lazO3d(file_path):
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def save_o3d_las(cloud, output_path):
    points = np.asarray(cloud.points)
    header = laspy.LasHeader(point_format=3, version="1.4")
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(output_path)

path=r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\MLS"

plots = [
    '3020', '3021', '3022', '3023', '3024',
    '3120', '3121', '3122', '3123', '3124',
    '3220', '3221', '3222', '3223', '3224',
    '3320', '3321', '3322', '3323', '3324',
    '3420', '3421', '3422', '3423', '3424'
]


# this loop is capable of harmonzing the cloud, trajectory, reference of a N number of subplots, given their
# true reference and expected reference. 

len(plots) # we have 25 plots. Each with a cloud, trajectory and reference on their own CRS

for plot in plots:
    print(plot)
    output_path_cloud =os.path.join(path,f"plot1\{plot}_cloud.ply")
    output_path_traj = os.path.join(path,f"plot1\{plot}_traj.ply")
    output_path_matrix = os.path.join(path, f"plot1\\{plot}_transformation.npy")

    if not os.path.exists(output_path_cloud) or not os.path.exists(output_path_traj):
        try:
            traj_path= os.path.join(path,f"{plot}results_traj_time.ply")
            cloud_path= os.path.join(path,f"{plot}results.laz")
            reference= os.path.join(path, f"{plot}results_trajref.txt")
            target_position=target_positions[plot]
            data = np.loadtxt(reference, skiprows=1)
            xyz = data[:, :3]  # First three columns
            source_points = xyz[:4]
            T1 = compute_transformation(source_points, target_position)
            np.save(output_path_matrix, T1)
            traj= o3d.io.read_point_cloud(traj_path)
            traj.transform(T1)
            o3d.io.write_point_cloud(output_path_traj, traj)
            pcd= lazO3d(cloud_path)
            pcd.transform(T1)
            _,_,zmin=pcd.get_min_bound()
            _,_,zmax=pcd.get_max_bound()
            min_bound=traj.get_min_bound()
            max_bound=traj.get_max_bound()
            min_bound[:2] -= 5  
            max_bound[:2] += 5
            min_bound[2:3]= zmin
            max_bound[2:3]= zmax
            cropped_cloud = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
            o3d.io.write_point_cloud(output_path_cloud, cropped_cloud)
        except:
            print("some sort of error")


#to ensure aligment of sequential tiles I am going to need a coffee, I have no idea how complicated this is going to get
#reference tile placed on 0,0

def evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_overlap, target_overlap, threshold, init_matrix)
    return eval_result.fitness, eval_result

def calculate_fitness(reference, target, threshold, init_matrix):
    bbox_reference = reference.get_axis_aligned_bounding_box()
    bbox_target = target.get_axis_aligned_bounding_box()

    min_bound = np.maximum(bbox_reference.min_bound, bbox_target.min_bound)
    max_bound = np.minimum(bbox_reference.max_bound, bbox_target.max_bound)
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    reference_overlap = reference.crop(crop_box)
    target_overlap = target.crop(crop_box)

    fitness, _ = evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix)
    return fitness, reference_overlap, target_overlap


plots = [
    '3020', '3021', '3022', '3023', '3024',
    '3120', '3121', '3122', '3123', '3124',
    '3220', '3221', '3222', '3223', '3224',
    '3320', '3321', '3322', '3323', '3324',
    '3420', '3421', '3422', '3423', '3424'
]

threshold = 0.1
init_matrix = np.identity(4)
score_threshold= 0.99

# Dictionary to store fitness scores
fitness_dict = {}

# Evaluate overlaps for adjacent plots
for plot in plots:
    print("the plot is", plot)
    reference_path = os.path.join(path, "plot1", f"{plot}_cloud.laz")
    reference = lazO3d(reference_path)
    # Get the plot indices
    plot_row = int(plot[0:2])
    plot_col = int(plot[2:4])
    adjacent_plots = []
    if plot_col + 1 <= 24:
        adjacent_plots.append(f"{plot_row}{plot_col + 1:02d}")
    if plot_row + 1 <= 34:
        adjacent_plots.append(f"{plot_row + 1:02d}{plot_col:02d}")

    for adj_plot in adjacent_plots:
        target_path = os.path.join(path, "plot1", f"{adj_plot}_cloud.laz")
        target = lazO3d(target_path)

        # Calculate fitness score
        fitness, reference_overlap, target_overlap = calculate_fitness(reference, target, threshold, init_matrix)
        fitness_dict[(plot, adj_plot)] = fitness
        print(f"Fitness score for {plot} -> {adj_plot}: {fitness}")
        if fitness < score_threshold:
            _,_,_,result_matrix = icp(reference_overlap, target_overlap, threshold, init_matrix, 4, 100, 3.7)
            if result_matrix is not None:
                target_transformed = target.transform(result_matrix)
                save_point_cloud(target_transformed, target_path)
                print(f"Aligned and saved: {adj_plot}")
            else:
                print("The ICP failed, no improvement")
        else:
            print("Fitness is enough")




