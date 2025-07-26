import os
import laspy
import open3d as o3d
import numpy as np

target_positions = {}
columns = {}
plots = []

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

def lazO3d(file_path):  # Reads laz files
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def calc_target(plot):
    col = int(plot[:2])
    row = int(plot[2:])
    target_positions[plot] = np.array([
        [col * 20, row * 20, 0],
        [col * 20, (row + 1) * 20, 0],
        [(col + 1) * 20, row * 20, 0],
        [(col + 1) * 20, (row + 1) * 20, 1000]
    ])

path = r'D:\MLS_alignment'
complete_cloud = o3d.geometry.PointCloud()
for plot in plots:
    if os.path.exists(os.path.join(path, plot, 'processed.laz')):
        print(f'{plot} started')
        pcd = lazO3d(os.path.join(path, plot, 'processed.laz'))
        print(f'{plot} loaded')
        calc_target(plot)
        b_box = o3d.geometry.AxisAlignedBoundingBox(target_positions[plot][0], target_positions[plot][3])
        pcd = pcd.crop(b_box)
        print(f'{plot} cropped')
        pcd = pcd.voxel_down_sample(voxel_size=0.2)
        print(f'{plot} down sampled')
        complete_cloud += pcd
        print(f'============================ {plot} COMPLETE ============================')

o3d.visualization.draw_geometries([complete_cloud])