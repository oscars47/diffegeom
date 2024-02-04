## heavily adapting this video: https://www.youtube.com/watch?v=C_WwL2mhxfw ##

import open3d as o3d
import numpy as np
import os
from time import time

from subdivide import stl_file_to_ar

def poisson_reconstruct_ar(curve_points, plot_pc_alone=False, plot_normals=True, plot_surface=True):
    '''Implement the Poisson surface reconstruction algorithm on a point cloud'''

    # create point cloud object from the curve points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(curve_points)

    # create a line set from the same points
    if plot_pc_alone:
        lines = [[i, i+1] for i in range(len(curve_points)-1)]  # define lines between consecutive points
        colors = [[1, 0, 0] for i in range(len(lines))]  # color the lines red
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(curve_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # visualize the point cloud and the line set
        o3d.visualization.draw_geometries([pcd, line_set], window_name="Curve Visualization")

    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # clear the normals
    # estimate normals
    pcd.estimate_normals()

    # orient normals
    pcd.orient_normals_consistent_tangent_plane(100)
    if plot_normals:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True, window_name="Point Cloud with Oriented Normals")

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # prepare mesh to be saved
    mesh.compute_triangle_normals()  # Compute face normals
    mesh.compute_vertex_normals()  # Optionally, compute vertex normals as well for other uses


    # Visualize the reconstructed mesh
    if plot_surface:
        o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Reconstruction")

    # save mesh as stl
    # add to folder, reconstruct
    if not os.path.exists('reconstructed'):
        os.makedirs('reconstructed')

    timestamp = str(int(time()))

    o3d.io.write_triangle_mesh(os.path.join('reconstructed', f'poisson_reconstructed_{timestamp}.stl'), mesh)

    return mesh

def poisson_reconstruct_stlfile(stl_file):
    # load the point cloud data
    curve_points = stl_file_to_ar(stl_file)

    # perform Poisson surface reconstruction
    poisson_reconstruct_ar(curve_points)

if __name__ == '__main__':

    def generate_hollow_cube(edge_length, center, num_points=1000):
        points = []
        half_edge = edge_length / 2
        for _ in range(num_points):
            face = np.random.choice(['x', 'y', 'z'])
            sign = np.random.choice([-half_edge, half_edge])
            if face == 'x':
                points.append([sign, np.random.uniform(-half_edge, half_edge), np.random.uniform(-half_edge, half_edge)])
            elif face == 'y':
                points.append([np.random.uniform(-half_edge, half_edge), sign, np.random.uniform(-half_edge, half_edge)])
            else:
                points.append([np.random.uniform(-half_edge, half_edge), np.random.uniform(-half_edge, half_edge), sign])
        return np.array(points) + center

    def generate_hollow_sphere(radius, center, num_points=1000):
        phi = np.random.uniform(0, np.pi, num_points)  # Angle from z-axis
        theta = np.random.uniform(0, 2 * np.pi, num_points)  # Angle from x-axis on the xy-plane
        x = radius * np.sin(phi) * np.cos(theta) + center[0]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[2]
        return np.vstack((x, y, z)).T

    # generate points
    cube_center = [0, 0, 0]  # center of the cube
    sphere_center = [0, -4, 0]  # center of the sphere connected to the cube
    num_points = 5000
    cube_points = generate_hollow_cube(2, cube_center, num_points=num_points)  # Edge length = 2
    sphere_points = generate_hollow_sphere(1, sphere_center, num_points=num_points)  # Radius = 1

    # combine cube and sphere points
    combined_points = np.vstack((cube_points, sphere_points))

    # perform Poisson surface reconstruction
    poisson_reconstruct_ar(combined_points)
