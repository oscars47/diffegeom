## heavily adapting this video: https://www.youtube.com/watch?v=C_WwL2mhxfw ##

import open3d as o3d
import numpy as np

from subdivide import stl_file_to_ar

def poisson_reconstruct_ar(curve_points, plot_pc_alone=False, plot_normals=True, plot_surface=True):
    '''Implement the Poisson surface reconstruction algorithm on a point cloud'''

    # Create a point cloud object from the curve points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(curve_points)

    # create a line set from the same points
    if plot_pc_alone:
        lines = [[i, i+1] for i in range(len(curve_points)-1)]  # Define lines between consecutive points
        colors = [[1, 0, 0] for i in range(len(lines))]  # Color the lines red
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

    # Visualize the reconstructed mesh
    if plot_surface:
        o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Reconstruction")

    return mesh

def poisson_reconstruct_stlfile(stl_file):
    # load the point cloud data
    curve_points = stl_file_to_ar(stl_file)

    # perform Poisson surface reconstruction
    poisson_reconstruct_ar(curve_points)
