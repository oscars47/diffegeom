import numpy as np
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d 
from copy import deepcopy
import os

### ------------------ read file/convert from mesh to ar ------------------ ###
def read_stl(file_path):
    # Read STL file
    file = mesh.Mesh.from_file(file_path)
    # print(file.vectors)
    return file

def convert_stl_to_ar(stl_mesh):
    '''Convert an stl mesh to an array'''
    # convert to stl format
    points = []
    for triangle in stl_mesh.vectors:
        for vertex in triangle:
            x = vertex[0]
            y = vertex[1]
            z = vertex[2]
            points.append([x,y,z])

    # convert to array
    points = np.array(points)
    return points

## --------- bouding box and center --------- ##

def find_bounding_box_stl(stl_mesh):
    min_coords = np.min(stl_mesh.vectors.reshape(-1, 3), axis=0)
    max_coords = np.max(stl_mesh.vectors.reshape(-1, 3), axis=0)
    return min_coords, max_coords

def find_bounding_box_ar(array):
    min_coords = np.min(array, axis=0)
    max_coords = np.max(array, axis=0)
    return min_coords, max_coords

def find_center(bounding_box):
    min_coords, max_coords = bounding_box
    center = (min_coords + max_coords) / 2
    return center

def transform_to_cube_stl(stl_mesh, bounding_box):
    min_coords, max_coords = bounding_box
    lengths = max_coords - min_coords

    # Target length is the maximum length among x, y, z dimensions
    target_length = np.max(lengths)

    # Scale factors for each axis
    scale_factors = target_length / lengths

    # Create scaling matrix
    scaling_matrix = np.diag(scale_factors)

    # Apply the scaling transformation to each vertex
    for i in range(len(stl_mesh.vectors)):
        for j in range(3): # Each triangle has 3 vertices
            # stl_mesh.vectors[i][j] = np.dot(scaling_matrix, stl_mesh.vectors[i][j] - min_coords) + min_coords
            stl_mesh.vectors[i][j] = np.dot(scaling_matrix, stl_mesh.vectors[i][j])

    return stl_mesh, scaling_matrix

def transform_to_cube_ar(array, bounding_box):
    min_coords, max_coords = bounding_box
    lengths = max_coords - min_coords

    # Target length is the maximum length among x, y, z dimensions
    target_length = np.max(lengths)

    # Scale factors for each axis
    scale_factors = target_length / lengths

    # Create scaling matrix
    scaling_matrix = np.diag(scale_factors)

    # Apply the scaling transformation to each vertex
    for i in range(len(array)):
        array[i] = np.dot(scaling_matrix, array[i])

    return array, scaling_matrix

### ------------------ subdivide ------------------ ###
def divide_and_subdivide_sections_stl(stl_mesh, center, delta, threshold, plot_intermediate=False, savename=None):
    subdivisions = {}
    num_phi_sections = int(np.pi / delta) + 1
    num_theta_sections = int(2 * np.pi / delta) + 1

    # Initialize subdivisions
    for phi_index in range(num_phi_sections):
        for theta_index in range(num_theta_sections):
            section_key = (phi_index, theta_index)
            subdivisions[section_key] = [[], delta]

    # Populate subdivisions with vertices
    # log the r, phi, theta of each vertex in the corresponding section
    r_ls = []
    phi_ls = []
    theta_ls = []
    for triangle in stl_mesh.vectors:
        for vertex in triangle:
            shifted_vertex = vertex - center
            r, phi, theta = cartesian_to_spherical(*shifted_vertex)
            r_ls.append(r)
            phi_ls.append(phi)
            theta_ls.append(theta)

            phi_section = phi // delta
            theta_section = (theta + np.pi) // delta  # Adjust theta to be in the range [0, 2π]

            section_key = (phi_section, theta_section)
            subdivisions[section_key][0].append((r, phi, theta))
            # print(f"Vertex assigned: {shifted_vertex} -> Section: {section_key}")

    if plot_intermediate:
        # plot r, phi, theta in 3d
        # convert to cartesian
        x_ls = []
        y_ls = []
        z_ls = []

        for r, phi, theta in zip(r_ls, phi_ls, theta_ls):
            x, y, z = r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)
            x_ls.append(x)
            y_ls.append(y)
            z_ls.append(z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_ls, y_ls, z_ls)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        if savename is not None:
            # add to 'figures' folder
            if not os.path.exists('figures'):
                os.makedirs('figures')
            plt.savefig(os.path.join('figures', savename+'_intermediate.png'))
        plt.show()
    
    # Now, subdivide sections based on the threshold
    final_subdivisions = {}
    for section_key, vectors in subdivisions.items():
        # print([vertex[0] for vertex in vertices])
        subdivided = subdivide_section(section_key, vectors, threshold)
        final_subdivisions.update(subdivided)

    return final_subdivisions

def divide_and_subdivide_sections_ar(array, center, delta, threshold, plot_intermediate=False, savename=None):
    subdivisions = {}
    num_phi_sections = int(np.pi / delta) + 1
    num_theta_sections = int(2 * np.pi / delta) + 1

    # Initialize subdivisions
    for phi_index in range(num_phi_sections):
        for theta_index in range(num_theta_sections):
            section_key = (phi_index, theta_index)
            subdivisions[section_key] = [[], delta]

    # Populate subdivisions with vertices
    # log the r, phi, theta of each vertex in the corresponding section
    r_ls = []
    phi_ls = []
    theta_ls = []
    for vertex in array:
        shifted_vertex = vertex - center
        r, phi, theta = cartesian_to_spherical(*shifted_vertex)
        r_ls.append(r)
        phi_ls.append(phi)
        theta_ls.append(theta)

        phi_section = phi // delta
        theta_section = (theta + np.pi) // delta

        section_key = (phi_section, theta_section)
        subdivisions[section_key][0].append((r, phi, theta))
        # print(f"Vertex assigned: {shifted_vertex} -> Section: {section_key}")

    if plot_intermediate:
        # plot r, phi, theta in 3d
        # convert to cartesian
        x_ls = []
        y_ls = []
        z_ls = []

        for r, phi, theta in zip(r_ls, phi_ls, theta_ls):
            x, y, z = r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)
            x_ls.append(x)
            y_ls.append(y)
            z_ls.append(z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_ls, y_ls, z_ls)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        if savename is not None:
            # add to 'figures' folder
            if not os.path.exists('figures'):
                os.makedirs('figures')
            plt.savefig(os.path.join('figures', savename+'_intermediate.png'))
        plt.show()

    # Now, subdivide sections based on the threshold
    final_subdivisions = {}
    for section_key, vectors in subdivisions.items():
        # print([vertex[0] for vertex in vertices])
        subdivided = subdivide_section(section_key, vectors, threshold)
        final_subdivisions.update(subdivided)

    return final_subdivisions


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)  # Inclination
    theta = np.arctan2(y, x)  # Azimuth
    return r, phi, theta

def spherical_to_cartesian(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def subdivide_section(section_key, vectors,threshold):
    vertices = deepcopy(vectors[0])
    delta = vectors[1]
    # radius is 0th element of each vertex
    radii = [vertex[0] for vertex in vertices]
    # mean_radius = np.mean(radii)
    std_radius = np.std(radii)

    # If the standard deviation is greater than the threshold, subdivide the section into a square
    if std_radius > threshold and len(vertices) > 1:
        new_delta = delta / 2  # subdivide the section
        new_subdivisions = {}

        ## update the phi and theta indices ##
        phi_index, theta_index = section_key
        # get phi and theta
        phi = phi_index * delta
        theta = theta_index * delta
        # get the new phi and theta indices
        new_phi_index = phi // new_delta
        new_theta_index = theta // new_delta

        # give a list of the new subdivisions
        new_phi_indices = [new_phi_index, new_phi_index+1]
        new_theta_indices = [new_theta_index, new_theta_index+1]

        # Initialize new subdivisions
        for new_phi_index in new_phi_indices:
            for new_theta_index in new_theta_indices:
                new_key = (new_phi_index, new_theta_index)
                new_subdivisions[new_key] = [[], new_delta]

        # Assign vertices to new subdivisions
        for r, phi, theta in vertices:
            new_phi_index = phi // new_delta
            new_theta_index = (theta + np.pi) // new_delta # Adjust theta to be in the range [0, 2π]

            # check if key exists
            new_key = (new_phi_index, new_theta_index)
            if new_key in new_subdivisions.keys():
                new_subdivisions[new_key][0].append((r, phi, theta))
            else: ## Note: sometimes the keys it wanted to add were close to the existing but not exactly, so I allowed it to add a new key
                # print(f"Vertex out of range in subdivide: {new_phi_index}, {new_theta_index} -> Phi: {phi}, Theta: {theta}")
                # print(f'possible keys: {new_subdivisions.keys()}')
                # print('adding new key')
                new_subdivisions[new_key] = [[(r, phi, theta)], new_delta]
                # print(f'new_subdivisions: {new_subdivisions[new_key]}')

        # Recursively subdivide further if necessary
        final_subdivisions = {}
        for new_key, new_vectors in new_subdivisions.items():
            subdivided = subdivide_section(new_key, new_vectors, threshold)

            final_subdivisions.update(subdivided)

        return final_subdivisions
    else:
        # If the standard deviation is within the threshold, don't subdivide further
        return {section_key: vectors}

def calculate_mean_std_in_sections(sections):
    stats = {}

    for section, radii in sections.items():
        if len(radii)>0:  # Check if the section has any points
            mean_radius = np.mean(radii)
            std_radius = np.std(radii)
            stats[section] = (mean_radius, std_radius)
        else:
            print(f"Section {section} has no points.")

    return stats

def calculate_centroid_of_section(section_key, division_dict):
    # print(division_dict)
    phi_index, theta_index = section_key
    if len(division_dict[section_key]) == 2:
        delta = division_dict[section_key][1]
        centroid_phi = (phi_index + 0.5) * delta
        centroid_theta = (theta_index + 0.5) * delta
        return centroid_phi, centroid_theta
    else:
        print(f"Section {section_key} has no points.")
        return None, None

def plot_subdivisions(subdivisions, savename=None):
    phi_values = []
    theta_values = []
    radii_values = []

    for section_key, vectors in subdivisions.items():
        if len(subdivisions[section_key]) > 0:
            centroid_phi, centroid_theta = calculate_centroid_of_section(section_key, subdivisions)
            mean_radii = np.mean([r for r, phi, theta in vectors[0]])

            phi_values.append(np.rad2deg(centroid_phi))
            theta_values.append(np.rad2deg(centroid_theta))
            radii_values.append(mean_radii)
        else:
            print(f"Section {section_key} has no points.")

    if not phi_values or not theta_values:
        print("No data to plot.")
        return
    
    radii_values = np.array(radii_values)
    phi_values = np.array(phi_values)
    theta_values = np.array(theta_values)
    
    # Convert spherical coordinates to Cartesian coordinates for the 3D plot
    x, y, z = spherical_to_cartesian(radii_values, phi_values, theta_values)

    # Create a figure with two subplots
    fig = plt.figure(figsize=(15, 6))

    # First subplot (original 2D scatter plot)
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(theta_values, phi_values, s=50, c=radii_values, cmap='viridis')
    plt.colorbar(scatter, ax=ax1, label='Mean Radius')
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Phi (degrees)')
    ax1.set_title('Centroids of Subdivisions with Radii Values')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 180)

    # Second subplot (3D plot)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter3d = ax2.scatter(x, y, z, c=radii_values, cmap='viridis')
    plt.colorbar(scatter3d, ax=ax2, label='Mean Radius')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Plot in Polar Coordinates')

    if savename is not None:
            # add to 'figures' folder
            if not os.path.exists('figures'):
                os.makedirs('figures')
            plt.savefig(os.path.join('figures', savename+'_result.png'))

    plt.show()


## ------------------ main ------------------ ##
def main_subdivision_stl(file_path=None, stl_file=None, threshold=1e-3, delta=30, plot_intermediate=False):
    '''
    * `read_stl` loads the mesh
    * `find_bounding box` draws a rectangular prism around the mesh
    * `find_center` computes the center of the box
    * `transform_to_cube_stl` performs linear transformation to make box uniform
    * recall `bounding_box` and `find_center` to get the new bounding box box and center
    * `divide_and_subdivide_sections` performs the slicing first based on an initial array of phi, theta. in particular, we go through each triangle in the mesh, each point in the triangle, compute the r, phi, theta vals, and see which of the theta, phi bins they fall into. if `plot_intermediate` is called, will output what the r, phi, theta looks like. then, calls subdivide_section to recursively divide each of the main sections into smaller ones by finding the std deviation of the radii from the center, which we want to be <= a certain threshold (by default 1e-3)
    * `plot_subdivisions` to see the results in phi, theta space colored by radius, as well as a 3D plot
    '''
    if stl_file is not None and file_path is None:
        print('Using stl_file')
        stl_mesh = stl_file

    elif file_path is not None and stl_file is None:
        print("Reading STL file...")
        stl_mesh = read_stl(file_path)

    else:
        raise ValueError('Must provide either file_path or stl_file')

    delta = np.deg2rad(delta)

    print("Finding bounding box...")
    bounding_box = find_bounding_box_stl(stl_mesh)


    print("Transforming to cube...")
    transformed_mesh, scaling_matrix = transform_to_cube_stl(stl_mesh, bounding_box)


    print("Finding center of revised box...")
    new_bounding_box = find_bounding_box_stl(transformed_mesh)
    center = find_center(new_bounding_box)

    print("Dividing and subdividing sections...")
    final_subdivisions = divide_and_subdivide_sections_stl(transformed_mesh, center, delta, threshold, plot_intermediate=plot_intermediate)

    # Plot the subdivisions
    plot_subdivisions(final_subdivisions)

def main_subdivision_ar(file_path =None, array = None, threshold=1e-3, delta=30, plot_intermediate=False, savename=None):
    '''
    * `read_stl` loads the mesh
    * `find_bounding box` draws a rectangular prism around the mesh
    * `find_center` computes the center of the box
    * `transform_to_cube_stl` performs linear transformation to make box uniform
    * recall `bounding_box` and `find_center` to get the new bounding box box and center
    * `divide_and_subdivide_sections` performs the slicing first based on an initial array of phi, theta. in particular, we go through each triangle in the mesh, each point in the triangle, compute the r, phi, theta vals, and see which of the theta, phi bins they fall into. if `plot_intermediate` is called, will output what the r, phi, theta looks like. then, calls subdivide_section to recursively divide each of the main sections into smaller ones by finding the std deviation of the radii from the center, which we want to be <= a certain threshold (by default 1e-3)
    * `plot_subdivisions` to see the results in phi, theta space colored by radius, as well as a 3D plot
    '''

    if array is not None and file_path is None:
        print('Using array')
        array = array

    elif file_path is not None and array is None:
        print("Reading STL file...")
        array = read_stl(file_path).vectors.reshape(-1, 3)

    else:
        raise ValueError('Must provide either file_path or array')

    delta = np.deg2rad(delta)

    print("Finding bounding box...")
    bounding_box = find_bounding_box_ar(array)


    print("Transforming to cube...")
    transformed_mesh, scaling_matrix = transform_to_cube_ar(array, bounding_box)


    print("Finding center of revised box...")
    new_bounding_box = find_bounding_box_ar(transformed_mesh)
    center = find_center(new_bounding_box)

    print("Dividing and subdividing sections...")
    final_subdivisions = divide_and_subdivide_sections_ar(transformed_mesh, center, delta, threshold, plot_intermediate=plot_intermediate, savename=savename)

    # Plot the subdivisions
    plot_subdivisions(final_subdivisions, savename=savename)

if __name__ == '__main__':
    file_path = "shapewithbox.stl"
    main_subdivision_stl(file_path)

   