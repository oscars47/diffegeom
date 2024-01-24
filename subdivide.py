import numpy as np
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d 
from copy import deepcopy

### ------------------ read file and transform ------------------ ###
def read_stl(file_path):
    # Read STL file
    file = mesh.Mesh.from_file(file_path)
    # print(file.vectors)
    return file

def find_bounding_box(stl_mesh):
    min_coords = np.min(stl_mesh.vectors.reshape(-1, 3), axis=0)
    max_coords = np.max(stl_mesh.vectors.reshape(-1, 3), axis=0)
    return min_coords, max_coords

def find_center(bounding_box):
    min_coords, max_coords = bounding_box
    center = (min_coords + max_coords) / 2
    return center

def transform_to_cube(stl_mesh, bounding_box):
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

### ------------------ subdivide ------------------ ###
def divide_and_subdivide_sections(stl_mesh, center, delta, threshold):
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

def plot_subdivisions(subdivisions):
    phi_values = []
    theta_values = []
    radii_values = []

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

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

    scatter = ax.scatter(theta_values, phi_values, s=50, c=radii_values, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Mean Radius')
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Phi (degrees)')
    ax.set_title('Centroids of Subdivisions with Radii Values')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 180)

    plt.show()

if __name__ == '__main__':
    file_path = "shapewithbox.stl"

    threshold = 0.001
    delta = np.deg2rad(30)

    print("Reading STL file...")
    stl_mesh = read_stl(file_path)

    print("Finding bounding box...")
    bounding_box = find_bounding_box(stl_mesh)


    print("Transforming to cube...")
    transformed_mesh, scaling_matrix = transform_to_cube(stl_mesh, bounding_box)


    print("Finding center of revised box...")
    new_bounding_box = find_bounding_box(transformed_mesh)
    center = find_center(new_bounding_box)

    print("Dividing and subdividing sections...")
    final_subdivisions = divide_and_subdivide_sections(transformed_mesh, center, delta, threshold)

    # Plot the subdivisions
    plot_subdivisions(final_subdivisions)