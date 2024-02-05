import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib.colors as mcolors
def read_stl(file_path):
    with open(file_path, 'r') as file:
        stl_data = file.readlines()
    return stl_data




def parse_stl(stl_data):
    faces = []
    for line in stl_data:
        if line.strip().startswith('vertex'):
            _, x, y, z = line.strip().split()
            faces[-1].append([float(x), float(y), float(z)])
        elif line.strip().startswith('facet'):
            faces.append([])
    return faces




def compute_centroid(face):
    vertices = np.array(face, dtype=np.float64)  # Ensure vertices are in 64-bit float
    centroid = np.mean(vertices, axis=0)
    return centroid




def compute_normal(face):
    v1 = np.array(face[1]) - np.array(face[0])
    v2 = np.array(face[2]) - np.array(face[0])
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)




def estimate_curvature(faces, k=10):
    centroids = np.array([compute_centroid(face) for face in faces])
    normals = np.array([compute_normal(face) for face in faces])
   
    # Using KDTree for efficient nearest neighbor search
    tree = KDTree(centroids)
    curvature_estimates = []




    for i, normal in enumerate(normals):
        _, idxs = tree.query(centroids[i], k=k+1) # +1 because the query includes the point itself
        adjacent_normals = normals[idxs[1:]] # exclude the first one, which is the point itself




        curvature = np.mean([np.arccos(np.clip(np.dot(normal, adj_normal), -1.0, 1.0)) for adj_normal in adjacent_normals])
        curvature_estimates.append((curvature, centroids[i]))




    return curvature_estimates




def find_minimum_curvature_centroid(curvature_estimates):
    return min(curvature_estimates, key=lambda x: x[0])[1]
def get_face_from_centroid(faces, target_centroid):
    for face in faces:
        centroid = compute_centroid(face)
        if np.allclose(centroid, target_centroid):
            return face
    return None
def reverse_transform_point(point, rotation_matrix, translation_vector):
    # Reverse the translation and rotation
    rotated_point = np.dot(rotation_matrix.T, point)  # Transpose matrix for inverse rotation
    translated_point = rotated_point - translation_vector
    return translated_point


# Function to plot all regions
def plot_all_regions_optimized(original_faces, grown_regions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = [plt.cm.jet(float(i)/max(len(grown_regions), 1)) for i in range(len(grown_regions))]


    # Gather all vertices for scaling
    all_vertices = []


    for region_index, region_indices in enumerate(grown_regions):
        region_faces = [original_faces[i] for i in region_indices]
        for face in region_faces:
            all_vertices.extend(face)
        poly3d = [[face[j] for j in range(3)] for face in region_faces]
        collection = Poly3DCollection(poly3d, alpha=1.0, linewidths=1)
        collection.set_facecolor(colors[region_index])
        ax.add_collection3d(collection)


    # Convert to numpy array for easier manipulation
    all_vertices = np.array(all_vertices)


    # Set axes limits
    max_range = np.array([all_vertices[:,0].max()-all_vertices[:,0].min(),
                          all_vertices[:,1].max()-all_vertices[:,1].min(),
                          all_vertices[:,2].max()-all_vertices[:,2].min()]).max() / 2.0


    mid_x = (all_vertices[:,0].max()+all_vertices[:,0].min()) * 0.5
    mid_y = (all_vertices[:,1].max()+all_vertices[:,1].min()) * 0.5
    mid_z = (all_vertices[:,2].max()+all_vertices[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


    plt.show()


def define_new_coordinate_system(face, centroid, normal):
    # Assuming the face is a triangle
    p1, p2, p3 = face




    # X-axis: Vector from one vertex to another in the face
    x_axis = np.array(p2) - np.array(p1)
    x_axis /= np.linalg.norm(x_axis)  # Normalize




    # Y-axis: Cross product of Z-axis and X-axis
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)  # Normalize




    # Z-axis is the normal vector, already normalized
    z_axis = normal




    # Coordinate system transformation matrix for rotation
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T




    # The translation vector is the negative of the centroid
    translation_vector = -np.array(centroid)




    return rotation_matrix, translation_vector
def transform_point(point, rotation_matrix, translation_vector):
    # Apply translation
    translated_point = point + translation_vector
    # Apply rotation
    rotated_point = np.dot(rotation_matrix, translated_point)
    return rotated_point




def transform_mesh(faces, rotation_matrix, translation_vector):
    transformed_faces = []
    for face in faces:
        transformed_face = [transform_point(np.array(vertex), rotation_matrix, translation_vector) for vertex in face]
        transformed_faces.append(transformed_face)
    return transformed_faces




def transform_centroids(centroids, rotation_matrix, translation_vector):
    transformed_centroids = [transform_point(centroid, rotation_matrix, translation_vector) for centroid in centroids]
    return transformed_centroids
def quadratic_surface_equation(params, x, y):
    a, b, c, d, e, f = params
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f




def fit_quadratic_surface(centroids):
    # Prepare the matrix A and vector B for the least squares problem
    A = []
    B = []




    for x, y, z in centroids:
        A.append([x**2, y**2, x*y, x, y, 1])
        B.append(z)




    A = np.array(A)
    B = np.array(B)




    # Solve the least squares problem
    coeffs, residuals, _, _ = np.linalg.lstsq(A, B, rcond=None)




    # The fitting error can be calculated as the square root of the mean of the squared residuals
    error = np.sqrt(residuals[0] / len(B)) if len(residuals) > 0 else 0




    return coeffs, error
def map_edges_to_faces(faces):
    edge_to_faces = {}
    for i, face in enumerate(faces):
        for j in range(3):  # Assuming triangular faces
            # Corrected call to create_edge
            edge = create_edge(face[j], face[(j + 1) % 3])
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(i)
    return edge_to_faces
def create_edge(vertex1, vertex2):
 
    return tuple(sorted([tuple(vertex1), tuple(vertex2)]))
def find_face_index(faces, target_face):
    for i, face in enumerate(faces):
        if all(np.allclose(face_vertex, target_vertex) for face_vertex, target_vertex in zip(face, target_face)):
            return i
    return -1
def find_adjacent_faces(face_index, faces, edge_to_faces):
    adjacent_faces = set()
    face = faces[face_index]




    for i in range(3):  # Assuming triangular faces
        # Corrected call to create_edge with two separate vertices
        edge = create_edge(face[i], face[(i + 1) % 3])
        for adj_face_index in edge_to_faces.get(edge, []):
            if adj_face_index != face_index:
                adjacent_faces.add(adj_face_index)




    return adjacent_faces
def is_edge_a_wall(edge, walls):




    # Sort the vertices of the edge to ensure consistent representation
    sorted_edge = tuple(sorted([tuple(vertex) for vertex in edge]))




    # Check if the sorted edge is in the set of walls
    return sorted_edge in walls
def find_shared_edge(face1, face2):
    """Find the shared edge between two faces, if it exists."""
    edges1 = {create_edge(face1[i], face1[(i + 1) % 3]) for i in range(3)}
    edges2 = {create_edge(face2[i], face2[(i + 1) % 3]) for i in range(3)}
    shared_edge = edges1.intersection(edges2)
    return next(iter(shared_edge), None)
grown_faces_global = set()


def grow_region(transformed_faces, initial_grown_region, edge_to_faces, threshold=0.1):
    walls = set()
    grown_region = set(initial_grown_region)


    # Mark initial faces as grown
    for face_index in initial_grown_region:
        grown_faces_global.add(face_index)


    while True:
        new_faces_added = False
        new_grown_region = set(grown_region)


        for face_index in grown_region:
            adjacent_faces = find_adjacent_faces(face_index, transformed_faces, edge_to_faces)


            for adj_face_index in adjacent_faces:
                if adj_face_index in grown_faces_global:
                    continue  # Skip if already grown


                temp_region = new_grown_region.union({adj_face_index})
                centroids = [compute_centroid(transformed_faces[i]) for i in temp_region]
                _, error = fit_quadratic_surface(centroids)


                if error <= threshold:
                    new_grown_region.add(adj_face_index)
                    grown_faces_global.add(adj_face_index)  # Mark as grown immediately
                    new_faces_added = True
                else:
                    shared_edge = find_shared_edge(transformed_faces[face_index], transformed_faces[adj_face_index])
                    if shared_edge and not is_edge_a_wall(shared_edge, walls):
                        walls.add(shared_edge)


        if not new_faces_added:
            break


        grown_region = new_grown_region


    return grown_region, walls
def quadratic_surface_equation(params, x, y):
    a, b, c, d, e, f = params
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
def transform_face(face, rotation_matrix, translation_vector):
    # Apply the given rotation and translation to all vertices in the face
    return [transform_point(np.array(vertex), rotation_matrix, translation_vector) for vertex in face]
def find_overlapping_centroids(grown_regions, faces):
    centroid_region_map = {}  # Map centroids to regions
    overlapping_centroids = set()  # Store overlapping centroids


    for region_index, region_indices in enumerate(grown_regions):
        for face_index in region_indices:
            centroid = tuple(compute_centroid(faces[face_index]))


            if centroid in centroid_region_map:
                # If the centroid is already mapped to a different region, it's an overlap
                if centroid_region_map[centroid] != region_index:
                    overlapping_centroids.add(centroid)
            else:
                centroid_region_map[centroid] = region_index


    return overlapping_centroids


def doll(file_path, existing_grown_regions):
    # Read and parse the STL file
    stl_data = read_stl(file_path)
    faces = parse_stl(stl_data)


    # Estimate curvature at each centroid
    curvature_estimates = estimate_curvature(faces)


    # Initialize unused_curvature_estimates as an empty list
    unused_curvature_estimates = []


    # Collect centroids of already grown regions
    existing_centroids = set()
    for grown_region_indices, walls, rot_matrix, trans_vector in existing_grown_regions:
        for idx in grown_region_indices:
            existing_centroids.add(tuple(compute_centroid(faces[idx])))


    # Find centroids not part of any grown region
    unused_curvature_estimates = [ce for ce in curvature_estimates if tuple(ce[1]) not in existing_centroids]


    # If there are no unused curvature points available, return None
    if not unused_curvature_estimates:
        print("No more unused curvature points available.")
        return None


    min_curvature_centroid = find_minimum_curvature_centroid(unused_curvature_estimates)


    # Find the corresponding face and compute its normal
    face_with_min_curvature = get_face_from_centroid(faces, min_curvature_centroid)
    normal_vector = compute_normal(face_with_min_curvature)


    # Define the new coordinate system
    rotation_matrix, translation_vector = define_new_coordinate_system(face_with_min_curvature, min_curvature_centroid, normal_vector)


    # Transform the entire mesh and centroids
    transformed_faces = transform_mesh(faces, rotation_matrix, translation_vector)
    transformed_centroids = transform_centroids([compute_centroid(face) for face in faces], rotation_matrix, translation_vector)


    # Map edges to faces for the transformed mesh
    edge_to_faces = map_edges_to_faces(transformed_faces)
    zero_centroid_face_index = next(i for i, face in enumerate(transformed_faces) if np.allclose(compute_centroid(face), [0.0, 0.0, 0.0]))


    # Initialize grown_region as a set containing the zero_centroid_face_index
    initial_grown_region = {zero_centroid_face_index}


    grown_region, walls = grow_region(transformed_faces, initial_grown_region, edge_to_faces)
   
    # Calculate the quadratic coefficients for the grown region
    grown_region_centroids = [compute_centroid(transformed_faces[i]) for i in grown_region]
    quadratic_coeffs, fitting_error = fit_quadratic_surface(grown_region_centroids)
    print(f"Fitting error: {fitting_error}")


    existing_grown_regions.append((grown_region, walls, rotation_matrix, translation_vector))
    grown_faces_global.update(grown_region)


    # Return the updated list of grown regions
    return existing_grown_regions
def check_for_repeating_indices(grown_regions):
    all_indices = set()
    for region in grown_regions:
        for index in region:
            if index in all_indices:
                return True  # Found a repeating index
            all_indices.add(index)
    return False
   


file_path = r"C:\Users\Larry\Downloads\ball_dent.STL" # Update with the correct path
existing_grown_regions = []
stl_data = read_stl(file_path)
original_faces = parse_stl(stl_data)


i = 0
while True:
    result = doll(file_path, existing_grown_regions)
    i += 1
    print(f"Number of regions processed: {i}")
    if result is None:
        break
    existing_grown_regions = result


# Extract just the indices of the regions for plotting
all_grown_regions_indices = [region_indices for region_indices, walls, rot_matrix, trans_vector in existing_grown_regions]


# Find overlapping points using the extracted indices
overlapping_centroids = find_overlapping_centroids(all_grown_regions_indices, original_faces)
print(f"Number of overlapping centroids: {len(overlapping_centroids)}")


# Plot all regions using the updated function
plot_all_regions_optimized(original_faces, all_grown_regions_indices)


# Count the number of faces and centroids
num_faces = len(original_faces)
centroids = [compute_centroid(face) for face in original_faces]
num_centroids = len(centroids)
print("Number of faces:", num_faces)
print("Number of centroids:", num_centroids)


# Check for repeating indices in regions
repeating_indices_exist = check_for_repeating_indices(all_grown_regions_indices)
print("Repeating indices exist:", repeating_indices_exist)

