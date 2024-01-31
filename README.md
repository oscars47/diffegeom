# diffegeom
Larry Liu's code for Prof. Gu's research.

# Updates:

## 1/30/24
- added ```plot_stl_sparse_random``` to sample random perturbed data around a mesh

## 1/26/24
- extending subdivide.py's divide_and_subdivide_sections to general numpy arrays of observations (just np arrays)
    * specifically, renamed main_subdivision() -> main_subdivision_stl() and main_subdivision_ar()
    * created new find_bounding_box_ar and transform_to_cube_ar() and divide_and_subdivide_sections()
    * subdivide_section logic remains the same since it only cares about the r,phi,theta points inside the different regions
    * added `convert_stl_to_ar` to convert stl mesh file into numpy array so we can do stuff (e.g., add noise or whatever) and see how that affects the fitting. 
- whole point is now we can just use numpy arrays to be agnositc about the source of the data and make the fitting algorithm general
- in ar_test.py, tested in 8 cases: with 1000 random points around a sphere, 1000 points around a sphere w noise (defined as temperature), 1000 points around torus, 1000 points around torus with noise. Noise values are 0.1, 0.01. noise is defined by sampling from a random distribution from 0 up to the temperature and adding to each point
- minor thing, can now pass a name for the project so the plots can be saved as pngs (if it is None, then won't save). automatically saves figures in a folder called `figures` (will automatically create)


# 1/20/24
- subdivide.py now correctly implements the recursive slicing based on polar coordinates for mesh file
- the description of the fuction `main_subdivision`: 
    * `read_stl` loads the mesh
    * `find_bounding box` draws a rectangular prism around the mesh
    * `find_center` computes the center of the box
    * `transform_to_cube` performs linear transformation to make box uniform
    * recall `bounding_box` and `find_center` to get the new bounding box box and center
    * `divide_and_subdivide_sections` performs the slicing first based on an initial array of phi, theta. in particular, we go through each triangle in the mesh, each point in the triangle, compute the r, phi, theta vals, and see which of the theta, phi bins they fall into. if `plot_intermediate` is called, will output what the r, phi, theta looks like. then, calls subdivide_section to recursively divide each of the main sections into smaller ones by finding the std deviation of the radii from the center, which we want to be <= a certain threshold (by default 1e-3)
    * `plot_subdivisions` to see the results in phi, theta space colored by radius, as well as a 3D plot
