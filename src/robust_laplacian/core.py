import numpy as np

import robust_laplacian_bindings as rlb

def mesh_laplacian(verts, faces, mollify_factor=1e-5):

    ## Validate input
    if not isinstance(verts, np.ndarray):
        raise ValueError("`verts` should be an instance of numpy array")
    if (len(verts.shape) != 2) or (verts.shape[1] != 3):
        raise ValueError("`verts` should have shape (V,3), shape is " + str(verts.shape))
    
    if not isinstance(faces, np.ndarray):
        raise ValueError(f"`faces` should be an instance of numpy array")
    if (len(faces.shape) != 2) or (faces.shape[1] != 3):
        raise ValueError("`faces` should have shape (F,3), shape is " + str(faces.shape))

    ## Call the main algorithm from the bindings
    L, M = rlb.buildMeshLaplacian(verts, faces, mollify_factor)

    ## Return the result
    return L, M

def point_cloud_laplacian(points, mollify_factor=1e-5, n_neighbors=30):

    ## Validate input
    if type(points) is not np.ndarray:
        raise ValueError("`points` should be a numpy array")
    if (len(points.shape) != 2) or (points.shape[1] != 3):
        raise ValueError("`points` should have shape (V,3), shape is " + str(points.shape))
    
    ## Call the main algorithm from the bindings
    L, M = rlb.buildPointCloudLaplacian(points, mollify_factor, n_neighbors)

    ## Return the result
    return L, M
