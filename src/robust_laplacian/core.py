import robust_laplacian_bindings as rlb

def mesh_laplacian(verts, faces, mollify_factor=1e-5, return_mass=True):

    ## Validate input

    ## Call the main algorithm from the bindings
    L, M = rlb.buildMeshLaplacian(verts, faces, mollify_factor)

    ## Return the result
    if(return_mass):
        return L, M
    else:
        return L
