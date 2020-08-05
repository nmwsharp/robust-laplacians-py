import os, sys

import meshio
import polyscope as ps
import numpy as np
import scipy.sparse.linalg as sla

# Path to where the bindings live
sys.path.append(os.path.join(os.path.dirname(__file__), "../build/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

import robust_laplacian

# Read input
points = meshio.read("/path/to/mesh.ply").points

# Build Laplacian
L, M = robust_laplacian.point_cloud_laplacian(points, mollify_factor=1e-5)

# Compute some eigenvectors
n_eig = 10
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

# Visualize
ps.init()
ps_cloud = ps.register_point_cloud("my cloud", points)
for i in range(n_eig):
    ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)
ps.show()
