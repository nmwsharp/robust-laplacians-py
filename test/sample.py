import os, sys

import polyscope as ps
import numpy as np
import scipy.sparse.linalg as sla
from plyfile import PlyData

# Path to where the bindings live
sys.path.append(os.path.join(os.path.dirname(__file__), "../build/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

import robust_laplacian

# Read input
plydata = PlyData.read("/path/to/cloud.ply")
points = np.vstack((
    plydata['vertex']['x'],
    plydata['vertex']['y'],
    plydata['vertex']['z']
)).T

# for meshes
# tri_data = plydata['face'].data['vertex_indices']
# faces = np.vstack(tri_data)

# Build Laplacian
L, M = robust_laplacian.point_cloud_laplacian(points, mollify_factor=1e-5)

# for meshes
# L, M = robust_laplacian.mesh_laplacian(points, faces, mollify_factor=1e-5)

# Compute some eigenvectors
n_eig = 10
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

# Visualize
ps.init()
ps_cloud = ps.register_point_cloud("my cloud", points)
for i in range(n_eig):
    ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)

# for meshes
# ps_surf = ps.register_surface_mesh("my surf", points, faces)
# for i in range(n_eig):
    # ps_surf.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)

ps.show()
