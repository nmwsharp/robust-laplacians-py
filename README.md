[![actions status linux](https://github.com/nmwsharp/robust-laplacians-py/workflows/Test%20Linux/badge.svg)](https://github.com/nmwsharp/robust-laplacians-py/actions)
[![actions status macOS](https://github.com/nmwsharp/robust-laplacians-py/workflows/Test%20macOS/badge.svg)](https://github.com/nmwsharp/robust-laplacians-py/actions)
[![actions status windows](https://github.com/nmwsharp/robust-laplacians-py/workflows/Test%20Windows/badge.svg)](https://github.com/nmwsharp/robust-laplacians-py/actions)
[![PyPI](https://img.shields.io/pypi/v/robust-laplacian?style=plastic)](https://pypi.org/project/robust-laplacian/)

A Python package for high-quality Laplace matrices on meshes and point clouds. `pip install robust_laplacian`

The Laplacian is at the heart of many algorithms across geometry processing, simulation, and machine learning. This library builds a high-quality, robust Laplace matrix which often improves the performance of these algorithms, and wraps it all up in a simple, single-function API! 

**Sample**: computing eigenvectors of the point cloud Laplacian
![demo image of eigenvectors on point cloud](https://github.com/nmwsharp/robust-laplacians-py/blob/master/teaser_cloud.jpg?raw=true)

Given as input a triangle mesh with arbitrary connectivity (could be nonmanifold, have boundary, etc), OR a point cloud, this library builds an `NxN` sparse Laplace matrix, where `N` is the number of vertices/points. This Laplace matrix is similar to the _cotan-Laplacian_ used widely in geometric computing, but internally the algorithm constructs an _intrinsic Delaunay triangulation_ of the surface, which gives the Laplace matrix great numerical properties. The resulting Laplacian is always a symmetric positive-definite matrix, with all positive edge weights. Additionally, this library performs _intrinsic mollification_ to alleviate floating-point issues with degenerate triangles.  

The resulting Laplace matrix `L` is a "weak" Laplace matrix, so we also generate a diagonal lumped mass matrix `M`, where each diagonal entry holds an area associated with the mesh element. The "strong" Laplacian can then be formed as `M^-1 L`, or a Poisson problem could be solved as `L x = M y`. 

A [C++ implementation and demo](https://github.com/nmwsharp/nonmanifold-laplacian) is available.

This library implements the algorithm described in [A Laplacian for Nonmanifold Triangle Meshes](http://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/NonmanifoldLaplace.pdf) by [Nicholas Sharp](http://nmwsharp.com) and [Keenan Crane](http://keenan.is/here) at SGP 2020 (where it won a best paper award!). See the paper for more details, and please use the citation given at the bottom if it contributes to academic work.

### Example

Build a point cloud Laplacian, compute its first 10 eigenvectors, and visualize with [Polyscope](https://polyscope.run/py/)

```shell
pip install numpy scipy plyfile polyscope robust_laplacian
```

```py
import robust_laplacian
from plyfile import PlyData
import numpy as np
import polyscope as ps
import scipy.sparse.linalg as sla

# Read input
plydata = PlyData.read("/path/to/cloud.ply")
points = np.vstack((
    plydata['vertex']['x'],
    plydata['vertex']['y'],
    plydata['vertex']['z']
)).T

# Build point cloud Laplacian
L, M = robust_laplacian.point_cloud_laplacian(points)

# (or for a mesh)
# L, M = robust_laplacian.mesh_laplacian(verts, faces)

# Compute some eigenvectors
n_eig = 10
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

# Visualize
ps.init()
ps_cloud = ps.register_point_cloud("my cloud", points)
for i in range(n_eig):
    ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)
ps.show()
```

**_NOTE:_** No one can agree on the sign convention for the Laplacian. This library builds the _positive semi-definite_ Laplace matrix, where the diagonal entries are positive and off-diagonal entries are negative. This is the _opposite_ of the sign used by e.g. libIGL in `igl.cotmatrix`, so you may need to flip a sign when converting code.

### API

This package exposes just two functions:

- `mesh_laplacian(verts, faces, mollify_factor=1e-5)`
  - `verts` is an `V x 3` numpy array of vertex positions
  - `faces`  is an `F x 3` numpy array of face indices, where each is a 0-based index referring to a vertex
  - `mollify_factor` amount of intrinsic mollifcation to perform. `0` disables, larger values will increase numerical stability, while very large values will slightly implicitly smooth out the geometry. The range of reasonable settings is roughly `0` to `1e-3`.  The default value should usually be sufficient.
  - `return L, M` a pair of scipy sparse matrices for the Laplacian `L` and mass matrix `M` 
- `point_cloud_laplacian(points, mollify_factor=1e-5, n_neighbors=30)` 
  - `points` is an `V x 3` numpy array of point positions
  - `mollify_factor` amount of intrinsic mollifcation to perform. `0` disables, larger values will increase numerical stability, while very large values will slightly implicitly smooth out the geometry. The range of reasonable settings is roughly `0` to `1e-3`.  The default value should usually be sufficient.
  - `n_neighbors` is the number of nearest neighbors to use when constructing local triangulations. This parameter has little effect on the resulting matrices, and the default value is almost always sufficient.
  - `return L, M` a pair of scipy sparse matrices for the Laplacian `L` and mass matrix `M` 

### Installation

The package is availabe via `pip`

```
pip install robust_laplacian
```

The underlying algorithm is implemented in C++; the pypi entry includes precompiled binaries for many platforms.

Very old versions of `pip` might need to be upgraded like `pip install pip --upgrade` to use the precompiled binaries.

Alternately, if no precompiled binary matches your system `pip` will attempt to compile from source on your machine.  This requires a working C++ toolchain, including cmake.

### Known limitations

- For point clouds, this repo uses a simple method to generate planar Delaunay triangulations, which may not be totally robust to collinear or degenerate point clouds.

### Dependencies

This python library is mainly a wrapper around the implementation in the [geometry-central](http://geometry-central.net) library; see there for further dependencies. Additionally, this library uses [pybind11](https://github.com/pybind/pybind11) to generate bindings, and [jc_voronoi](https://github.com/JCash/voronoi) for 2D Delaunay triangulation on point clouds. All are permissively licensed.

### Citation

```
@article{Sharp:2020:LNT,
  author={Nicholas Sharp and Keenan Crane},
  title={{A Laplacian for Nonmanifold Triangle Meshes}},
  journal={Computer Graphics Forum (SGP)},
  volume={39},
  number={5},
  year={2020}
}
```

### For developers

This repo is configured with CI on github actions to build wheels across platform.

### Deploy a new version

- Commit the desired updates to the `master` branch (or via PR). Include the string `[ci build]` in the commit message to ensure a build happens.
- Watch the github actions builds to ensure the test & build stages succeed and all wheels are compiled.
- While you're waiting, update the docs.
- Create a commit bumping the version in `pyproject.toml`. Include the string `[ci publish]` in the commit message and push.  This will kick off a new github actions build which deploys the wheels to PyPI after compilation. Use the github UI to create a new release + tag matching the version in `pyproject.toml`.
