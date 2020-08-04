#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"

namespace py = pybind11;

using namespace geometrycentral;
using namespace geometrycentral::surface;


// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

std::tuple<SparseMatrix<double>, SparseMatrix<double>>
buildMeshLaplacian(const DenseMatrix<double>& vMat, const DenseMatrix<size_t>& fMat, double mollifyFactor) {

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(vMat, fMat);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  double minVal = 999999;
  for (size_t i = 0; i < M.rows(); i++) {
    minVal = std::fmin(minVal, M.coeffRef(i, i));
  }
  std::cout << "min  M = " << minVal << std::endl;

  return std::make_tuple(L, M);
}

// Actual binding code
// clang-format off
PYBIND11_MODULE(robust_laplacian_bindings, m) {
  m.doc() = "Robust laplacian low-level bindings";
  
  m.def("buildMeshLaplacian", &buildMeshLaplacian, "build the mesh Laplacian", 
      py::arg("vMat"), py::arg("fMat"), py::arg("mollifyFactor"));
}

// clang-format on
