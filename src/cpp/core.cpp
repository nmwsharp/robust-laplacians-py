#include "point_cloud_utilities.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
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

// Parameters related to unused elements. Maybe expose these as parameters?
double laplacianReplaceVal = 1.0;
double massReplaceVal = -1e-3;


std::tuple<SparseMatrix<double>, SparseMatrix<double>>
buildMeshLaplacian(const DenseMatrix<double>& vMat, const DenseMatrix<size_t>& fMat, double mollifyFactor) {

  // First, load a simple polygon mesh
  SimplePolygonMesh simpleMesh;

  // Copy to std vector representation
  simpleMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < simpleMesh.vertexCoordinates.size(); iP++) {
    simpleMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  simpleMesh.polygons.resize(fMat.rows());
  for (size_t iF = 0; iF < simpleMesh.polygons.size(); iF++) {
    simpleMesh.polygons[iF] = std::vector<size_t>{fMat(iF, 0), fMat(iF, 1), fMat(iF, 2)};
  }

  // Remove any unused vertices
  std::vector<size_t> oldToNewMap = simpleMesh.stripUnusedVertices();


  // Build the rich mesh data structure
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  // Do the hard work, calling the geometry-central function
  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(simpleMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildPointCloudLaplacian(const DenseMatrix<double>& vMat,
                                                                                double mollifyFactor, size_t nNeigh) {

  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }

  // Generate the local triangulations for the point cloud
  Neighbors_t neigh = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L / 3.;
  M = M / 3.;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

// Actual binding code
// clang-format off
PYBIND11_MODULE(robust_laplacian_bindings, m) {
  m.doc() = "Robust laplacian low-level bindings";
  

  m.def("buildMeshLaplacian", &buildMeshLaplacian, "build the mesh Laplacian", 
      py::arg("vMat"), py::arg("fMat"), py::arg("mollifyFactor"));
  
  m.def("buildPointCloudLaplacian", &buildPointCloudLaplacian, "build the point cloud Laplacian", 
      py::arg("vMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
}

// clang-format on
