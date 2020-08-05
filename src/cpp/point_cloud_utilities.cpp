#include "point_cloud_utilities.h"

#include "geometrycentral/utilities/knn.h"

#include "Eigen/Dense"

#include <cfloat>

// jcv Voronoi library
#define JC_VORONOI_IMPLEMENTATION
#define JCV_REAL_TYPE double
#define JCV_ATAN2 atan2
#define JCV_SQRT sqrt
#define JCV_FLT_MAX DBL_MAX
#define JCV_PI 3.141592653589793115997963468544185161590576171875
#include "jc_voronoi/jc_voronoi.h"

std::vector<std::vector<size_t>> generate_knn(const std::vector<Vector3>& points, size_t k) {

  geometrycentral::NearestNeighborFinder finder(points);

  std::vector<std::vector<size_t>> result;
  for (size_t i = 0; i < points.size(); i++) {
    result.emplace_back(finder.kNearestNeighbors(i, k));
    result.back().insert(result.back().begin(), i); // add the center point to the front
  }

  return result;
}


std::vector<Vector3> generate_normals(const std::vector<Vector3>& points, const Neighbors_t& neigh) {

  using namespace Eigen;

  std::vector<Vector3> normals(points.size());

  for (size_t iPt = 0; iPt < points.size(); iPt++) {
    size_t nNeigh = neigh[iPt].size();
    Vector3 center = points[iPt];
    MatrixXd localMat(3, neigh[iPt].size() - 1);

    for (size_t iN = 1; iN < nNeigh; iN++) {
      Vector3 neighPos = points[neigh[iPt][iN]] - center;
      localMat(0, iN - 1) = neighPos.x;
      localMat(1, iN - 1) = neighPos.y;
      localMat(2, iN - 1) = neighPos.z;
    }

    // Smallest singular vector is best normal
    JacobiSVD<MatrixXd> svd(localMat, ComputeThinU);
    Vector3d bestNormal = svd.matrixU().col(2);

    Vector3 N{bestNormal(0), bestNormal(1), bestNormal(2)};
    N = unit(N);
    normals[iPt] = N;
  }

  return normals;
}


std::vector<std::vector<Vector2>> generate_coords_projection(const std::vector<Vector3>& points,
                                                             const std::vector<Vector3> normals,
                                                             const Neighbors_t& neigh) {
  std::vector<std::vector<Vector2>> coords(points.size());

  for (size_t iPt = 0; iPt < points.size(); iPt++) {
    size_t nNeigh = neigh[iPt].size();
    coords[iPt].resize(nNeigh);
    Vector3 center = points[iPt];
    Vector3 normal = normals[iPt];

    // build an arbitrary tangent basis
    Vector3 basisX, basisY;
    auto r = normal.buildTangentBasis();
    basisX = r[0];
    basisY = r[1];

    for (size_t iN = 0; iN < nNeigh; iN++) {
      Vector3 vec = points[neigh[iPt][iN]] - center;
      vec = vec.removeComponent(normal);

      Vector2 coord{dot(basisX, vec), dot(basisY, vec)};
      coords[iPt][iN] = coord;
    }
  }

  return coords;
}


LocalTriangulationResult build_delaunay_triangulations(const std::vector<std::vector<Vector2>>& coords,
                                                       const Neighbors_t& neigh, bool generateAllTris) {
  size_t nPts = coords.size();
  LocalTriangulationResult result;
  result.voronoiAreas.resize(nPts);
  result.pointTriangles.resize(nPts);


  for (size_t iPt = 0; iPt < nPts; iPt++) {
    size_t nNeigh = neigh[iPt].size();
    //std::cout << "\nPoint has " << nNeigh << " neighbors" << std::endl;

    // Copy neighbor coords to a raw buffer
    std::vector<jcv_point> rawCoords;
    double lenScale = norm(coords[iPt].back());
    for (size_t iN = 0; iN < nNeigh; iN++) {
      Vector2 p = coords[iPt][iN];

      // If there is a point other than the center point right on top of the origin, perturb it slightly. jcv doesn't do
      // great with duplicate points, and we can't recover if something happens near the origin.
      if (iN != 0 && norm(p) < 1e-6 * lenScale) {
        p.x += 1e-6 * lenScale;
      }

      rawCoords.push_back({p.x, p.y});

      //std::cout << "  p = " << p << std::endl;
    }

    // run the Voronoi algorithm
    jcv_diagram diagram;
    memset(&diagram, 0, sizeof(jcv_diagram));
    jcv_diagram_generate(nNeigh, &rawCoords[0], 0, 0, &diagram);

    // find the site at the center vertex (is this predictable?)
    const jcv_site* centerSite = nullptr;
    {
      const jcv_site* sites = jcv_diagram_get_sites(&diagram);
      for (int i = 0; i < diagram.numsites; i++) {
        const jcv_site* site = &sites[i];
        if (static_cast<size_t>(site->index) == 0) {
          centerSite = site;
          break;
        }
      }
      if (centerSite == nullptr) throw std::runtime_error("could not find site for center vertex");
    }

    // == Get the area of the center cell
    double voronoiArea = 0;
    {
      const jcv_graphedge* e = centerSite->edges;
      while (e) {
        Vector2 p0{centerSite->p.x, centerSite->p.y};
        Vector2 p1{e->pos[0].x, e->pos[0].y};
        Vector2 p2{e->pos[1].x, e->pos[1].y};

        // (here triangle is not a Delaunay triangle, but part of an implicit fan triangulation from the center)
        double triArea = 0.5 * std::abs(cross(p1 - p0, p2 - p0));
        voronoiArea += triArea;

        e = e->next;
      }
    }
    result.voronoiAreas[iPt] = voronoiArea;

    // == Find all triangles
    if (generateAllTris) {
      result.allTriangles.resize(nPts);

      // Build a list of all edges
      const jcv_edge* edge = jcv_diagram_get_edges(&diagram);
      std::vector<std::vector<size_t>> localEdges(nNeigh);
      while (edge) {
        if (edge->sites[0] && edge->sites[1]) {
          size_t indA = edge->sites[0]->index;
          size_t indB = edge->sites[1]->index;

          localEdges[indA].push_back(indB);
          localEdges[indB].push_back(indA);
        }
        edge = jcv_diagram_get_next_edge(edge);
      }

      // Find triplets in the edge list
      // (this has an O(d) factor that could be removed with a hashset, but since degree will be small this is probably
      // faster)

      for (size_t iA = 0; iA < nNeigh; iA++) {
        for (size_t iB : localEdges[iA]) {
          if (!(iA < iB)) continue; // only consider if iA < iB < iC

          for (size_t iC : localEdges[iA]) {
            if (!(iB < iC)) continue; // only consider if iA < iB < iC

            // Check if iB is connected to iC
            std::vector<size_t>& bNeigh = localEdges[iB];
            if (std::find(bNeigh.begin(), bNeigh.end(), iC) == bNeigh.end()) continue;
            // This is a good triangle!

            size_t triA = iA;
            size_t triB = iB;
            size_t triC = iC;

            // Swap to orient if needed
            Vector2 pA = coords[iPt][triA];
            Vector2 pB = coords[iPt][triB];
            Vector2 pC = coords[iPt][triC];
            if (cross(pB - pA, pC - pA) < 0.) std::swap(triB, triC);

            std::array<size_t, 3> triInds = {triA, triB, triC};
            result.allTriangles[iPt].push_back(triInds);
          }
        }
      }
    }

    // == Find triangles connected to the source
    {

      // First pass, mark all neighbors with an edge to the source
      std::vector<char> neighConnected(nNeigh, false);
      const jcv_graphedge* e = centerSite->edges;
      while (e) {
        if (e->neighbor) {
          size_t neighInd = e->neighbor->index;
          neighConnected[neighInd] = true;
        }
        e = e->next;
      }

      // Second pass, any edge between two connected points completes a triangle
      const jcv_edge* edge = jcv_diagram_get_edges(&diagram);
      while (edge) {
        if (edge->sites[0] && edge->sites[1]) {
          size_t indA = edge->sites[0]->index;
          size_t indB = edge->sites[1]->index;

          if (neighConnected[indA] && neighConnected[indB]) {
            // found a triangle!

            // check orientation to make sure we emit a CCW triangle
            Vector2 pA = coords[iPt][indA];
            Vector2 pB = coords[iPt][indB];
            if (cross(pA, pB) < 0.) std::swap(indA, indB);

            std::array<size_t, 3> triInds = {0, indA, indB};
            result.pointTriangles[iPt].push_back(triInds);
          }
        }
        edge = jcv_diagram_get_next_edge(edge);
      }
    }


    jcv_diagram_free(&diagram);
  }

  return result;
}



