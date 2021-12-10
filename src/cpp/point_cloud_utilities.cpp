#include "point_cloud_utilities.h"

#include "geometrycentral/utilities/elementary_geometry.h"
#include "geometrycentral/utilities/knn.h"

#include "Eigen/Dense"

#include <cfloat>
#include <numeric>

std::vector<std::vector<size_t>> generate_knn(const std::vector<Vector3>& points, size_t k) {

  geometrycentral::NearestNeighborFinder finder(points);

  std::vector<std::vector<size_t>> result;
  for (size_t i = 0; i < points.size(); i++) {
    result.emplace_back(finder.kNearestNeighbors(i, k));
  }

  return result;
}


std::vector<Vector3> generate_normals(const std::vector<Vector3>& points, const Neighbors_t& neigh) {

  std::vector<Vector3> normals(points.size());

  for (size_t iPt = 0; iPt < points.size(); iPt++) {
    size_t nNeigh = neigh[iPt].size();

    // Compute centroid
    Vector3 center{0., 0., 0.};
    for (size_t iN = 0; iN < nNeigh; iN++) {
      center += points[neigh[iPt][iN]];
    }
    center /= nNeigh + 1;

    // Assemble matrix os vectors from centroid
    Eigen::MatrixXd localMat(3, neigh[iPt].size());
    for (size_t iN = 0; iN < nNeigh; iN++) {
      Vector3 neighPos = points[neigh[iPt][iN]] - center;
      localMat(0, iN) = neighPos.x;
      localMat(1, iN) = neighPos.y;
      localMat(2, iN) = neighPos.z;
    }

    // Smallest singular vector is best normal
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(localMat, Eigen::ComputeThinU);
    Eigen::Vector3d bestNormal = svd.matrixU().col(2);

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

// For each planar-projected neighborhood, generate the triangles in the Delaunay triangulation which are incident on
// the center vertex.
//
// This could be done robustly via e.g. Shewchuk's triangle.c. However, instead we use a simple self-contained strategy
// which leverages the needs of this particular situation. In particular, we don't really care about getting exactly the
// Delaunay triangulation; we're just looking for any sane triangulation to use as input the the subsequent step. We
// just use Delaunay because we like the property that (in the limit of sampling), it's a triple-cover of the domain;
// with other strategies it's hard to quantify how many times our triangles cover the domain. This makes the problem
// easier, because for degenerate/underdetermined cases, we're happy to output any triangulation, even if it's not the
// Delaunay triangulation in exact arithmetic.
//
// This strategy works by angularly sorting points relative to the neighborhood center, then walking around circle
// identifying pairs of edges which form Delaunay triangles (more details inline). In particular, using a sorting of the
// points helps to distinguish indeterminate cases and always output some triangles. Additionally, a few heuristics are
// included for handling of degenerate and collinear points. This routine has O(n*k^2) complexity, where k is the
// neighborhood size).
LocalTriangulationResult build_delaunay_triangulations(const std::vector<std::vector<Vector2>>& coords,
                                                       const Neighbors_t& neigh) {

  // A few innocent numerical parameters
  const double PERTURB_THRESH = 1e-7;         // in units of relative length
  const double ANGLE_COLLINEAR_THRESH = 1e-5; // in units of radians
  const double OUTSIDE_EPS = 1e-4;            // in units of relative length

  // NOTE: This is not robust if the entire neighbohood is coincident (or very nearly coincident) with the centerpoint.
  // Though in that case, the generate_normals() routine will probably also have issues.

  size_t nPts = coords.size();
  LocalTriangulationResult result;
  result.pointTriangles.resize(nPts);

  for (size_t iPt = 0; iPt < nPts; iPt++) {
    size_t nNeigh = neigh[iPt].size();
    double lenScale = norm(coords[iPt].back());

    // Something is hopelessly degenerate, don't even bother trying. No triangles for this point.
    if (!std::isfinite(lenScale) || lenScale <= 0) {
      continue;
    }

    // Local copies of points
    std::vector<Vector2> perturbPoints = coords[iPt];
    std::vector<size_t> perturbInds = neigh[iPt];

    { // Perturb points which are extremely close to the source
      for (size_t iNeigh = 0; iNeigh < nNeigh; iNeigh++) {
        Vector2& neighPt = perturbPoints[iNeigh];
        double dist = norm(neighPt);
        if (dist < lenScale * PERTURB_THRESH) { // need to perturb
          Vector2 dir = normalize(neighPt);
          if (!isfinite(dir)) { // even direction is degenerate :(
            // pick a direction from index
            double thetaDir = (2. * M_PI * iNeigh) / nNeigh;
            dir = Vector2::fromAngle(thetaDir);
          }

          // Set the distance from the origin for the pertubed point. Including the index avoids creating many
          // co-circular points; no need to stress the Delaunay triangulation unnessecarily.
          double len = (1. + static_cast<double>(iNeigh) / nNeigh) * lenScale * PERTURB_THRESH * 10;

          neighPt = len * dir; // update the point
        }
      }
    }


    size_t closestPointInd = 0;
    double closestPointDist = std::numeric_limits<double>::infinity();
    bool hasBoundary = false;
    { // Find the starting point for the angular search.
      // If there is boundary, it's the beginning of the interior region; otherwise its the closest point.
      // (either way, this point is guaranteed to appear in the triangulation)
      // NOTE: boundary check is actually done after inline sort below, since its cheaper there

      for (size_t iNeigh = 0; iNeigh < nNeigh; iNeigh++) {
        Vector2 neighPt = perturbPoints[iNeigh];
        double thisPointDist = norm(neighPt);
        if (thisPointDist < closestPointDist) {
          closestPointDist = thisPointDist;
          closestPointInd = iNeigh;
        }
      }
    }


    std::vector<size_t> sortInds(nNeigh);
    { // = Angularly sort the points CCW, such that the closest point comes first

      // Angular sort
      std::vector<double> pointAngles(nNeigh);
      for (size_t i = 0; i < nNeigh; i++) {
        pointAngles[i] = arg(perturbPoints[i]);
      }
      std::iota(std::begin(sortInds), std::end(sortInds), 0);
      std::sort(sortInds.begin(), sortInds.end(),
                [&](const size_t& a, const size_t& b) -> bool { return pointAngles[a] < pointAngles[b]; });

      // Check if theres a gap of >= PI between any two consecutive points. If so it's a boundary.
      double largestGap = -1;
      size_t largestGapEndInd = 0;
      for (size_t i = 0; i < nNeigh; i++) {
        size_t j = (i + 1) % nNeigh;
        double angleI = pointAngles[sortInds[i]];
        double angleJ = pointAngles[sortInds[j]];
        double gap;
        if (i + 1 == nNeigh) {
          gap = angleJ - (angleI + 2 * M_PI);
        } else {
          gap = angleJ - angleI;
        }

        if (gap > largestGap) {
          largestGap = gap;
          largestGapEndInd = j;
        }
      }

      // The start of the cyclic ordering is either
      size_t firstInd;
      if (largestGap > (M_PI - ANGLE_COLLINEAR_THRESH)) {
        firstInd = largestGapEndInd;
        hasBoundary = true;
      } else {
        firstInd = std::distance(sortInds.begin(), std::find(sortInds.begin(), sortInds.end(), closestPointInd));
        hasBoundary = false;
      }

      // Cyclically permute to ensure starting point comes first
      std::rotate(sortInds.begin(), sortInds.begin() + firstInd, sortInds.end());
    }

    size_t edgeStartInd = 0;
    std::vector<std::array<size_t, 3>>& thisPointTriangles = result.pointTriangles[iPt]; // accumulate result

    // end point should wrap around the check the first point only if there is no boundary
    size_t searchEnd = nNeigh + (hasBoundary ? 0 : 1);

    // Walk around the angularly-sorted points, forming triangles spanning angular regions. To construct each triangle,
    // we start with leg at edgeStartInd, then search over edgeEndInd to find the first other end which has an empty
    // circumcircle. Once it is found, we form a triangle and being searching again from edgeEndInd.
    //
    // At first, this might sound like it has n^3 complexity, since there are n^2 triangles to consider, and testing
    // each costs n. However, since we march around the angular direction in increasing order, we will only test at most
    // O(n) triangles, leading to n^2 complexity.
    while (edgeStartInd < nNeigh) {
      size_t iStart = sortInds[edgeStartInd];
      Vector2 startPos = perturbPoints[iStart];

      // lookahead and find the first triangle we can form with an empty (or nearly empty) circumcircle
      bool foundTri = false;
      for (size_t edgeEndInd = edgeStartInd + 1; edgeEndInd < searchEnd; edgeEndInd++) {
        size_t iEnd = sortInds[edgeEndInd % nNeigh];
        Vector2 endPos = perturbPoints[iEnd];

        // If the start and end points are too close to being colinear, don't bother
        Vector2 startPosDir = unit(startPos);
        Vector2 endPosDir = unit(endPos);
        if (std::fabs(cross(startPosDir, endPosDir)) < ANGLE_COLLINEAR_THRESH) {
          continue;
        }

        // Find the circumcenter and circumradius
        geometrycentral::RayRayIntersectionResult2D isect =
            rayRayIntersection(0.5 * startPos, startPosDir.rotate90(), 0.5 * endPos, -endPosDir.rotate90());
        Vector2 circumcenter = 0.5 * startPos + isect.tRay1 * startPosDir.rotate90();
        double circumradius = norm(circumcenter);

        // Find the minimum distance to the circumcenter
        double nearestDistSq = std::numeric_limits<double>::infinity();
        double circumradSqConservative = (circumradius - lenScale * OUTSIDE_EPS);
        circumradSqConservative *= circumradSqConservative;
        for (size_t iTest = 0; iTest < nNeigh; iTest++) {
          if (iTest == iStart || iTest == iEnd) continue; // skip the points forming the triangle
          double thisDistSq = norm2(circumcenter - perturbPoints[iTest]);
          nearestDistSq = std::fmin(nearestDistSq, thisDistSq);

          // if it's already strictly inside, no need to keep searching
          if (nearestDistSq < circumradSqConservative) break;
        }
        double nearestDist = std::sqrt(nearestDistSq);

        // Accept the triangle if its circumcircle is sufficiently empty
        // NOTE: The choice of signs in this expression is important: we preferential DO accept triangles whos
        // circumcircle is barely empty. This makes sense here because our circular loop already avoids any risk of
        // accepting overlapping triangles; the risk is in not accepting any, so we should preferrentially accept.
        if (nearestDist + lenScale * OUTSIDE_EPS > circumradius) {
          std::array<size_t, 3> triInds = {std::numeric_limits<size_t>::max(), iStart, iEnd};
          thisPointTriangles.push_back(triInds);

          // advance the circular search to find a triangle starting at this edge
          edgeStartInd = edgeEndInd;
          foundTri = true;
          break;
        }
      }

      // if we couldn't find any triangles, increment the start index
      if (!foundTri) {
        edgeStartInd++;
      }
    }
  }

  return result;
}
