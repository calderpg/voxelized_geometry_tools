#include <voxelized_geometry_tools/mesh_rasterizer.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/parallelism.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForLoop;
using common_robotics_utilities::parallelism::ThreadWorkRange;

namespace voxelized_geometry_tools
{
namespace
{
bool PointProjectsInsideTriangle(
    const Eigen::Vector3d& p_Mv1, const Eigen::Vector3d& p_Mv2,
    const Eigen::Vector3d& p_Mv3, const Eigen::Vector3d& p_MQ)
{
  const auto same_side = [] (
      const Eigen::Vector3d& p_MA, const Eigen::Vector3d& p_MB,
      const Eigen::Vector3d& p_Mp1, const Eigen::Vector3d& p_Mp2) -> bool
  {
    const Eigen::Vector3d v_AB = p_MB - p_MA;
    const Eigen::Vector3d cross1 = v_AB.cross(p_Mp1 - p_MA);
    const Eigen::Vector3d cross2 = v_AB.cross(p_Mp2 - p_MA);
    return (cross1.dot(cross2) >= 0.0);
  };

  return (same_side(p_Mv1, p_Mv2, p_Mv3, p_MQ) &&
          same_side(p_Mv2, p_Mv3, p_Mv1, p_MQ) &&
          same_side(p_Mv3, p_Mv1, p_Mv2, p_MQ));
}

Eigen::Vector3d ClosestPointOnLineSegment(
    const Eigen::Vector3d& p_MA, const Eigen::Vector3d& p_MB,
    const Eigen::Vector3d& p_MQ)
{
  const Eigen::Vector3d v_AB = p_MB - p_MA;
  const Eigen::Vector3d v_AQ = p_MQ - p_MA;

  const double ratio = v_AB.dot(v_AQ) / v_AB.squaredNorm();
  const double clamped =
      common_robotics_utilities::utility::ClampValue(ratio, 0.0, 1.0);

  return p_MA + (v_AB * clamped);
}

Eigen::Vector3d CalcClosestPointOnTriangle(
    const Eigen::Vector3d& p_Mv1, const Eigen::Vector3d& p_Mv2,
    const Eigen::Vector3d& p_Mv3, const Eigen::Vector3d& normal,
    const Eigen::Vector3d& p_MQ)
{
  Eigen::Vector3d p_MQclosest;

  if (PointProjectsInsideTriangle(p_Mv1, p_Mv2, p_Mv3, p_MQ))
  {
    const Eigen::Vector3d v_v1Q = p_MQ - p_Mv1;
    // Project query point to triangle plane.
    const Eigen::Vector3d p_MQprojected =
        p_Mv1 + common_robotics_utilities::math::VectorRejection(normal, v_v1Q);
    p_MQclosest = p_MQprojected;
  }
  else
  {
    const Eigen::Vector3d p_MQclosest12 =
        ClosestPointOnLineSegment(p_Mv1, p_Mv2, p_MQ);
    const Eigen::Vector3d p_MQclosest23 =
        ClosestPointOnLineSegment(p_Mv2, p_Mv3, p_MQ);
    const Eigen::Vector3d p_MQclosest31 =
        ClosestPointOnLineSegment(p_Mv3, p_Mv1, p_MQ);
    const double d_closest12_squared = p_MQclosest12.squaredNorm();
    const double d_closest23_squared = p_MQclosest23.squaredNorm();
    const double d_closest31_squared = p_MQclosest31.squaredNorm();
    if (d_closest12_squared <= d_closest23_squared &&
        d_closest12_squared <= d_closest31_squared)
    {
      p_MQclosest = p_MQclosest12;
    }
    else if (d_closest23_squared <= d_closest12_squared &&
             d_closest23_squared <= d_closest31_squared)
    {
      p_MQclosest = p_MQclosest23;
    }
    else
    {
      p_MQclosest = p_MQclosest31;
    }
  }

  return p_MQclosest;
}
}  // namespace

void RasterizeTriangle(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    const size_t triangle_index,
    voxelized_geometry_tools::CollisionMap& collision_map,
    const bool enforce_collision_map_contains_triangle)
{
  if (!collision_map.IsInitialized())
  {
    throw std::invalid_argument("collision_map must be initialized");
  }

  const double min_check_radius = collision_map.GetResolution() * 0.5;
  const double max_check_radius = min_check_radius * std::sqrt(3.0);
  const double max_check_radius_squared = std::pow(max_check_radius, 2.0);

  const Eigen::Vector3i& triangle = triangles.at(triangle_index);
  const Eigen::Vector3d& p_Mv1 = vertices.at(static_cast<size_t>(triangle(0)));
  const Eigen::Vector3d& p_Mv2 = vertices.at(static_cast<size_t>(triangle(1)));
  const Eigen::Vector3d& p_Mv3 = vertices.at(static_cast<size_t>(triangle(2)));

  const Eigen::Vector3d v1v2 = p_Mv2 - p_Mv1;
  const Eigen::Vector3d v1v3 = p_Mv3 - p_Mv1;

  // This assumes that the triangle is well-conditioned. Long and skinny
  // triangles may lead to problems.
  const Eigen::Vector3d normal = v1v2.cross(v1v3);

  const double x_min = std::min({p_Mv1.x(), p_Mv2.x(), p_Mv3.x()});
  const double y_min = std::min({p_Mv1.y(), p_Mv2.y(), p_Mv3.y()});
  const double z_min = std::min({p_Mv1.z(), p_Mv2.z(), p_Mv3.z()});

  const double x_max = std::max({p_Mv1.x(), p_Mv2.x(), p_Mv3.x()});
  const double y_max = std::max({p_Mv1.y(), p_Mv2.y(), p_Mv3.y()});
  const double z_max = std::max({p_Mv1.z(), p_Mv2.z(), p_Mv3.z()});

  const auto min_index =
      collision_map.LocationToGridIndex(x_min, y_min, z_min);
  const auto max_index =
      collision_map.LocationToGridIndex(x_max, y_max, z_max);

  for (int64_t x_index = min_index.X(); x_index <= max_index.X();
       x_index++)
  {
    for (int64_t y_index = min_index.Y(); y_index <= max_index.Y();
         y_index++)
    {
      for (int64_t z_index = min_index.Z(); z_index <= max_index.Z();
           z_index++)
      {
        const common_robotics_utilities::voxel_grid::GridIndex
            current_index(x_index, y_index, z_index);
        const Eigen::Vector3d p_MQ =
            collision_map.GridIndexToLocation(current_index).head<3>();

        const Eigen::Vector3d p_MQclosest =
            CalcClosestPointOnTriangle(p_Mv1, p_Mv2, p_Mv3, normal, p_MQ);

        const double distance_squared = (p_MQclosest - p_MQ).squaredNorm();

        /// TODO(calderpg) Improve this with more a accurate triangle-box check.
        /// This is a coarse approximation of true triangle-box collision.
        /// The right solution is something along the lines of:
        ///
        /// if distance_squared > max_check_radius_squared:
        ///   triangle_intersects = false
        /// else if distance_squared < min_check_radius_squared:
        ///   triangle_intersects = true
        /// else:
        ///   check_index = collision_map.LocationToGridIndex3d(p_MQclosest)
        ///   if check_index == current_index:
        ///     triangle_intersects = true;
        ///   else:
        ///     Fall back to a better triangle-box check
        ///
        /// For now the following approximation is enough.
        const bool triangle_intersects =
            (distance_squared <= max_check_radius_squared);

        if (triangle_intersects)
        {
          auto query = collision_map.GetIndexMutable(current_index);
          if (query)
          {
            query.Value().Occupancy() = 1.0f;
          }
          else if (enforce_collision_map_contains_triangle)
          {
            throw std::runtime_error(
                "Triangle is not contained by collision map");
          }
        }
      }
    }
  }
}

void RasterizeMesh(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    voxelized_geometry_tools::CollisionMap& collision_map,
    const bool enforce_collision_map_contains_mesh,
    const DegreeOfParallelism& parallelism)
{
  if (!collision_map.IsInitialized())
  {
    throw std::invalid_argument("collision_map must be initialized");
  }


  // Helper lambda for each thread's work
  const auto per_thread_work = [&](const ThreadWorkRange& work_range)
  {
    for (int64_t triangle_index = work_range.GetRangeStart();
         triangle_index < work_range.GetRangeEnd();
         triangle_index++)
    {
      RasterizeTriangle(
        vertices, triangles, static_cast<size_t>(triangle_index), collision_map,
        enforce_collision_map_contains_mesh);
    }
  };

  // Raycast all points in the pointcloud. Use OpenMP if available, if not fall
  // back to manual dispatch via std::async.
  StaticParallelForLoop(
      parallelism, 0, static_cast<int64_t>(triangles.size()), per_thread_work,
      ParallelForBackend::BEST_AVAILABLE);
}

voxelized_geometry_tools::CollisionMap RasterizeMeshIntoCollisionMap(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    const double resolution,
    const DegreeOfParallelism& parallelism)
{
  if (resolution <= 0.0)
  {
    throw std::invalid_argument("resolution must be greater than zero");
  }

  Eigen::Vector3d lower_corner =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d upper_corner =
      Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

  for (size_t idx = 0; idx < vertices.size(); idx++)
  {
    const Eigen::Vector3d& vertex = vertices.at(idx);
    lower_corner = lower_corner.cwiseMin(vertex);
    upper_corner = upper_corner.cwiseMax(vertex);
  }

  const Eigen::Vector3d object_size = upper_corner - lower_corner;

  const double buffer_size = resolution * 2.0;
  const common_robotics_utilities::voxel_grid::GridSizes filter_grid_sizes(
      resolution, object_size.x() + buffer_size, object_size.y() + buffer_size,
      object_size.z() + buffer_size);

  const Eigen::Isometry3d X_OG(Eigen::Translation3d(
      lower_corner.x() - resolution,
      lower_corner.y() - resolution,
      lower_corner.z() - resolution));

  const voxelized_geometry_tools::CollisionCell empty_cell(0.0f);
  voxelized_geometry_tools::CollisionMap collision_map(
      X_OG, "mesh", filter_grid_sizes, empty_cell);

  RasterizeMesh(vertices, triangles, collision_map, true, parallelism);

  return collision_map;
}
}  // namespace voxelized_geometry_tools

