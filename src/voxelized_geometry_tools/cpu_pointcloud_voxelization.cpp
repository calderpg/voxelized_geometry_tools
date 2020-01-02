#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>

#include <omp.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
CollisionMap CpuPointCloudVoxelizer::VoxelizePointClouds(
    const CollisionMap& static_environment, const double step_size_multiplier,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointCloudWrapperPtr>& pointclouds) const
{
  if (!static_environment.IsInitialized())
  {
    throw std::invalid_argument("!static_environment.IsInitialized()");
  }
  if (step_size_multiplier > 1.0 || step_size_multiplier <= 0.0)
  {
    throw std::invalid_argument("step_size_multiplier is not in (0, 1]");
  }
  const std::chrono::time_point<std::chrono::steady_clock> start_time =
      std::chrono::steady_clock::now();
  // Pose of grid G in world W.
  const Eigen::Isometry3d& X_WG = static_environment.GetOriginTransform();
  // Get grid resolution and size parameters.
  const auto& grid_size = static_environment.GetGridSizes();
  const double cell_size = static_environment.GetResolution();
  const double step_size = cell_size * step_size_multiplier;
  // For each cloud, raycast it into its own "tracking grid"
  VectorCpuVoxelizationTrackingGrid tracking_grids(
          pointclouds.size(),
          CpuVoxelizationTrackingGrid(
              X_WG, grid_size, CpuVoxelizationTrackingCell()));
  for (size_t idx = 0; idx < pointclouds.size(); idx++)
  {
    const PointCloudWrapperPtr& cloud_ptr = pointclouds.at(idx);
    if (cloud_ptr)
    {
      CpuVoxelizationTrackingGrid& tracking_grid = tracking_grids.at(idx);
      RaycastPointCloud(*cloud_ptr, step_size, tracking_grid);
    }
    else
    {
      throw std::runtime_error("PointCloudWrapperPtr is null");
    }
  }
  const std::chrono::time_point<std::chrono::steady_clock> raycasted_time =
      std::chrono::steady_clock::now();
  // Combine & filter
  const auto result = CombineAndFilterGrids(
      static_environment, filter_options, tracking_grids);
  const std::chrono::time_point<std::chrono::steady_clock> done_time =
      std::chrono::steady_clock::now();
  std::cout
      << "Raycasting time "
      << std::chrono::duration<double>(raycasted_time - start_time).count()
      << ", filtering time "
      << std::chrono::duration<double>(done_time - raycasted_time).count()
      << std::endl;
  return result;
}

void CpuPointCloudVoxelizer::RaycastPointCloud(
    const PointCloudWrapper& cloud, const double step_size,
    CpuVoxelizationTrackingGrid& tracking_grid) const
{
  // Get X_WC, the transform from world to the origin of the pointcloud
  const Eigen::Isometry3d& X_WC = cloud.GetPointCloudOriginTransform();
  // Get the origin of X_WC
  const Eigen::Vector4d p_WCo = X_WC * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
#pragma omp parallel for
  for (int64_t idx = 0; idx < cloud.Size(); idx++)
  {
    // Location of point P in frame of camera C
    const Eigen::Vector4d p_CP = cloud.GetPointLocationDouble(idx);
    // Location of point P in world W
    const Eigen::Vector4d p_WP = X_WC * p_CP;
    // Ray from camera to point
    const Eigen::Vector4d ray = p_WP - p_WCo;
    const double ray_length = ray.norm();
    // Step along ray
    const int32_t num_steps =
        std::max(1, static_cast<int32_t>(std::floor(ray_length / step_size)));
    common_robotics_utilities::voxel_grid::GridIndex last_index(-1, -1, -1);
    for (int32_t step = 0; step < num_steps; step++)
    {
      bool in_grid = false;
      const double ratio =
          static_cast<double>(step) / static_cast<double>(num_steps);
      const Eigen::Vector4d p_WQ = p_WCo + (ray * ratio);
      const common_robotics_utilities::voxel_grid::GridIndex index =
          tracking_grid.LocationToGridIndex4d(p_WQ);
      // We don't want to double count in the same cell multiple times
      if (!(index == last_index))
      {
        auto query = tracking_grid.GetMutable(index);
        // We must check query.second to see if the query is within bounds.
        if (query)
        {
          in_grid = true;
          query.Value().seen_free_count.fetch_add(1);
        }
        else
        {
          if (in_grid)
          {
            // We've left the grid and there's no reason to keep going.
            break;
          }
        }
      }
      last_index = index;
    }
    // Set the point itself as filled
    const common_robotics_utilities::voxel_grid::GridIndex index =
          tracking_grid.LocationToGridIndex4d(p_WP);
    auto query = tracking_grid.GetMutable(index);
    // We must check query.second to see if the query is within bounds.
    if (query)
    {
      query.Value().seen_filled_count.fetch_add(1);
    }
  }
}

voxelized_geometry_tools::CollisionMap
CpuPointCloudVoxelizer::CombineAndFilterGrids(
    const CollisionMap& static_environment,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const VectorCpuVoxelizationTrackingGrid& tracking_grids) const
{
  CollisionMap filtered_grid = static_environment;
  // Because we want to improve performance and don't need to know where in the
  // grid we are, we can take advantage of the dense backing vector to iterate
  // through the grid data, rather than the grid cells.
  auto& filtered_grid_backing_store = filtered_grid.GetMutableRawData();
#pragma omp parallel for
  for (size_t voxel = 0; voxel < filtered_grid_backing_store.size(); voxel++)
  {
    voxelized_geometry_tools::CollisionCell& current_cell =
        filtered_grid_backing_store.at(voxel);
    // Filled cells stay filled, we don't work with them.
    // We only change cells that are unknown or empty.
    if (current_cell.Occupancy() <= 0.5)
    {
      int32_t seen_filled = 0;
      int32_t seen_free = 0;
      for (size_t idx = 0; idx < tracking_grids.size(); idx++)
      {
        const CpuVoxelizationTrackingCell& grid_cell =
            tracking_grids.at(idx).GetImmutableRawData().at(voxel);
        const int32_t free_count = grid_cell.seen_free_count.load();
        const int32_t filled_count = grid_cell.seen_filled_count.load();
        const SeenAs seen_as =
            filter_options.CountsSeenAs(free_count, filled_count);
        if (seen_as == SeenAs::FREE)
        {
          seen_free += 1;
        }
        else if (seen_as == SeenAs::FILLED)
        {
          seen_filled += 1;
        }
      }
      if (seen_filled > 0)
      {
        // If any camera saw something here, it is filled.
        current_cell.Occupancy() = 1.0;
      }
      else if (seen_free >= filter_options.NumCamerasSeenFree())
      {
        // Did enough cameras see this empty?
        current_cell.Occupancy() = 0.0;
      }
      else
      {
        // Otherwise, it is unknown.
        current_cell.Occupancy() = 0.5;
      }
    }
  }
  return filtered_grid;
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
