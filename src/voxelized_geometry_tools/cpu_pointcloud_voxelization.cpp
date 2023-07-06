#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/openmp_helpers.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/device_voxelization_interface.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace
{
/// Struct to store free/filled counts in a voxel grid. Counts are stored in
/// std::atomic<int32_t> since a given cell may be manipulated by multiple
/// threads simultaneously and this gives us atomic fetch_add() to increment the
/// stored values.
struct CpuVoxelizationTrackingCell
{
  std::atomic<int32_t> seen_free_count;
  std::atomic<int32_t> seen_filled_count;

  CpuVoxelizationTrackingCell()
  {
    seen_free_count.store(0);
    seen_filled_count.store(0);
  }

  CpuVoxelizationTrackingCell(
      const int32_t seen_free, const int32_t seen_filled)
  {
    seen_free_count.store(seen_free);
    seen_filled_count.store(seen_filled);
  }

  /// We need copy constructor since std::atomics do not have copy constructors.
  CpuVoxelizationTrackingCell(const CpuVoxelizationTrackingCell& other)
  {
    seen_free_count.store(other.seen_free_count.load());
    seen_filled_count.store(other.seen_filled_count.load());
  }

  /// We need assignment operator since std::atomics do not have it.
  CpuVoxelizationTrackingCell& operator =
      (const CpuVoxelizationTrackingCell& other)
  {
    if (this != &other)
    {
      this->seen_free_count.store(other.seen_free_count.load());
      this->seen_filled_count.store(other.seen_filled_count.load());
    }
    return *this;
  }
};

using CpuVoxelizationTrackingGrid = common_robotics_utilities::voxel_grid
    ::VoxelGrid<CpuVoxelizationTrackingCell>;
using CpuVoxelizationTrackingGridAllocator =
      Eigen::aligned_allocator<CpuVoxelizationTrackingGrid>;
using VectorCpuVoxelizationTrackingGrid =
    std::vector<CpuVoxelizationTrackingGrid,
                CpuVoxelizationTrackingGridAllocator>;

void RaycastPointCloud(
    const PointCloudWrapper& cloud, const double step_size,
    const common_robotics_utilities::openmp_helpers::DegreeOfParallelism&
        parallelism,
    CpuVoxelizationTrackingGrid& tracking_grid)
{
  // Get X_GW, the transform from grid origin to world
  const Eigen::Isometry3d& X_GW = tracking_grid.GetInverseOriginTransform();
  // Get X_WC, the transform from world to the origin of the pointcloud
  const Eigen::Isometry3d& X_WC = cloud.GetPointCloudOriginTransform();
  // Transform X_GC, transform from grid origin to the origin of the pointcloud
  const Eigen::Isometry3d X_GC = X_GW * X_WC;
  // Get the pointcloud origin in grid frame
  const Eigen::Vector4d p_GCo = X_GC * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  // Get the max range
  const double max_range = cloud.MaxRange();
  CRU_OMP_PARALLEL_FOR_DEGREE(parallelism)
  for (int64_t idx = 0; idx < cloud.Size(); idx++)
  {
    // Location of point P in frame of camera C
    const Eigen::Vector4d p_CP = cloud.GetPointLocationVector4d(idx);
    // Skip invalid points marked with NaN or infinity
    if (std::isfinite(p_CP(0)) && std::isfinite(p_CP(1)) &&
        std::isfinite(p_CP(2)))
    {
      // Location of point P in grid G
      const Eigen::Vector4d p_GP = X_GC * p_CP;
      // Ray from camera to point
      const Eigen::Vector4d ray = p_GP - p_GCo;
      const double ray_length = ray.norm();
      // Step along ray
      const int32_t num_steps =
          std::max(1, static_cast<int32_t>(std::floor(ray_length / step_size)));
      common_robotics_utilities::voxel_grid::GridIndex last_index(-1, -1, -1);
      bool ray_crossed_grid = false;
      for (int32_t step = 0; step < num_steps; step++)
      {
        const double ratio =
            static_cast<double>(step) / static_cast<double>(num_steps);
        if ((ratio * ray_length) > max_range)
        {
          // We've gone beyond max range of the sensor
          break;
        }
        const Eigen::Vector4d p_GQ = p_GCo + (ray * ratio);
        const common_robotics_utilities::voxel_grid::GridIndex index =
            tracking_grid.LocationInGridFrameToGridIndex4d(p_GQ);
        // We don't want to double count in the same cell multiple times
        if (!(index == last_index))
        {
          auto query = tracking_grid.GetIndexMutable(index);
          // We must check to see if the query is within bounds.
          if (query)
          {
            ray_crossed_grid = true;
            query.Value().seen_free_count.fetch_add(1);
          }
          else if (ray_crossed_grid)
          {
            // We've left the grid and there's no reason to keep going.
            break;
          }
        }
        last_index = index;
      }
      // Set the point itself as filled, if it is in range
      if (ray_length <= max_range)
      {
        const common_robotics_utilities::voxel_grid::GridIndex index =
              tracking_grid.LocationInGridFrameToGridIndex4d(p_GP);
        auto query = tracking_grid.GetIndexMutable(index);
        // We must check to see if the query is within bounds.
        if (query)
        {
          query.Value().seen_filled_count.fetch_add(1);
        }
      }
    }
  }
}

void CombineAndFilterGrids(
    const PointCloudVoxelizationFilterOptions& filter_options,
    const VectorCpuVoxelizationTrackingGrid& tracking_grids,
    const common_robotics_utilities::openmp_helpers::DegreeOfParallelism&
        parallelism,
    CollisionMap& filtered_grid)
{
  // Because we want to improve performance and don't need to know where in the
  // grid we are, we can take advantage of the dense backing vector to iterate
  // through the grid data, rather than the grid cells.
  CRU_OMP_PARALLEL_FOR_DEGREE(parallelism)
  for (int64_t voxel = 0; voxel < filtered_grid.GetTotalCells(); voxel++)
  {
    auto& current_cell = filtered_grid.GetDataIndexMutable(voxel);
    // Filled cells stay filled, we don't work with them.
    // We only change cells that are unknown or empty.
    if (current_cell.Occupancy() <= 0.5)
    {
      int32_t seen_filled = 0;
      int32_t seen_free = 0;
      for (size_t idx = 0; idx < tracking_grids.size(); idx++)
      {
        const CpuVoxelizationTrackingCell& grid_cell =
            tracking_grids.at(idx).GetDataIndexImmutable(voxel);
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
}
}  // namespace

CpuPointCloudVoxelizer::CpuPointCloudVoxelizer(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
{
  const int32_t cpu_parallelize =
      RetrieveOptionOrDefault(options, "CPU_PARALLELIZE", 1, logging_fn);
  const int32_t cpu_num_threads =
      RetrieveOptionOrDefault(options, "CPU_NUM_THREADS", -1, logging_fn);
  if (cpu_parallelize > 0)
  {
    parallelism_ =
        common_robotics_utilities::openmp_helpers::DegreeOfParallelism(
            cpu_num_threads);
  }
  else
  {
    parallelism_ =
        common_robotics_utilities::openmp_helpers::DegreeOfParallelism(false);
  }
}

VoxelizerRuntime CpuPointCloudVoxelizer::DoVoxelizePointClouds(
    const CollisionMap& static_environment, const double step_size_multiplier,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
    CollisionMap& output_environment) const
{
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
    const PointCloudWrapperSharedPtr& cloud_ptr = pointclouds.at(idx);
    CpuVoxelizationTrackingGrid& tracking_grid = tracking_grids.at(idx);
    RaycastPointCloud(*cloud_ptr, step_size, Parallelism(), tracking_grid);
  }
  const std::chrono::time_point<std::chrono::steady_clock> raycasted_time =
      std::chrono::steady_clock::now();
  // Combine & filter
  CombineAndFilterGrids(
      filter_options, tracking_grids, Parallelism(), output_environment);
  const std::chrono::time_point<std::chrono::steady_clock> done_time =
      std::chrono::steady_clock::now();
  return VoxelizerRuntime(
      std::chrono::duration<double>(raycasted_time - start_time).count(),
      std::chrono::duration<double>(done_time - raycasted_time).count());
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
