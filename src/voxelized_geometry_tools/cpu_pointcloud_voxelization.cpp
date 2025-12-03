#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/parallelism.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/device_voxelization_interface.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForIndexLoop;
using common_robotics_utilities::voxel_grid::GridIndex;

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
CpuPointCloudVoxelizer::CpuPointCloudVoxelizer(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
{
  const int32_t cpu_parallelize =
      RetrieveOptionOrDefault(options, "CPU_PARALLELIZE", 1, logging_fn);
  const int32_t cpu_num_threads =
      RetrieveOptionOrDefault(options, "CPU_NUM_THREADS", -1, logging_fn);
  if (cpu_parallelize > 0 && cpu_num_threads >= 1)
  {
    parallelism_ = DegreeOfParallelism(cpu_num_threads);
    if (logging_fn)
    {
      logging_fn(
          "Configured parallelism using provided number of threads "
          + std::to_string(cpu_num_threads));
    }
  }
  else if (cpu_parallelize > 0)
  {
    parallelism_ = DegreeOfParallelism::FromOmp();
    if (logging_fn)
    {
      logging_fn(
          "Configured parallelism using OpenMP num threads "
          + std::to_string(Parallelism().GetNumThreads()));
    }
  }
  else
  {
    parallelism_ = DegreeOfParallelism::None();
    if (logging_fn)
    {
      logging_fn("Parallelism disabled");
    }
  }
}

void CpuPointCloudVoxelizer::RaycastPointCloud(
    const PointCloudWrapper& cloud,
    CpuVoxelizationTrackingGrid& tracking_grid) const
{
  if (tracking_grid.IsInitialized() && tracking_grid.HasUniformVoxelSize())
  {
    DoRaycastPointCloud(cloud, tracking_grid);
  }
  else
  {
    throw std::invalid_argument("Invalid tracking_grid provided");
  }
}

void CpuPointCloudVoxelizer::RaycastSinglePoint(
    const Eigen::Vector4d& p_GCo, const Eigen::Vector4d& p_GP,
    const double max_range, CpuVoxelizationTrackingGrid& tracking_grid) const
{
  const auto is_point_finite = [](const Eigen::Vector4d& point)
  {
    return std::isfinite(point(0)) && std::isfinite(point(1)) &&
           std::isfinite(point(2));
  };

  if (!is_point_finite(p_GCo))
  {
    throw std::invalid_argument("Non-finite p_GCo provided");
  }

  if (!is_point_finite(p_GP))
  {
    throw std::invalid_argument("Non-finite p_GP provided");
  }

  if (!tracking_grid.IsInitialized() || !tracking_grid.HasUniformVoxelSize())
  {
    throw std::invalid_argument("Invalid tracking_grid provided");
  }

  const GridIndex p_GCo_index =
      tracking_grid.LocationInGridFrameToGridIndex4d(p_GCo);
  DoRaycastSinglePoint(p_GCo, p_GCo_index, p_GP, max_range, tracking_grid);
}

void CpuPointCloudVoxelizer::CombineAndFilterGrids(
    const PointCloudVoxelizationFilterOptions& filter_options,
    const VectorCpuVoxelizationTrackingGrid& tracking_grids,
    OccupancyMap& filtered_grid) const
{
  for (const auto& tracking_grid : tracking_grids)
  {
    if (!tracking_grid.IsInitialized() || !tracking_grid.HasUniformVoxelSize())
    {
      throw std::invalid_argument("Invalid tracking_grid provided");
    }

    if (tracking_grid.ControlSizes() != filtered_grid.ControlSizes())
    {
      throw std::invalid_argument(
          "Incompatible control sizes between tracking_grid and filtered_grid");
    }
  }

  DoCombineAndFilterGrids(filter_options, tracking_grids, filtered_grid);
}

VoxelizerRuntime CpuPointCloudVoxelizer::DoVoxelizePointClouds(
    const OccupancyMap& static_environment,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
    OccupancyMap& output_environment) const
{
  const std::chrono::time_point<std::chrono::steady_clock> start_time =
      std::chrono::steady_clock::now();
  // Pose of grid G in world W.
  const Eigen::Isometry3d& X_WG = static_environment.OriginTransform();
  // Get grid resolution and size parameters.
  const auto& grid_sizes = static_environment.ControlSizes();
  // For each cloud, raycast it into its own "tracking grid"
  VectorCpuVoxelizationTrackingGrid tracking_grids(
      pointclouds.size(),
      CpuVoxelizationTrackingGrid(
          X_WG, grid_sizes, CpuVoxelizationTrackingCell()));
  for (size_t idx = 0; idx < pointclouds.size(); idx++)
  {
    const PointCloudWrapperSharedPtr& cloud_ptr = pointclouds.at(idx);
    CpuVoxelizationTrackingGrid& tracking_grid = tracking_grids.at(idx);
    DoRaycastPointCloud(*cloud_ptr, tracking_grid);
  }
  const std::chrono::time_point<std::chrono::steady_clock> raycasted_time =
      std::chrono::steady_clock::now();
  // Combine & filter
  DoCombineAndFilterGrids(filter_options, tracking_grids, output_environment);
  const std::chrono::time_point<std::chrono::steady_clock> done_time =
      std::chrono::steady_clock::now();
  return VoxelizerRuntime(
      std::chrono::duration<double>(raycasted_time - start_time).count(),
      std::chrono::duration<double>(done_time - raycasted_time).count());
}

void CpuPointCloudVoxelizer::DoRaycastPointCloud(
    const PointCloudWrapper& cloud,
    CpuVoxelizationTrackingGrid& tracking_grid) const
{
  // Get X_GW, the transform from grid origin to world
  const Eigen::Isometry3d& X_GW = tracking_grid.InverseOriginTransform();
  // Get X_WC, the transform from world to the origin of the pointcloud
  const Eigen::Isometry3d& X_WC = cloud.PointCloudOriginTransform();
  // Transform X_GC, transform from grid origin to the origin of the pointcloud
  const Eigen::Isometry3d X_GC = X_GW * X_WC;
  // Get the pointcloud origin in grid frame
  const Eigen::Vector4d p_GCo = X_GC * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  // Get the index for the origin in grid frame
  const GridIndex p_GCo_index =
      tracking_grid.LocationInGridFrameToGridIndex4d(p_GCo);
  // Get the max range
  const double max_range = cloud.MaxRange();

  // Helper lambda for raycasting a single point
  const auto per_item_work = [&](const int32_t, const int64_t point_index)
  {
    // Location of point P in frame of camera C
    const Eigen::Vector4d p_CP = cloud.GetPointLocationVector4d(point_index);
    // Skip invalid points marked with NaN or infinity
    if (std::isfinite(p_CP(0)) && std::isfinite(p_CP(1)) &&
        std::isfinite(p_CP(2)))
    {
      // Location of point P in grid G
      const Eigen::Vector4d p_GP = X_GC * p_CP;
      // Raycast point
      DoRaycastSinglePoint(p_GCo, p_GCo_index, p_GP, max_range, tracking_grid);
    }
  };

  // Raycast all points in the pointcloud. Use OpenMP if available, if not fall
  // back to manual dispatch via std::async.
  StaticParallelForIndexLoop(
      Parallelism(), 0, cloud.Size(), per_item_work,
      ParallelForBackend::BEST_AVAILABLE);
}

void CpuPointCloudVoxelizer::DoRaycastSinglePoint(
    const Eigen::Vector4d& p_GCo, const GridIndex& p_GCo_index,
    const Eigen::Vector4d& p_GP, const double max_range,
    CpuVoxelizationTrackingGrid& tracking_grid) const
{
  // This entire method is adapted from section 7.4.2 (p 324) of
  // Real-Time Collision Detection (Ericson, 2005)

  // Step 1: limit the final point to the provided maximum range.
  const Eigen::Vector4d ray = p_GP - p_GCo;
  const double ray_length = ray.norm();
  const bool clipped = ray_length > max_range;

  Eigen::Vector4d p_GFinal = p_GP;

  if (clipped)
  {
    p_GFinal = p_GCo + (ray * (max_range / ray_length));
  }

  // Step 2: get a starting point within the grid.
  const bool origin_in_grid = tracking_grid.CheckGridIndexInBounds(p_GCo_index);

  Eigen::Vector4d p_GStart = p_GCo;
  if (!origin_in_grid)
  {
    // Adapted from section 5.3.3 (p 179) of Real-Time Collision Detection.
    const Eigen::Vector3d grid_sizes = tracking_grid.GridSizes();

    double tmin = 0.0;
    double tmax = max_range;

    const Eigen::Vector4d direction = ray / ray_length;

    // Threshold for considering an axis direction as flat.
    const double flat_threshold = 1e-10;

    for (int axis = 0; axis < 3; axis++)
    {
      if (std::abs(direction(axis)) < flat_threshold)
      {
        // If the direction verctor is nearly zero, make sure it is within the
        // axis range of the grid; if not, terminate.
        const bool in_slab =
            p_GCo(axis) >= 0.0 && p_GCo(axis) < grid_sizes(axis);

        if (!in_slab)
        {
          return;
        }
      }
      else
      {
        // Check against the low and high planes of the current axis.
        const double ood = 1.0 / direction(axis);

        const double tlow = (0.0 - p_GCo(axis)) * ood;
        const double thigh = (grid_sizes(axis) - p_GCo(axis)) * ood;

        const double t1 = (tlow <= thigh) ? tlow : thigh;
        const double t2 = (tlow <= thigh) ? thigh : tlow;

        if (t1 > tmin)
        {
          tmin = t1;
        }
        if (t2 > tmax)
        {
          tmax = t2;
        }

        if (tmin > tmax)
        {
          // Line segment does not interset the grid, terminate.
          return;
        }
      }
    }

    // Nudge the point slightly farther into the grid to avoid any edge cases.
    const double nudge = 1e-10;
    p_GStart = p_GCo + (direction * (tmin + nudge));
  }

  // Step 3: grab indices for start and final points.
  const GridIndex p_GStart_index =
      tracking_grid.LocationInGridFrameToGridIndex4d(p_GStart);

  const GridIndex p_GFinal_index =
      tracking_grid.LocationInGridFrameToGridIndex4d(p_GFinal);

  // Step 4: get axis steps.
  const auto step_from_diff = [](const int64_t diff)
  {
    if (diff > 0)
    {
      return 1;
    }
    else if (diff < 0)
    {
      return -1;
    }
    else
    {
      return 0;
    }
  };

  const int64_t x_step =
      step_from_diff(p_GFinal_index.X() - p_GStart_index.X());
  const int64_t y_step =
      step_from_diff(p_GFinal_index.Y() - p_GStart_index.Y());
  const int64_t z_step =
      step_from_diff(p_GFinal_index.Z() - p_GStart_index.Z());

  // Step 5: compute the control values.
  const double voxel_size = tracking_grid.VoxelXSize();
  const double half_voxel_size = voxel_size * 0.5;
  const Eigen::Vector4d half_voxel_offset(
      half_voxel_size, half_voxel_size, half_voxel_size, 0.0);

  const Eigen::Vector4d p_GStart_index_center =
      tracking_grid.GridIndexToLocationInGridFrame(p_GStart_index);
  const Eigen::Vector4d voxel_bottom_corner =
      p_GStart_index_center - half_voxel_offset;
  const Eigen::Vector4d voxel_top_corner =
      p_GStart_index_center + half_voxel_offset;

  const auto get_axis_t_value = [](
      const double point_axis, const double ray_axis,
      const double voxel_min_axis, const double voxel_max_axis)
  {
    if (ray_axis > 0.0)
    {
      const double max_within_voxel = voxel_max_axis - point_axis;
      return std::abs(max_within_voxel / ray_axis);
    }
    else if (ray_axis < -0.0)
    {
      const double max_within_voxel = point_axis - voxel_min_axis;
      return std::abs(max_within_voxel / ray_axis);
    }
    else
    {
      return std::numeric_limits<double>::infinity();
    }
  };

  const double tx_initial = get_axis_t_value(
      p_GStart(0), ray(0), voxel_bottom_corner(0), voxel_top_corner(0));
  const double ty_initial = get_axis_t_value(
      p_GStart(1), ray(1), voxel_bottom_corner(1), voxel_top_corner(1));
  const double tz_initial = get_axis_t_value(
      p_GStart(2), ray(2), voxel_bottom_corner(2), voxel_top_corner(2));

  const double delta_tx = std::abs(voxel_size / ray(0));
  const double delta_ty = std::abs(voxel_size / ray(1));
  const double delta_tz = std::abs(voxel_size / ray(2));

  // Step 6: set the final point.
  auto p_GFinal_query = tracking_grid.GetIndexMutable(p_GFinal_index);
  if (p_GFinal_query)
  {
    if (clipped)
    {
      // If the actual point was clipped, mark the final voxel as seen-free.
      p_GFinal_query.Value().seen_free_count.fetch_add(1);
    }
    else
    {
      // If the point was not clipped, mark the final voxel as seen-filled.
      p_GFinal_query.Value().seen_filled_count.fetch_add(1);
    }
  }

  /// Iterate along line.
  GridIndex current_index = p_GStart_index;
  double tx = tx_initial;
  double ty = ty_initial;
  double tz = tz_initial;

  while (current_index != p_GFinal_index)
  {
    // Update the current voxel.
    auto query = tracking_grid.GetIndexMutable(current_index);
    if (query)
    {
      // If the query is in bounds, update the seen-free count.
      query.Value().seen_free_count.fetch_add(1);
    }
    else
    {
      // If the query is out of bounds, we are done.
      break;
    }

    // Step.
    if (tx <= ty && tx <= tz)
    {
      if (current_index.X() == p_GFinal_index.X())
      {
        // If we would step out of range, we are done.
        break;
      }
      current_index.X() += x_step;
      tx += delta_tx;
    }
    else if (ty <= tx && ty <= tz)
    {
      if (current_index.Y() == p_GFinal_index.Y())
      {
        // If we would step out of range, we are done.
        break;
      }
      current_index.Y() += y_step;
      ty += delta_ty;
    }
    else
    {
      if (current_index.Z() == p_GFinal_index.Z())
      {
        // If we would step out of range, we are done.
        break;
      }
      current_index.Z() += z_step;
      tz += delta_tz;
    }
  }
}

void CpuPointCloudVoxelizer::DoCombineAndFilterGrids(
    const PointCloudVoxelizationFilterOptions& filter_options,
    const VectorCpuVoxelizationTrackingGrid& tracking_grids,
    OccupancyMap& filtered_grid) const
{
  // Because we want to improve performance and don't need to know where in the
  // grid we are, we can take advantage of the dense backing vector to iterate
  // through the grid data, rather than the grid cells.

  // Helper lambda for each item's work
  const auto per_item_work = [&](const int32_t, const int64_t voxel_index)
  {
    auto& current_cell = filtered_grid.GetDataIndexMutable(voxel_index);
    // Filled cells stay filled, we don't work with them.
    // We only change cells that are unknown or empty.
    if (current_cell.Occupancy() <= 0.5)
    {
      int32_t seen_filled = 0;
      int32_t seen_free = 0;
      for (size_t idx = 0; idx < tracking_grids.size(); idx++)
      {
        const CpuVoxelizationTrackingCell& grid_cell =
            tracking_grids.at(idx).GetDataIndexImmutable(voxel_index);
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
        current_cell.SetOccupancy(1.0);
      }
      else if (seen_free >= filter_options.NumCamerasSeenFree())
      {
        // Did enough cameras see this empty?
        current_cell.SetOccupancy(0.0);
      }
      else
      {
        // Otherwise, it is unknown.
        current_cell.SetOccupancy(0.5);
      }
    }
  };

  // Use OpenMP if available, if not fall back to manual dispatch via
  // std::async.
  StaticParallelForIndexLoop(
      Parallelism(), 0, filtered_grid.NumTotalVoxels(), per_item_work,
      ParallelForBackend::BEST_AVAILABLE);
}
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
