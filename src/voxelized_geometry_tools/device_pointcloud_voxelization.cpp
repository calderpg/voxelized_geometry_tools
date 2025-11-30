#include <voxelized_geometry_tools/device_pointcloud_voxelization.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/utility.hpp>
#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::DynamicParallelForIndexLoop;
using common_robotics_utilities::parallelism::ParallelForBackend;

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
DevicePointCloudVoxelizer::DevicePointCloudVoxelizer(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
{
  const int32_t dispatch_parallelize =
      RetrieveOptionOrDefault(options, "DISPATCH_PARALLELIZE", 1, logging_fn);
  const int32_t dispatch_num_threads =
      RetrieveOptionOrDefault(options, "DISPATCH_NUM_THREADS", -1, logging_fn);
  if (dispatch_parallelize > 0 && dispatch_num_threads >= 1)
  {
    dispatch_parallelism_ = DegreeOfParallelism(dispatch_num_threads);
    if (logging_fn)
    {
      logging_fn(
          "Configured dispatch parallelism using provided number of threads "
          + std::to_string(dispatch_num_threads));
    }
  }
  else if (dispatch_parallelize > 0)
  {
    dispatch_parallelism_ = DegreeOfParallelism::FromOmp();
    if (logging_fn)
    {
      logging_fn(
          "Configured dispatch parallelism using OpenMP num threads "
          + std::to_string(DispatchParallelism().GetNumThreads()));
    }
  }
  else
  {
    dispatch_parallelism_ = DegreeOfParallelism::None();
    if (logging_fn)
    {
      logging_fn("Dispatch parallelism disabled");
    }
  }
}

VoxelizerRuntime DevicePointCloudVoxelizer::DoVoxelizePointClouds(
    const OccupancyMap& static_environment, const double step_size_multiplier,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
    OccupancyMap& output_environment) const
{
  EnforceAvailable();

  const std::chrono::time_point<std::chrono::steady_clock> start_time =
      std::chrono::steady_clock::now();

  // Allocate device-side memory for tracking grids. Note that at least one grid
  // is always allocated so that filtering is consistent, even if no points are
  // raycast.
  const size_t num_tracking_grids =
      std::max(pointclouds.size(), static_cast<size_t>(1));

  std::unique_ptr<TrackingGridsHandle> tracking_grids =
      helper_interface_->PrepareTrackingGrids(
          static_environment.NumTotalVoxels(),
          static_cast<int32_t>(num_tracking_grids));
  if (tracking_grids->GetNumTrackingGrids() != num_tracking_grids)
  {
    throw std::runtime_error("Failed to allocate device tracking grid");
  }

  // Get X_GW, the transform from grid origin to world
  const Eigen::Isometry3d& X_GW = static_environment.InverseOriginTransform();

  // Prepare grid data
  const float inverse_step_size =
      static_cast<float>(1.0 /
          (static_environment.Resolution() * step_size_multiplier));
  const float inverse_cell_size =
      static_cast<float>(static_environment.ControlSizes().InverseVoxelXSize());
  const int32_t num_x_cells =
      static_cast<int32_t>(static_environment.NumXVoxels());
  const int32_t num_y_cells =
      static_cast<int32_t>(static_environment.NumYVoxels());
  const int32_t num_z_cells =
      static_cast<int32_t>(static_environment.NumZVoxels());

  // Lambda for the raycasting of a single pointcloud.
  const auto per_item_work = [&](const int32_t, const int64_t pointcloud_index)
  {
    const PointCloudWrapperSharedPtr& pointcloud =
        pointclouds.at(static_cast<size_t>(pointcloud_index));

    // Only do work if the pointcloud is non-empty, to avoid passing empty
    // arrays into the device interface.
    if (pointcloud->Size() > 0)
    {
      // Get X_WC, the transform from world to the origin of the pointcloud
      const Eigen::Isometry3d& X_WC =
          pointcloud->PointCloudOriginTransform();
      // X_GC, transform from grid origin to the origin of the pointcloud
      const Eigen::Isometry3f grid_pointcloud_transform_float =
          (X_GW * X_WC).cast<float>();

      const float max_range = static_cast<float>(pointcloud->MaxRange());

      // Copy pointcloud
      std::vector<float> raw_points(
          static_cast<size_t>(pointcloud->Size()) * 3, 0.0);
      for (int64_t point = 0; point < pointcloud->Size(); point++)
      {
        pointcloud->CopyPointLocationIntoVectorFloat(
            point, raw_points, point * 3);
      }

      // Raycast
      helper_interface_->RaycastPoints(
          raw_points, max_range, grid_pointcloud_transform_float.data(),
          inverse_step_size, inverse_cell_size, num_x_cells, num_y_cells,
          num_z_cells, *tracking_grids, static_cast<size_t>(pointcloud_index));
    }
  };

  DynamicParallelForIndexLoop(
      DispatchParallelism(), 0, static_cast<int64_t>(pointclouds.size()),
      per_item_work, ParallelForBackend::BEST_AVAILABLE);

  const std::chrono::time_point<std::chrono::steady_clock> raycasted_time =
      std::chrono::steady_clock::now();

  // Filter
  const float percent_seen_free =
      static_cast<float>(filter_options.PercentSeenFree());
  const int32_t outlier_points_threshold =
      filter_options.OutlierPointsThreshold();
  const int32_t num_cameras_seen_free =
      filter_options.NumCamerasSeenFree();

  std::unique_ptr<FilterGridHandle> filter_grid =
      helper_interface_->PrepareFilterGrid(
          static_environment.NumTotalVoxels(),
          static_environment.GetImmutableRawData().data());

  helper_interface_->FilterTrackingGrids(
      *tracking_grids, percent_seen_free, outlier_points_threshold,
      num_cameras_seen_free, *filter_grid);

  // Retrieve & return
  helper_interface_->RetrieveFilteredGrid(
      *filter_grid, output_environment.GetMutableRawData().data());

  const std::chrono::time_point<std::chrono::steady_clock> done_time =
      std::chrono::steady_clock::now();

  return VoxelizerRuntime(
      std::chrono::duration<double>(raycasted_time - start_time).count(),
      std::chrono::duration<double>(done_time - raycasted_time).count());
}

CudaPointCloudVoxelizer::CudaPointCloudVoxelizer(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
    : DevicePointCloudVoxelizer(options, logging_fn)
{
  device_name_ = "CudaPointCloudVoxelizer";
  helper_interface_ = std::unique_ptr<DeviceVoxelizationHelperInterface>(
      cuda_helpers::MakeCudaVoxelizationHelper(options, logging_fn));
  EnforceAvailable();
}

OpenCLPointCloudVoxelizer::OpenCLPointCloudVoxelizer(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
    : DevicePointCloudVoxelizer(options, logging_fn)
{
  device_name_ = "OpenCLPointCloudVoxelizer";
  helper_interface_ = std::unique_ptr<DeviceVoxelizationHelperInterface>(
      opencl_helpers::MakeOpenCLVoxelizationHelper(options, logging_fn));
  EnforceAvailable();
}
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
