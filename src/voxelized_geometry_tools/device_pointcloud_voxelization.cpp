#include <voxelized_geometry_tools/device_pointcloud_voxelization.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>
#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
VoxelizerRuntime DevicePointCloudVoxelizer::DoVoxelizePointClouds(
    const CollisionMap& static_environment, const double step_size_multiplier,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointCloudWrapperPtr>& pointclouds,
    CollisionMap& output_environment) const
{
  EnforceAvailable();

  // We shortcut here if there's no work to do.
  if (pointclouds.empty())
  {
    return VoxelizerRuntime(0.0, 0.0);
  }

  const std::chrono::time_point<std::chrono::steady_clock> start_time =
      std::chrono::steady_clock::now();

  // Allocate device-side memory for tracking grids
  std::unique_ptr<TrackingGridsHandle> tracking_grids =
      helper_interface_->PrepareTrackingGrids(
          static_environment.GetTotalCells(),
          static_cast<int32_t>(pointclouds.size()));
  if (tracking_grids->GetNumTrackingGrids() != pointclouds.size())
  {
    throw std::runtime_error("Failed to allocate device tracking grid");
  }

  // Prepare grid data
  const Eigen::Isometry3f inverse_grid_origin_transform_float =
      static_environment.GetInverseOriginTransform().cast<float>();
  const float inverse_step_size =
      static_cast<float>(1.0 /
          (static_environment.GetResolution() * step_size_multiplier));
  const float inverse_cell_size =
      static_cast<float>(static_environment.GetGridSizes().InvCellXSize());
  const int32_t num_x_cells =
      static_cast<int32_t>(static_environment.GetNumXCells());
  const int32_t num_y_cells =
      static_cast<int32_t>(static_environment.GetNumYCells());
  const int32_t num_z_cells =
      static_cast<int32_t>(static_environment.GetNumZCells());

  // Do raycasting of the pointclouds
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (size_t idx = 0; idx < pointclouds.size(); idx++)
  {
    const PointCloudWrapperPtr& pointcloud = pointclouds.at(idx);
    const Eigen::Isometry3f pointcloud_origin_transform_float =
        pointcloud->GetPointCloudOriginTransform().cast<float>();
    const float max_range = static_cast<float>(pointcloud->MaxRange());

    // Copy pointcloud
    std::vector<float> raw_points(pointcloud->Size() * 3, 0.0);
    for (int64_t point = 0; point < pointcloud->Size(); point++)
    {
      pointcloud->CopyPointLocationIntoVectorFloat(
          point, raw_points, point * 3);
    }

    // Raycast
    helper_interface_->RaycastPoints(
        raw_points, pointcloud_origin_transform_float.data(), max_range,
        inverse_grid_origin_transform_float.data(), inverse_step_size,
        inverse_cell_size, num_x_cells, num_y_cells, num_z_cells,
        *tracking_grids, idx);
  }

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
          static_environment.GetTotalCells(),
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
    const std::map<std::string, int32_t>& options)
{
  device_name_ = "CudaPointCloudVoxelizer";
  helper_interface_ = std::unique_ptr<DeviceVoxelizationHelperInterface>(
      cuda_helpers::MakeCudaVoxelizationHelper(options));
  EnforceAvailable();
}

OpenCLPointCloudVoxelizer::OpenCLPointCloudVoxelizer(
    const std::map<std::string, int32_t>& options)
{
  device_name_ = "OpenCLPointCloudVoxelizer";
  helper_interface_ = std::unique_ptr<DeviceVoxelizationHelperInterface>(
      opencl_helpers::MakeOpenCLVoxelizationHelper(options));
  EnforceAvailable();
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
