#include <voxelized_geometry_tools/cuda_pointcloud_voxelization.hpp>

#include <omp.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
CudaPointCloudVoxelizer::CudaPointCloudVoxelizer()
{
  if (!cuda_helpers::IsAvailable())
  {
    throw std::runtime_error("CudaPointCloudVoxelizer not available");
  }
}

CollisionMap CudaPointCloudVoxelizer::VoxelizePointClouds(
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
  if (cuda_helpers::IsAvailable())
  {
    const std::chrono::time_point<std::chrono::steady_clock> start_time =
        std::chrono::steady_clock::now();
    // Prepare and copy over pointclouds to the device
    std::vector<float*> device_pointcloud_ptrs(pointclouds.size(), nullptr);
#pragma omp parallel for
    for (size_t idx = 0; idx < pointclouds.size(); idx++)
    {
      const PointCloudWrapperPtr& pointcloud = pointclouds.at(idx);
      // Copy pointcloud
      std::vector<float> raw_points(pointcloud->Size() * 3, 0.0);
      for (int64_t point = 0; point < pointcloud->Size(); point++)
      {
        pointcloud->CopyPointLocationIntoVectorFloat(
            point, raw_points, point * 3);
      }
      auto device_pointcloud_ptr =
          cuda_helpers::PreparePointCloud(
              static_cast<int32_t>(pointcloud->Size()), raw_points.data());
      if (device_pointcloud_ptr != nullptr)
      {
        device_pointcloud_ptrs[idx] = device_pointcloud_ptr;
      }
      else
      {
        throw std::runtime_error("Failed to allocate device pointcloud");
      }
    }
    // Allocate device-side memory for tracking grids
    const int32_t num_tracking_grids = static_cast<int32_t>(pointclouds.size());
    std::vector<int32_t*> device_tracking_grid_ptrs(
        pointclouds.size(), nullptr);
    for (size_t idx = 0; idx < pointclouds.size(); idx++)
    {
      auto device_tracking_grid_ptr =
          cuda_helpers::PrepareTrackingGrid(static_environment.GetTotalCells());
      if (device_tracking_grid_ptr != nullptr)
      {
        device_tracking_grid_ptrs[idx] = device_tracking_grid_ptr;
      }
      else
      {
        throw std::runtime_error("Failed to allocate device tracking grid");
      }
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
    const std::chrono::time_point<std::chrono::steady_clock> prepared_time =
        std::chrono::steady_clock::now();
    // Do raycasting of the pointclouds
#pragma omp parallel for
    for (size_t idx = 0; idx < pointclouds.size(); idx++)
    {
      const PointCloudWrapperPtr& pointcloud = pointclouds.at(idx);
      const Eigen::Isometry3f pointcloud_origin_transform_float =
          pointcloud->GetPointCloudOriginTransform().cast<float>();
      // Raycast
      cuda_helpers::RaycastPoints(
          device_pointcloud_ptrs.at(idx),
          static_cast<int32_t>(pointcloud->Size()),
          pointcloud_origin_transform_float.data(),
          inverse_grid_origin_transform_float.data(), inverse_step_size,
          inverse_cell_size, num_x_cells, num_y_cells, num_z_cells,
          device_tracking_grid_ptrs.at(idx));
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
    auto device_filter_grid_ptr =
        cuda_helpers::PrepareFilterGrid(
            static_environment.GetTotalCells(),
            static_environment.GetImmutableRawData().data());
    if (device_filter_grid_ptr == nullptr)
    {
      throw std::runtime_error("Failed to allocate device filter grid");
    }
    cuda_helpers::FilterTrackingGrids(
        static_environment.GetTotalCells(), num_tracking_grids,
        device_tracking_grid_ptrs.data(), device_filter_grid_ptr,
        percent_seen_free, outlier_points_threshold, num_cameras_seen_free);
    // Retrieve & return
    CollisionMap filtered_grid = static_environment;
    cuda_helpers::RetrieveFilteredGrid(
        static_environment.GetTotalCells(), device_filter_grid_ptr,
        filtered_grid.GetMutableRawData().data());
    // Cleanup device memory
    cuda_helpers::CleanupDeviceMemory(
        num_tracking_grids, device_pointcloud_ptrs.data(),
        num_tracking_grids, device_tracking_grid_ptrs.data(),
        device_filter_grid_ptr);
    const std::chrono::time_point<std::chrono::steady_clock> done_time =
        std::chrono::steady_clock::now();
    std::cout
        << "Preparation time "
        << std::chrono::duration<double>(prepared_time - start_time).count()
        << ", raycasting time "
        << std::chrono::duration<double>(raycasted_time - prepared_time).count()
        << ", filtering time "
        << std::chrono::duration<double>(done_time - raycasted_time).count()
        << std::endl;
    return filtered_grid;
  }
  else
  {
    throw std::runtime_error("CudaPointcloudVoxelizer not available");
  }
}
}
}
