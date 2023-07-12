#include <voxelized_geometry_tools/device_pointcloud_voxelization.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/utility.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>
#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

using common_robotics_utilities::openmp_helpers::DegreeOfParallelism;

namespace voxelized_geometry_tools
{
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
    const CollisionMap& static_environment, const double step_size_multiplier,
    const PointCloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
    CollisionMap& output_environment) const
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
          static_environment.GetTotalCells(),
          static_cast<int32_t>(num_tracking_grids));
  if (tracking_grids->GetNumTrackingGrids() != num_tracking_grids)
  {
    throw std::runtime_error("Failed to allocate device tracking grid");
  }

  // Get X_GW, the transform from grid origin to world
  const Eigen::Isometry3d& X_GW =
      static_environment.GetInverseOriginTransform();

  // Prepare grid data
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

  // Lambda for the raycasting of a single pointcloud.
  const auto raycast_cloud = [&](const size_t idx)
  {
    const PointCloudWrapperSharedPtr& pointcloud = pointclouds.at(idx);

    // Only do work if the pointcloud is non-empty, to avoid passing empty
    // arrays into the device interface.
    if (pointcloud->Size() > 0)
    {
      // Get X_WC, the transform from world to the origin of the pointcloud
      const Eigen::Isometry3d& X_WC =
          pointcloud->GetPointCloudOriginTransform();
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
          num_z_cells, *tracking_grids, idx);
    }
  };

  // Dispatch worker threads.
  const int32_t num_dispatch_threads = DispatchParallelism().GetNumThreads();

  size_t workers_dispatched = 0;
  size_t num_live_workers = 0;

  std::mutex cv_mutex;
  std::condition_variable cv;

  std::list<std::future<void>> active_workers;

  while (active_workers.size() > 0 || workers_dispatched < pointclouds.size())
  {
    // Check for completed workers.
    for (auto worker = active_workers.begin(); worker != active_workers.end();)
    {
      if (common_robotics_utilities::utility::IsFutureReady(*worker))
      {
        // This call to future.get() is necessary to propagate any exception
        // thrown during simulation execution.
        worker->get();
        // Erase returns iterator to the next node in the list.
        worker = active_workers.erase(worker);
      }
      else
      {
        // Advance to next node in the list.
        ++worker;
      }
    }

    // Dispatch new workers.
    while (static_cast<int32_t>(active_workers.size()) < num_dispatch_threads
           && workers_dispatched < pointclouds.size())
    {
      {
        std::lock_guard<std::mutex> lock(cv_mutex);
        num_live_workers++;
      }
      active_workers.emplace_back(std::async(
          std::launch::async,
          [&raycast_cloud, &cv, &cv_mutex, &num_live_workers](
              const size_t pointcloud_idx)
          {
            raycast_cloud(pointcloud_idx);
            {
              std::lock_guard<std::mutex> lock(cv_mutex);
              num_live_workers--;
            }
            cv.notify_all();
          },
          workers_dispatched));
      workers_dispatched++;
    }

    // Wait until a worker completes.
    if (active_workers.size() > 0)
    {
      std::unique_lock<std::mutex> wait_lock(cv_mutex);
      cv.wait(
          wait_lock,
          [&num_live_workers, &active_workers]()
          {
            return num_live_workers < active_workers.size();
          });
    }
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
}  // namespace voxelized_geometry_tools
