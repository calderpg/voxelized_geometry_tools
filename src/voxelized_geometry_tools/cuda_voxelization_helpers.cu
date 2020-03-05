#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <iostream>
#include <string>
#include <vector>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
void CudaCheckErrors(const std::string& msg)
{
  const cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    const std::string error_string(cudaGetErrorString(last_error));
    throw std::runtime_error("[" + msg + "] Cuda error [" + error_string + "]");
  }
}

__global__
void RaycastPoint(
    const float* const device_points_ptr, const int32_t num_points,
    const float* const device_pointcloud_origin_transform_ptr,
    const float* const device_tracking_grid_inverse_origin_transform_ptr,
    const float inverse_step_size, const float inverse_cell_size,
    const int32_t num_x_cells, const int32_t num_y_cells,
    const int32_t num_z_cells, const int32_t stride1, const int32_t stride2,
    int32_t* const device_tracking_grid_ptr)
{
  const int32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index < num_points)
  {
    const float ox = device_pointcloud_origin_transform_ptr[12];
    const float oy = device_pointcloud_origin_transform_ptr[13];
    const float oz = device_pointcloud_origin_transform_ptr[14];
    const float px = device_points_ptr[(point_index * 3) + 0];
    const float py = device_points_ptr[(point_index * 3) + 1];
    const float pz = device_points_ptr[(point_index * 3) + 2];
    const float wx = device_pointcloud_origin_transform_ptr[0] * px
                     + device_pointcloud_origin_transform_ptr[4] * py
                     + device_pointcloud_origin_transform_ptr[8] * pz
                     + device_pointcloud_origin_transform_ptr[12];
    const float wy = device_pointcloud_origin_transform_ptr[1] * px
                     + device_pointcloud_origin_transform_ptr[5] * py
                     + device_pointcloud_origin_transform_ptr[9] * pz
                     + device_pointcloud_origin_transform_ptr[13];
    const float wz = device_pointcloud_origin_transform_ptr[2] * px
                     + device_pointcloud_origin_transform_ptr[6] * py
                     + device_pointcloud_origin_transform_ptr[10] * pz
                     + device_pointcloud_origin_transform_ptr[14];
    const float rx = wx - ox;
    const float ry = wy - oy;
    const float rz = wz - oz;
    const float current_ray_length = sqrtf((rx * rx) + (ry * ry) + (rz * rz));
    const float num_steps =
        floor(current_ray_length * inverse_step_size);
    int32_t previous_x_cell = -1;
    int32_t previous_y_cell = -1;
    int32_t previous_z_cell = -1;
    for (float step = 0.0; step < num_steps; step += 1.0)
    {
      bool in_grid = false;
      const float elapsed_ratio = step / num_steps;
      const float cx = (rx * elapsed_ratio) + ox;
      const float cy = (ry * elapsed_ratio) + oy;
      const float cz = (rz * elapsed_ratio) + oz;
      const float gx =
          device_tracking_grid_inverse_origin_transform_ptr[0] * cx
          + device_tracking_grid_inverse_origin_transform_ptr[4] * cy
          + device_tracking_grid_inverse_origin_transform_ptr[8] * cz
          + device_tracking_grid_inverse_origin_transform_ptr[12];
      const float gy =
          device_tracking_grid_inverse_origin_transform_ptr[1] * cx
          + device_tracking_grid_inverse_origin_transform_ptr[5] * cy
          + device_tracking_grid_inverse_origin_transform_ptr[9] * cz
          + device_tracking_grid_inverse_origin_transform_ptr[13];
      const float gz =
          device_tracking_grid_inverse_origin_transform_ptr[2] * cx
          + device_tracking_grid_inverse_origin_transform_ptr[6] * cy
          + device_tracking_grid_inverse_origin_transform_ptr[10] * cz
          + device_tracking_grid_inverse_origin_transform_ptr[14];
      const int32_t x_cell = static_cast<int32_t>(gx * inverse_cell_size);
      const int32_t y_cell = static_cast<int32_t>(gy * inverse_cell_size);
      const int32_t z_cell = static_cast<int32_t>(gz * inverse_cell_size);
      if (x_cell != previous_x_cell || y_cell != previous_y_cell
          || z_cell != previous_z_cell)
      {
        if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0
            && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)
        {
          in_grid = true;
          const int32_t cell_index =
              (x_cell * stride1) + (y_cell * stride2) + z_cell;
          // Increase free count
          atomicAdd(&(device_tracking_grid_ptr[cell_index * 2]), 1);
        }
        else if (in_grid)
        {
          // We've left the grid and there's no reason to keep going.
          break;
        }
      }
      previous_x_cell = x_cell;
      previous_y_cell = y_cell;
      previous_z_cell = z_cell;
    }
    // Set the point itself as filled
    const float gx =
        device_tracking_grid_inverse_origin_transform_ptr[0] * wx
        + device_tracking_grid_inverse_origin_transform_ptr[4] * wy
        + device_tracking_grid_inverse_origin_transform_ptr[8] * wz
        + device_tracking_grid_inverse_origin_transform_ptr[12];
    const float gy =
        device_tracking_grid_inverse_origin_transform_ptr[1] * wx
        + device_tracking_grid_inverse_origin_transform_ptr[5] * wy
        + device_tracking_grid_inverse_origin_transform_ptr[9] * wz
        + device_tracking_grid_inverse_origin_transform_ptr[13];
    const float gz =
        device_tracking_grid_inverse_origin_transform_ptr[2] * wx
        + device_tracking_grid_inverse_origin_transform_ptr[6] * wy
        + device_tracking_grid_inverse_origin_transform_ptr[10] * wz
        + device_tracking_grid_inverse_origin_transform_ptr[14];
    const int32_t x_cell = static_cast<int32_t>(gx * inverse_cell_size);
    const int32_t y_cell = static_cast<int32_t>(gy * inverse_cell_size);
    const int32_t z_cell = static_cast<int32_t>(gz * inverse_cell_size);
    if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0
        && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)
    {
      const int32_t cell_index =
          (x_cell * stride1) + (y_cell * stride2) + z_cell;
      // Increase filled count
      atomicAdd(&(device_tracking_grid_ptr[(cell_index * 2) + 1]), 1);
    }
  }
}

__global__
void FilterGrids(
    const int64_t num_cells, const int32_t num_grids,
    const int32_t* const device_tracking_grids_ptr,
    float* const device_filter_grid_ptr, const float percent_seen_free,
    const int32_t outlier_points_threshold, const int32_t num_cameras_seen_free)
{
  const int32_t voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_index < num_cells)
  {
    const float current_occupancy = device_filter_grid_ptr[voxel_index * 2];
    // Filled cells stay filled, we don't work with them.
    // We only change cells that are unknown or empty.
    if (current_occupancy <= 0.5)
    {
      int32_t cameras_seen_filled = 0;
      int32_t cameras_seen_free = 0;
      for (int32_t idx = 0; idx < num_grids; idx++)
      {
        const int32_t* const device_tracking_grid_ptr =
            device_tracking_grids_ptr + (idx * num_cells * 2);
        const int32_t free_count = device_tracking_grid_ptr[voxel_index * 2];
        const int32_t filled_count =
            device_tracking_grid_ptr[(voxel_index * 2) + 1];
        const int32_t filtered_filled_count =
            (filled_count >= outlier_points_threshold) ? filled_count : 0;
        if (free_count > 0 && filtered_filled_count > 0)
        {
          const float current_percent_seen_free =
              static_cast<float>(free_count)
              / static_cast<float>(free_count + filtered_filled_count);
          if (current_percent_seen_free >= percent_seen_free)
          {
            cameras_seen_free += 1;
          }
          else
          {
            cameras_seen_filled += 1;
          }
        }
        else if (free_count > 0)
        {
          cameras_seen_free += 1;
        }
        else if (filtered_filled_count > 0)
        {
          cameras_seen_filled += 1;
        }
      }
      if (cameras_seen_filled > 0)
      {
        // If any camera saw something here, it is filled.
        device_filter_grid_ptr[voxel_index * 2] = 1.0;
      }
      else if (cameras_seen_free >= num_cameras_seen_free)
      {
        // Did enough cameras see this empty?
        device_filter_grid_ptr[voxel_index * 2] = 0.0;
      }
      else
      {
        // Otherwise, it is unknown.
        device_filter_grid_ptr[voxel_index * 2] = 0.5;
      }
    }
  }
}

int32_t RetrieveOptionOrDefault(
    const std::map<std::string, int32_t>& options, const std::string& option,
    const int32_t default_value)
{
  auto found_itr = options.find(option);
  if (found_itr != options.end())
  {
    const int32_t value = found_itr->second;
    std::cout << "Option [" << option << "] found with value [" << value << "]"
              << std::endl;
    return value;
  }
  else
  {
    std::cout << "Option [" << option << "] not found, default ["
              << default_value << "]" << std::endl;
    return default_value;
  }
}

class RealCudaVoxelizationHelperInterface
    : public CudaVoxelizationHelperInterface
{
public:
  explicit RealCudaVoxelizationHelperInterface(
      const std::map<std::string, int32_t>& options)
  {
    const int32_t cuda_device =
        RetrieveOptionOrDefault(options, "CUDA_DEVICE", 0);
    try
    {
      int32_t device_count = 0;
      cudaGetDeviceCount(&device_count);
      CudaCheckErrors("Failed to get device count");
      if (cuda_device >= 0 && cuda_device < device_count)
      {
        cuda_device_num_ = cuda_device;
        SetCudaDevice();
      }
      else
      {
        std::cerr << "CUDA_DEVICE = " << cuda_device << " out of range for "
                  << device_count << " devices" << std::endl;
        cuda_device_num_ = -1;
      }
    }
    catch (const std::runtime_error& ex)
    {
      std::cerr << "Failed to load CUDA runtime and set device: "
                << ex.what() << std::endl;
      cuda_device_num_ = -1;
    }
  }

  ~RealCudaVoxelizationHelperInterface() override
  {
    CleanupAllocatedMemory();
  }

  bool IsAvailable() const override { return (cuda_device_num_ >= 0); }

  std::vector<int64_t> PrepareTrackingGrids(
      const int64_t num_cells, const int32_t num_grids) override
  {
    CleanupTrackingGridsMemory();
    const size_t tracking_grids_size =
        sizeof(int32_t) * 2 * num_cells * num_grids;
    cudaMalloc(&device_tracking_grids_ptr_, tracking_grids_size);
    CudaCheckErrors("Failed to allocate device tracking grids");
    cudaMemset(device_tracking_grids_ptr_, 0, tracking_grids_size);
    CudaCheckErrors("Failed to zero device tracking grids");
    std::vector<int64_t> tracking_grid_offsets(num_grids, 0);
    for (int32_t num_grid = 0; num_grid < num_grids; num_grid++)
    {
      tracking_grid_offsets.at(num_grid) = num_grid * num_cells * 2;
    }
    return tracking_grid_offsets;
  }

  void RaycastPoints(
      const std::vector<float>& raw_points,
      const float* const pointcloud_origin_transform,
      const float* const inverse_grid_origin_transform,
      const float inverse_step_size, const float inverse_cell_size,
      const int32_t num_x_cells, const int32_t num_y_cells,
      const int32_t num_z_cells,
      const int64_t tracking_grid_starting_offset) override
  {
    SetCudaDevice();
    const int32_t num_points = raw_points.size() / 3;
    // Copy the points
    const size_t points_size = sizeof(float) * raw_points.size();
    float* device_points_ptr = nullptr;
    cudaMalloc(&device_points_ptr, points_size);
    CudaCheckErrors("Failed to allocate device points");
    cudaMemcpy(device_points_ptr, raw_points.data(), points_size,
               cudaMemcpyHostToDevice);
    CudaCheckErrors("Failed to memcpy the points to the device");
    // Copy pointcloud origin transform
    const size_t transform_size = sizeof(float) * 16;
    float* device_pointcloud_origin_transform_ptr = nullptr;
    cudaMalloc(&device_pointcloud_origin_transform_ptr, transform_size);
    CudaCheckErrors("Failed to allocate device pointcloud origin transform");
    cudaMemcpy(
        device_pointcloud_origin_transform_ptr, pointcloud_origin_transform,
        transform_size, cudaMemcpyHostToDevice);
    CudaCheckErrors("Failed to memcpy the pointcloud origin transform");
    // Copy grid inverse origin transform
    float* device_tracking_grid_inverse_origin_transform_ptr = nullptr;
    cudaMalloc(
        &device_tracking_grid_inverse_origin_transform_ptr, transform_size);
    CudaCheckErrors("Failed to allocate device grid inverse origin transform");
    cudaMemcpy(
        device_tracking_grid_inverse_origin_transform_ptr,
        inverse_grid_origin_transform, transform_size, cudaMemcpyHostToDevice);
    CudaCheckErrors("Failed to memcpy the grid inverse origin transform");
    // Prepare for raycasting
    const int32_t stride1 = num_y_cells * num_z_cells;
    const int32_t stride2 = num_z_cells;
    // Call the CUDA kernel
    const int32_t num_threads = 256;
    const int32_t num_blocks = (num_points + (num_threads - 1)) / num_threads;
    int32_t* const device_tracking_grid_ptr =
        device_tracking_grids_ptr_ + tracking_grid_starting_offset;
    RaycastPoint<<<num_blocks, num_threads>>>(
        device_points_ptr, num_points, device_pointcloud_origin_transform_ptr,
        device_tracking_grid_inverse_origin_transform_ptr,
        inverse_step_size, inverse_cell_size, num_x_cells, num_y_cells,
        num_z_cells, stride1, stride2, device_tracking_grid_ptr);
    // Free the device memory
    cudaFree(device_points_ptr);
    CudaCheckErrors("Failed to free device points");
    cudaFree(device_pointcloud_origin_transform_ptr);
    CudaCheckErrors("Failed to free device pointcloud origin transform");
    cudaFree(device_tracking_grid_inverse_origin_transform_ptr);
    CudaCheckErrors(
        "Failed to free device tracking grid inverse origin tranform");
  }

  void PrepareFilterGrid(
       const int64_t num_cells, const void* host_data_ptr) override
  {
    CleanupFilterGridMemory();
    const size_t filter_grid_size = sizeof(float) * num_cells * 2;
    cudaMalloc(&device_filter_grid_ptr_, filter_grid_size);
    CudaCheckErrors("Failed to allocate device filter grid");
    cudaMemcpy(device_filter_grid_ptr_, host_data_ptr, filter_grid_size,
               cudaMemcpyHostToDevice);
    CudaCheckErrors("Failed to memcpy the static environment to the device");
  }

  void FilterTrackingGrids(
       const int64_t num_cells, const int32_t num_grids,
       const float percent_seen_free, const int32_t outlier_points_threshold,
       const int32_t num_cameras_seen_free) override
  {
    // Call the CUDA kernel
    const int32_t num_threads = 256;
    const int32_t num_blocks = (num_cells + (num_threads - 1)) / num_threads;
    FilterGrids<<<num_blocks, num_threads>>>(
        num_cells, num_grids, device_tracking_grids_ptr_,
        device_filter_grid_ptr_, percent_seen_free, outlier_points_threshold,
        num_cameras_seen_free);
  }

  void RetrieveTrackingGrid(
      const int64_t num_cells, const int64_t tracking_grid_starting_index,
      void* host_data_ptr) override
  {
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    const size_t tracking_grid_size = sizeof(int32_t) * num_cells * 2;
    cudaMemcpy(host_data_ptr,
               device_tracking_grids_ptr_ + tracking_grid_starting_index,
               tracking_grid_size, cudaMemcpyDeviceToHost);
    CudaCheckErrors("Failed to memcpy the tracking grid back to the host");
  }

  void RetrieveFilteredGrid(
      const int64_t num_cells, void* host_data_ptr) override
  {
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    const size_t filter_grid_size = sizeof(float) * num_cells * 2;
    cudaMemcpy(host_data_ptr, device_filter_grid_ptr_, filter_grid_size,
               cudaMemcpyDeviceToHost);
    CudaCheckErrors("Failed to memcpy the filter grid back to the host");
  }

  void CleanupAllocatedMemory() override
  {
    CleanupTrackingGridsMemory();
    CleanupFilterGridMemory();
  }

  void SetCudaDevice()
  {
    cudaSetDevice(cuda_device_num_);
    CudaCheckErrors("Failed to set device");
  }

private:
  void CleanupTrackingGridsMemory()
  {
    if (device_tracking_grids_ptr_ != nullptr)
    {
      cudaFree(device_tracking_grids_ptr_);
      CudaCheckErrors("Failed to free device tracking grids");
      device_tracking_grids_ptr_ = nullptr;
    }
  }

  void CleanupFilterGridMemory()
  {
    if (device_filter_grid_ptr_ != nullptr)
    {
      cudaFree(device_filter_grid_ptr_);
      CudaCheckErrors("Failed to free device filter grid");
      device_filter_grid_ptr_ = nullptr;
    }
  }

  int32_t cuda_device_num_ = -1;
  int32_t* device_tracking_grids_ptr_ = nullptr;
  float* device_filter_grid_ptr_ = nullptr;
};

CudaVoxelizationHelperInterface* MakeHelperInterface(
    const std::map<std::string, int32_t>& options)
{
  return new RealCudaVoxelizationHelperInterface(options);
}
}  // namespace cuda_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
