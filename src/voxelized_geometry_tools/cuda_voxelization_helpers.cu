#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
bool IsAvailable() { return true; }

#define CudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__
void RaycastPoint(
    const float* const device_points_ptr, const int32_t num_points,
    const float* const device_pointcloud_origin_transform_ptr,
    const float* const device_tracking_grid_inverse_origin_transform_ptr,
    const float inverse_cell_size, const int32_t num_x_cells,
    const int32_t num_y_cells, const int32_t num_z_cells, const int32_t stride1,
    const int32_t stride2, int32_t* const device_tracking_grid_ptr)
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
    const float step_size = (1.0 / inverse_cell_size) * 0.5f;
    const float rx = wx - ox;
    const float ry = wy - oy;
    const float rz = wz - oz;
    const float current_ray_length = sqrt((rx * rx) + (ry * ry) + (rz * rz));
    const float num_steps = floor(current_ray_length / step_size);
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
    const int64_t num_cells, const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs,
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
      for (int32_t idx = 0; idx < num_device_tracking_grids; idx++)
      {
        int32_t* const device_tracking_grid_ptr =
            device_tracking_grid_ptrs[idx];
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

int32_t* PrepareTrackingGrid(const int64_t num_cells)
{
  const size_t tracking_grid_size = sizeof(int32_t) * num_cells * 2;
  int32_t* device_tracking_grid_ptr = nullptr;
  cudaMalloc(&device_tracking_grid_ptr, tracking_grid_size);
  CudaCheckErrors("Failed to allocate device tracking grid");
  cudaMemset(device_tracking_grid_ptr, 0, tracking_grid_size);
  CudaCheckErrors("Failed to zero device tracking grid");
  return device_tracking_grid_ptr;
}

void RaycastPoints(
    const float* const points, const int32_t num_points,
    const float* const pointcloud_origin_transform,
    const float* const inverse_grid_origin_transform,
    const float inverse_cell_size, const int32_t num_x_cells,
    const int32_t num_y_cells, const int32_t num_z_cells,
    int32_t* const device_tracking_grid_ptr)
{
  // Copy points
  const size_t points_size = sizeof(float) * num_points * 3;
  float* device_points_ptr = nullptr;
  cudaMalloc(&device_points_ptr, points_size);
  CudaCheckErrors("Failed to allocate device points");
  cudaMemcpy(device_points_ptr, points, points_size, 
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
  RaycastPoint<<<num_blocks, num_threads>>>(
      device_points_ptr, num_points, device_pointcloud_origin_transform_ptr,
      device_tracking_grid_inverse_origin_transform_ptr, inverse_cell_size,
      num_x_cells, num_y_cells, num_z_cells, stride1, stride2,
      device_tracking_grid_ptr);
  cudaDeviceSynchronize();
  // Free the device memory
  cudaFree(device_points_ptr);
  CudaCheckErrors("Failed to free device points");
  cudaFree(device_pointcloud_origin_transform_ptr);
  CudaCheckErrors("Failed to free device pointcloud origin transform");
  cudaFree(device_tracking_grid_inverse_origin_transform_ptr);
  CudaCheckErrors(
      "Failed to free device tracking grid inverse origin tranform");
}

float* PrepareFilterGrid(const int64_t num_cells, const void* host_data_ptr)
{
  const size_t filter_grid_size = sizeof(float) * num_cells * 2;
  float* device_filter_grid_ptr = nullptr;
  cudaMalloc(&device_filter_grid_ptr, filter_grid_size);
  CudaCheckErrors("Failed to allocate device filter grid");
  cudaMemcpy(device_filter_grid_ptr, host_data_ptr, filter_grid_size, 
             cudaMemcpyHostToDevice);
  CudaCheckErrors("Failed to memcpy the static environment to the device");
  return device_filter_grid_ptr;
}

void FilterTrackingGrids(
    const int64_t num_cells, const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs,
    float* const device_filter_grid_ptr, const float percent_seen_free,
    const int32_t outlier_points_threshold, const int32_t num_cameras_seen_free)
{
  const size_t device_tracking_grid_ptrs_size =
      sizeof(int32_t*) * num_device_tracking_grids;
  int32_t** device_tracking_grid_ptrs_ptr = nullptr;
  cudaMalloc(&device_tracking_grid_ptrs_ptr, device_tracking_grid_ptrs_size);
  CudaCheckErrors("Failed to allocate device tracking grid ptr storage");
  cudaMemcpy(device_tracking_grid_ptrs_ptr, device_tracking_grid_ptrs,
             device_tracking_grid_ptrs_size, cudaMemcpyHostToDevice);
  CudaCheckErrors("Failed to memcpy the device tracking grid ptrs to device");
  // Call the CUDA kernel
  cudaDeviceSynchronize();
  const int32_t num_threads = 256;
  const int32_t num_blocks = (num_cells + (num_threads - 1)) / num_threads;
  FilterGrids<<<num_blocks, num_threads>>>(
      num_cells, num_device_tracking_grids, device_tracking_grid_ptrs_ptr,
      device_filter_grid_ptr, percent_seen_free, outlier_points_threshold,
      num_cameras_seen_free);
  // Free the device memory
  cudaFree(device_tracking_grid_ptrs_ptr);
  CudaCheckErrors("Failed to free device tracking grid ptr storage");
}

void RetrieveTrackingGrid(
    const int64_t num_cells, const int32_t* const device_tracking_grid_ptr,
    void* host_data_ptr)
{
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  const size_t tracking_grid_size = sizeof(int32_t) * num_cells * 2;
  cudaMemcpy(host_data_ptr, device_tracking_grid_ptr, tracking_grid_size,
             cudaMemcpyDeviceToHost);
  CudaCheckErrors("Failed to memcpy the tracking grid back to the host");
}

void RetrieveFilteredGrid(
    const int64_t num_cells, const float* const device_filter_grid_ptr,
    void* host_data_ptr)
{
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  const size_t filter_grid_size = sizeof(float) * num_cells * 2;
  cudaMemcpy(host_data_ptr, device_filter_grid_ptr, filter_grid_size,
             cudaMemcpyDeviceToHost);
  CudaCheckErrors("Failed to memcpy the filter grid back to the host");
}

void CleanupDeviceMemory(
    const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs, float* device_filter_grid_ptr)
{
  for (int32_t idx = 0; idx < num_device_tracking_grids; idx++)
  {
    auto device_tracking_grid_ptr = device_tracking_grid_ptrs[idx];
    // Free the device memory
    cudaFree(device_tracking_grid_ptr);
    CudaCheckErrors("Failed to free device tracking grid");
  }
  // Free the device memory
  cudaFree(device_filter_grid_ptr);
  CudaCheckErrors("Failed to free device filter grid");
}
}
}
}
