#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
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
    const float max_range,
    const float* const device_grid_pointcloud_transform_ptr,
    const float inverse_step_size, const float inverse_cell_size,
    const int32_t num_x_cells, const int32_t num_y_cells,
    const int32_t num_z_cells, const int32_t stride1, const int32_t stride2,
    int32_t* const device_tracking_grid_ptr)
{
  const int32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index < num_points)
  {
    // Point in pointcloud frame
    const float px = device_points_ptr[(point_index * 3) + 0];
    const float py = device_points_ptr[(point_index * 3) + 1];
    const float pz = device_points_ptr[(point_index * 3) + 2];
    // Skip invalid points marked with NaN or infinity
    if (isfinite(px) && isfinite(py) && isfinite(pz))
    {
      // Pointcloud origin in grid frame
      const float ox = device_grid_pointcloud_transform_ptr[12];
      const float oy = device_grid_pointcloud_transform_ptr[13];
      const float oz = device_grid_pointcloud_transform_ptr[14];
      // Point in grid frame
      const float gx = device_grid_pointcloud_transform_ptr[0] * px
                       + device_grid_pointcloud_transform_ptr[4] * py
                       + device_grid_pointcloud_transform_ptr[8] * pz
                       + device_grid_pointcloud_transform_ptr[12];
      const float gy = device_grid_pointcloud_transform_ptr[1] * px
                       + device_grid_pointcloud_transform_ptr[5] * py
                       + device_grid_pointcloud_transform_ptr[9] * pz
                       + device_grid_pointcloud_transform_ptr[13];
      const float gz = device_grid_pointcloud_transform_ptr[2] * px
                       + device_grid_pointcloud_transform_ptr[6] * py
                       + device_grid_pointcloud_transform_ptr[10] * pz
                       + device_grid_pointcloud_transform_ptr[14];
      const float rx = gx - ox;
      const float ry = gy - oy;
      const float rz = gz - oz;
      const float current_ray_length = sqrtf((rx * rx) + (ry * ry) + (rz * rz));
      const float num_steps = floor(current_ray_length * inverse_step_size);
      int32_t previous_x_cell = -1;
      int32_t previous_y_cell = -1;
      int32_t previous_z_cell = -1;
      bool ray_crossed_grid = false;
      for (float step = 0.0; step < num_steps; step += 1.0)
      {
        const float elapsed_ratio = step / num_steps;
        if ((elapsed_ratio * current_ray_length) > max_range)
        {
          // We've gone beyond max range of the sensor
          break;
        }
        const float qx = (rx * elapsed_ratio) + ox;
        const float qy = (ry * elapsed_ratio) + oy;
        const float qz = (rz * elapsed_ratio) + oz;
        const int32_t x_cell =
            static_cast<int32_t>(std::floor(qx * inverse_cell_size));
        const int32_t y_cell =
            static_cast<int32_t>(std::floor(qy * inverse_cell_size));
        const int32_t z_cell =
            static_cast<int32_t>(std::floor(qz * inverse_cell_size));
        if (x_cell != previous_x_cell || y_cell != previous_y_cell
            || z_cell != previous_z_cell)
        {
          if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0
              && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)
          {
            ray_crossed_grid = true;
            const int32_t cell_index =
                (x_cell * stride1) + (y_cell * stride2) + z_cell;
            // Increase free count
            atomicAdd(&(device_tracking_grid_ptr[cell_index * 2]), 1);
          }
          else if (ray_crossed_grid)
          {
            // We've left the grid and there's no reason to keep going.
            break;
          }
        }
        previous_x_cell = x_cell;
        previous_y_cell = y_cell;
        previous_z_cell = z_cell;
      }
      // Set the point itself as filled, if it is in range
      if (current_ray_length <= max_range)
      {
        const int32_t x_cell =
            static_cast<int32_t>(std::floor(gx * inverse_cell_size));
        const int32_t y_cell =
            static_cast<int32_t>(std::floor(gy * inverse_cell_size));
        const int32_t z_cell =
            static_cast<int32_t>(std::floor(gz * inverse_cell_size));
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

class CudaTrackingGridsHandle : public TrackingGridsHandle
{
public:
  CudaTrackingGridsHandle(
      int32_t* const tracking_grids_buffer,
      const std::vector<int64_t>& tracking_grid_starting_offsets,
      const int64_t num_cells_per_grid)
      : TrackingGridsHandle(tracking_grid_starting_offsets, num_cells_per_grid),
        tracking_grids_buffer_(tracking_grids_buffer)
  {
    if (tracking_grids_buffer_ == nullptr)
    {
      throw std::invalid_argument(
          "Cannot create CudaTrackingGridsHandle with null buffer");
    }
  }

  ~CudaTrackingGridsHandle() override
  {
    cudaFree(tracking_grids_buffer_);
    CudaCheckErrors("Failed to free device tracking grids buffer");
    tracking_grids_buffer_ = nullptr;
  }

  int32_t* GetBuffer() const { return tracking_grids_buffer_; }

private:
  int32_t* tracking_grids_buffer_ = nullptr;
};

class CudaFilterGridHandle : public FilterGridHandle
{
public:
  CudaFilterGridHandle(
      float* const filter_grid_buffer, const int64_t num_cells)
      : FilterGridHandle(num_cells), filter_grid_buffer_(filter_grid_buffer)
  {
    if (filter_grid_buffer_ == nullptr)
    {
      throw std::invalid_argument(
          "Cannot create CudaFilterGridHandle with null buffer");
    }
  }

  ~CudaFilterGridHandle() override
  {
    cudaFree(filter_grid_buffer_);
    CudaCheckErrors("Failed to free device filter grid buffer");
    filter_grid_buffer_ = nullptr;
  }

  float* GetBuffer() const { return filter_grid_buffer_; }

private:
  float* filter_grid_buffer_ = nullptr;
};

class CudaVoxelizationHelperInterface : public DeviceVoxelizationHelperInterface
{
public:
  explicit CudaVoxelizationHelperInterface(
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

        cudaDeviceProp device_properties;
        std::memset(&device_properties, 0, sizeof(device_properties));
        cudaGetDeviceProperties(&device_properties, cuda_device);
        CudaCheckErrors("Failed to get device properties");
        const std::string device_name(device_properties.name);

        std::cout << "Using CUDA device [" << cuda_device << "] - Name: ["
                  << device_name << "]" << std::endl;
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

  bool IsAvailable() const override { return (cuda_device_num_ >= 0); }

  std::unique_ptr<TrackingGridsHandle> PrepareTrackingGrids(
      const int64_t num_cells, const int32_t num_grids) override
  {
    const size_t tracking_grids_size =
        sizeof(int32_t) * 2 * num_cells * num_grids;
    int32_t* tracking_grids_buffer = nullptr;
    cudaMalloc(&tracking_grids_buffer, tracking_grids_size);
    CudaCheckErrors("Failed to allocate device tracking grids");
    cudaMemset(tracking_grids_buffer, 0, tracking_grids_size);
    CudaCheckErrors("Failed to zero device tracking grids");

    std::vector<int64_t> tracking_grid_offsets(num_grids, 0);
    for (int32_t num_grid = 0; num_grid < num_grids; num_grid++)
    {
      tracking_grid_offsets.at(num_grid) = num_grid * num_cells * 2;
    }

    return std::unique_ptr<TrackingGridsHandle>(
        new CudaTrackingGridsHandle(
            tracking_grids_buffer, tracking_grid_offsets, num_cells));
  }

  void RaycastPoints(
      const std::vector<float>& raw_points, const float max_range,
      const float* const grid_pointcloud_transform,
      const float inverse_step_size, const float inverse_cell_size,
      const int32_t num_x_cells, const int32_t num_y_cells,
      const int32_t num_z_cells, TrackingGridsHandle& tracking_grids,
      const size_t tracking_grid_index) override
  {
    CudaTrackingGridsHandle& real_tracking_grids =
        dynamic_cast<CudaTrackingGridsHandle&>(tracking_grids);

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

    // Copy grid pointcloud transform
    const size_t transform_size = sizeof(float) * 16;
    float* device_grid_pointcloud_transform_ptr = nullptr;
    cudaMalloc(&device_grid_pointcloud_transform_ptr, transform_size);
    CudaCheckErrors("Failed to allocate device grid pointcloud transform");
    cudaMemcpy(
        device_grid_pointcloud_transform_ptr, grid_pointcloud_transform,
        transform_size, cudaMemcpyHostToDevice);
    CudaCheckErrors("Failed to memcpy the grid pointcloud transform");

    // Prepare for raycasting
    const int32_t stride1 = num_y_cells * num_z_cells;
    const int32_t stride2 = num_z_cells;
    // Call the CUDA kernel
    const int32_t num_threads = 256;
    const int32_t num_blocks = (num_points + (num_threads - 1)) / num_threads;
    const size_t starting_index =
        real_tracking_grids.GetTrackingGridStartingOffset(tracking_grid_index);
    int32_t* const device_tracking_grid_ptr =
        real_tracking_grids.GetBuffer() + starting_index;
    RaycastPoint<<<num_blocks, num_threads>>>(
        device_points_ptr, num_points, max_range,
        device_grid_pointcloud_transform_ptr, inverse_step_size,
        inverse_cell_size, num_x_cells, num_y_cells, num_z_cells, stride1,
        stride2, device_tracking_grid_ptr);

    // Free the device memory
    cudaFree(device_points_ptr);
    CudaCheckErrors("Failed to free device points");
    cudaFree(device_grid_pointcloud_transform_ptr);
    CudaCheckErrors("Failed to free device grid pointcloud transform");
  }

  std::unique_ptr<FilterGridHandle> PrepareFilterGrid(
       const int64_t num_cells, const void* host_data_ptr) override
  {
    const size_t filter_grid_size = sizeof(float) * num_cells * 2;
    float* filter_grid_buffer = nullptr;
    cudaMalloc(&filter_grid_buffer, filter_grid_size);
    CudaCheckErrors("Failed to allocate device filter grid");
    cudaMemcpy(filter_grid_buffer, host_data_ptr, filter_grid_size,
               cudaMemcpyHostToDevice);
    CudaCheckErrors("Failed to memcpy the static environment to the device");

    return std::unique_ptr<FilterGridHandle>(new CudaFilterGridHandle(
        filter_grid_buffer, num_cells));
  }

  void FilterTrackingGrids(
      const TrackingGridsHandle& tracking_grids, const float percent_seen_free,
      const int32_t outlier_points_threshold,
      const int32_t num_cameras_seen_free,
      FilterGridHandle& filter_grid) override
  {
    const CudaTrackingGridsHandle& real_tracking_grids =
        dynamic_cast<const CudaTrackingGridsHandle&>(tracking_grids);
    CudaFilterGridHandle& real_filter_grid =
        dynamic_cast<CudaFilterGridHandle&>(filter_grid);

    // Call the CUDA kernel
    const int32_t num_threads = 256;
    const int32_t num_blocks =
        (real_tracking_grids.NumCellsPerGrid() + (num_threads - 1))
        / num_threads;
    FilterGrids<<<num_blocks, num_threads>>>(
        real_tracking_grids.NumCellsPerGrid(),
        static_cast<int32_t>(real_tracking_grids.GetNumTrackingGrids()),
        real_tracking_grids.GetBuffer(), real_filter_grid.GetBuffer(),
        percent_seen_free, outlier_points_threshold, num_cameras_seen_free);
  }

  void RetrieveTrackingGrid(
      const TrackingGridsHandle& tracking_grids,
      const size_t tracking_grid_index, void* host_data_ptr) override
  {
    const CudaTrackingGridsHandle& real_tracking_grids =
        dynamic_cast<const CudaTrackingGridsHandle&>(tracking_grids);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    const size_t item_size = sizeof(int32_t) * 2;
    const size_t tracking_grid_size =
        real_tracking_grids.NumCellsPerGrid() * item_size;
    const size_t starting_index =
        real_tracking_grids.GetTrackingGridStartingOffset(tracking_grid_index);
    cudaMemcpy(host_data_ptr,
               real_tracking_grids.GetBuffer() + starting_index,
               tracking_grid_size, cudaMemcpyDeviceToHost);
    CudaCheckErrors("Failed to memcpy the tracking grid back to the host");
  }

  void RetrieveFilteredGrid(
      const FilterGridHandle& filter_grid, void* host_data_ptr) override
  {
    const CudaFilterGridHandle& real_filter_grid =
        dynamic_cast<const CudaFilterGridHandle&>(filter_grid);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    const size_t item_size = sizeof(float) * 2;
    const size_t buffer_size = real_filter_grid.NumCells() * item_size;
    cudaMemcpy(host_data_ptr, real_filter_grid.GetBuffer(), buffer_size,
               cudaMemcpyDeviceToHost);
    CudaCheckErrors("Failed to memcpy the filter grid back to the host");
  }

  void SetCudaDevice()
  {
    cudaSetDevice(cuda_device_num_);
    CudaCheckErrors("Failed to set device");
  }

private:
  int32_t cuda_device_num_ = -1;
};

std::vector<AvailableDevice> GetAvailableDevices()
{
  std::vector<AvailableDevice> available_devices;

  try
  {
    int32_t device_count = 0;
    cudaGetDeviceCount(&device_count);
    CudaCheckErrors("Failed to get device count");

    for (int32_t device_idx = 0; device_idx < device_count; device_idx++)
    {
      cudaDeviceProp device_properties;
      std::memset(&device_properties, 0, sizeof(device_properties));
      cudaGetDeviceProperties(&device_properties, device_idx);
      CudaCheckErrors("Failed to get device properties");
      const std::string device_name(device_properties.name);
      const std::string full_name = "CUDA - Device: [" + device_name + "]";

      std::map<std::string, int32_t> device_options;
      device_options["CUDA_DEVICE"] = device_idx;

      available_devices.push_back(AvailableDevice(full_name, device_options));
    }
  }
  catch (const std::runtime_error& ex)
  {
    std::cerr << ex.what() << std::endl;
  }

  return available_devices;
}

std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeCudaVoxelizationHelper(const std::map<std::string, int32_t>& options)
{
  return std::unique_ptr<DeviceVoxelizationHelperInterface>(
      new CudaVoxelizationHelperInterface(options));
}
}  // namespace cuda_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
