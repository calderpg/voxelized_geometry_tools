#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
namespace
{
constexpr int32_t kDefaultThreadsPerBlock = 256;

void CudaCheckErrors(const cudaError_t error, const std::string& msg)
{
  if (error != cudaSuccess)
  {
    const std::string error_string(cudaGetErrorString(error));
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
    const float current_occupancy = device_filter_grid_ptr[voxel_index];
    // Filled cells stay filled, we don't work with them.
    // We only change cells that are unknown or empty.
    if (current_occupancy <= 0.5)
    {
      int32_t cameras_seen_filled = 0;
      int32_t cameras_seen_free = 0;
      for (int32_t idx = 0; idx < num_grids; idx++)
      {
        const int32_t tracking_grid_offset = num_cells * 2 * idx;
        const int32_t tracking_grid_index =
            tracking_grid_offset + (voxel_index * 2);
        const int32_t free_count =
            device_tracking_grids_ptr[tracking_grid_index];
        const int32_t filled_count =
            device_tracking_grids_ptr[tracking_grid_index + 1];
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
        device_filter_grid_ptr[voxel_index] = 1.0;
      }
      else if (cameras_seen_free >= num_cameras_seen_free)
      {
        // Did enough cameras see this empty?
        device_filter_grid_ptr[voxel_index] = 0.0;
      }
      else
      {
        // Otherwise, it is unknown.
        device_filter_grid_ptr[voxel_index] = 0.5;
      }
    }
  }
}

template<typename T>
class CudaBuffer
{
public:
  explicit CudaBuffer(const CudaBuffer<T>&) = delete;
  CudaBuffer<T>& operator=(const CudaBuffer<T>&) = delete;

  explicit CudaBuffer(CudaBuffer<T>&& other)
  {
    buffer_ = other.buffer_;
    other.buffer_ = nullptr;
  }

  CudaBuffer<T>& operator=(CudaBuffer<T>&& other)
  {
    if (this != std::addressof(other))
    {
      Clear();

      buffer_ = other.buffer_;
      other.buffer_ = nullptr;
    }
    return *this;
  }

  CudaBuffer() = default;

  explicit CudaBuffer(const size_t num_elements)
  {
    if (num_elements == 0)
    {
      throw std::runtime_error("num_elements must be > 0");
    }

    const size_t buffer_size = sizeof(T) * num_elements;
    buffer_ = AllocateBuffer(buffer_size);

    const cudaError_t memset_error = cudaMemset(buffer_, 0, buffer_size);
    if (memset_error != cudaSuccess)
    {
      FreeBuffer();
      CudaCheckErrors(memset_error, "Failed to zero resource for CudaBuffer");
    }
  }

  CudaBuffer(const size_t num_elements, const void* const to_copy)
  {
    if (num_elements == 0)
    {
      throw std::runtime_error("num_elements must be > 0");
    }
    if (to_copy == nullptr)
    {
      throw std::runtime_error("to_copy cannot be nullptr");
    }

    const size_t buffer_size = sizeof(T) * num_elements;
    buffer_ = AllocateBuffer(buffer_size);

    const cudaError_t memcpy_error =
        cudaMemcpy(buffer_, to_copy, buffer_size, cudaMemcpyHostToDevice);
    if (memcpy_error != cudaSuccess)
    {
      FreeBuffer();
      CudaCheckErrors(memcpy_error, "Failed to memcpy resource to CudaBuffer");
    }
  }

  ~CudaBuffer() { FreeBuffer(); }

  void Clear() { FreeBuffer(); }

  T* Get() const { return buffer_; }

private:
  static T* AllocateBuffer(const size_t buffer_size)
  {
    T* buffer = nullptr;
    const cudaError_t malloc_error = cudaMalloc(&buffer, buffer_size);
    CudaCheckErrors(malloc_error, "Failed to allocate resource for CudaBuffer");
    return buffer;
  }

  void FreeBuffer()
  {
    if (buffer_ != nullptr)
    {
      CudaCheckErrors(
          cudaFree(buffer_),
          "Failed to free resource held by CudaBuffer");
      buffer_ = nullptr;
    }
  }

  T* buffer_ = nullptr;
};

class CudaTrackingGridsHandle : public TrackingGridsHandle
{
public:
  CudaTrackingGridsHandle(
      CudaBuffer<int32_t>&& tracking_grids_buffer,
      const std::vector<int64_t>& tracking_grid_starting_offsets,
      const int64_t num_cells_per_grid)
      : TrackingGridsHandle(tracking_grid_starting_offsets, num_cells_per_grid),
        tracking_grids_buffer_(std::move(tracking_grids_buffer)) {}

  ~CudaTrackingGridsHandle() override = default;

  int32_t* GetBuffer() const { return tracking_grids_buffer_.Get(); }

private:
  CudaBuffer<int32_t> tracking_grids_buffer_;
};

class CudaFilterGridHandle : public FilterGridHandle
{
public:
  CudaFilterGridHandle(
      CudaBuffer<float>&& filter_grid_buffer, const int64_t num_cells)
      : FilterGridHandle(num_cells),
        filter_grid_buffer_(std::move(filter_grid_buffer)) {}

  ~CudaFilterGridHandle() override = default;

  float* GetBuffer() const { return filter_grid_buffer_.Get(); }

private:
  CudaBuffer<float> filter_grid_buffer_;
};

class CudaVoxelizationHelperInterface : public DeviceVoxelizationHelperInterface
{
public:
  CudaVoxelizationHelperInterface(
      const std::map<std::string, int32_t>& options,
      const LoggingFunction& logging_fn)
  {
    const int32_t cuda_threads_per_block = RetrieveOptionOrDefault(
        options, "CUDA_THREADS_PER_BLOCK", -1, logging_fn);
    if (cuda_threads_per_block > 0)
    {
      cuda_threads_per_block_ = cuda_threads_per_block;
      if (logging_fn)
      {
        logging_fn(
            "Set CUDA threads per block to specified " +
            std::to_string(cuda_threads_per_block_));
      }
    }
    else
    {
      cuda_threads_per_block_ = kDefaultThreadsPerBlock;
      if (logging_fn)
      {
        logging_fn(
            "Set CUDA threads per block to default " +
            std::to_string(cuda_threads_per_block_));
      }
    }

    const int32_t cuda_device = RetrieveOptionOrDefault(
        options, "CUDA_DEVICE", 0, logging_fn);
    try
    {
      int32_t device_count = 0;
      CudaCheckErrors(
          cudaGetDeviceCount(&device_count),
          "Failed to get device count");
      if (cuda_device >= 0 && cuda_device < device_count)
      {
        cuda_device_num_ = cuda_device;
        SetCudaDevice();

        cudaDeviceProp device_properties;
        std::memset(&device_properties, 0, sizeof(device_properties));
        CudaCheckErrors(
            cudaGetDeviceProperties(&device_properties, cuda_device),
            "Failed to get device properties");
        const std::string device_name(device_properties.name);

        if (logging_fn)
        {
          logging_fn(
              "Using CUDA device [" + std::to_string(cuda_device) +
              "] - Name: [" + device_name + "]");
        }
      }
      else
      {
        if (logging_fn)
        {
          logging_fn(
              "CUDA_DEVICE = " + std::to_string(cuda_device) +
              " out of range for " + std::to_string(device_count) + " devices");
        }
        cuda_device_num_ = -1;
      }
    }
    catch (const std::runtime_error& ex)
    {
      if (logging_fn)
      {
        logging_fn(
            "Failed to load CUDA runtime and set device: " +
            std::string(ex.what()));
      }
      cuda_device_num_ = -1;
    }
  }

  bool IsAvailable() const override { return (cuda_device_num_ >= 0); }

  std::unique_ptr<TrackingGridsHandle> PrepareTrackingGrids(
      const int64_t num_cells, const int32_t num_grids) override
  {
    const size_t tracking_grids_elements =
        2 * static_cast<size_t>(num_cells * num_grids);

    std::vector<int64_t> tracking_grid_offsets(
        static_cast<size_t>(num_grids), 0);
    for (int32_t num_grid = 0; num_grid < num_grids; num_grid++)
    {
      tracking_grid_offsets.at(static_cast<size_t>(num_grid)) =
          num_grid * num_cells * 2;
    }

    return std::unique_ptr<TrackingGridsHandle>(new CudaTrackingGridsHandle(
        CudaBuffer<int32_t>(tracking_grids_elements), tracking_grid_offsets,
        num_cells));
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
    const int32_t num_points = static_cast<int32_t>(raw_points.size() / 3);

    // Copy the points to the device
    const CudaBuffer<float> device_points(raw_points.size(), raw_points.data());

    // Copy grid pointcloud transform to the device
    const CudaBuffer<float> device_grid_pointcloud_transform(
        16, grid_pointcloud_transform);

    // Prepare for raycasting
    const int32_t stride1 = num_y_cells * num_z_cells;
    const int32_t stride2 = num_z_cells;
    // Call the CUDA kernel
    const int32_t num_threads = CudaThreadsPerBlock();
    const int32_t num_blocks = CalcNumBlocks(num_points);
    const size_t starting_index =
        real_tracking_grids.GetTrackingGridStartingOffset(tracking_grid_index);
    int32_t* const device_tracking_grid_ptr =
        real_tracking_grids.GetBuffer() + starting_index;
    RaycastPoint<<<num_blocks, num_threads>>>(
        device_points.Get(), num_points, max_range,
        device_grid_pointcloud_transform.Get(), inverse_step_size,
        inverse_cell_size, num_x_cells, num_y_cells, num_z_cells, stride1,
        stride2, device_tracking_grid_ptr);
  }

  std::unique_ptr<FilterGridHandle> PrepareFilterGrid(
       const int64_t num_cells, const void* host_data_ptr) override
  {
    const size_t filter_grid_elements = static_cast<size_t>(num_cells);

    return std::unique_ptr<FilterGridHandle>(new CudaFilterGridHandle(
        CudaBuffer<float>(filter_grid_elements, host_data_ptr), num_cells));
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
    const int32_t num_threads = CudaThreadsPerBlock();
    const int32_t num_blocks = CalcNumBlocks(
        static_cast<int32_t>(real_tracking_grids.NumCellsPerGrid()));
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
    CudaCheckErrors(
        cudaDeviceSynchronize(),
        "RetrieveTrackingGrid failed to synchronize");
    const size_t item_size = sizeof(int32_t) * 2;
    const size_t tracking_grid_size =
        real_tracking_grids.NumCellsPerGrid() * item_size;
    const size_t starting_index =
        real_tracking_grids.GetTrackingGridStartingOffset(tracking_grid_index);
    CudaCheckErrors(
        cudaMemcpy(
            host_data_ptr,
            real_tracking_grids.GetBuffer() + starting_index,
            tracking_grid_size, cudaMemcpyDeviceToHost),
        "Failed to memcpy the tracking grid back to the host");
  }

  void RetrieveFilteredGrid(
      const FilterGridHandle& filter_grid, void* host_data_ptr) override
  {
    const CudaFilterGridHandle& real_filter_grid =
        dynamic_cast<const CudaFilterGridHandle&>(filter_grid);

    // Wait for GPU to finish before accessing on host
    CudaCheckErrors(
        cudaDeviceSynchronize(),
        "RetrieveFilteredGrid failed to synchronize");
    const size_t item_size = sizeof(float);
    const size_t buffer_size = real_filter_grid.NumVoxels() * item_size;
    CudaCheckErrors(
        cudaMemcpy(
            host_data_ptr, real_filter_grid.GetBuffer(), buffer_size,
            cudaMemcpyDeviceToHost),
        "Failed to memcpy the filter grid back to the host");
  }

  void SetCudaDevice()
  {
    CudaCheckErrors(
        cudaSetDevice(cuda_device_num_),
        "Failed to set device");
  }

private:
  int32_t CalcNumBlocks(const int32_t num_items) const
  {
    const int32_t num_threads = CudaThreadsPerBlock();
    const int32_t num_blocks = (num_items + (num_threads - 1)) / num_threads;
    return num_blocks;
  }

  int32_t CudaThreadsPerBlock() const { return cuda_threads_per_block_; }

  int32_t cuda_threads_per_block_ = 0;
  int32_t cuda_device_num_ = -1;
};
}  // namespace

std::vector<AvailableDevice> GetAvailableDevices()
{
  std::vector<AvailableDevice> available_devices;

  try
  {
    int32_t device_count = 0;
    CudaCheckErrors(
        cudaGetDeviceCount(&device_count),
        "Failed to get device count");

    for (int32_t device_idx = 0; device_idx < device_count; device_idx++)
    {
      cudaDeviceProp device_properties;
      std::memset(&device_properties, 0, sizeof(device_properties));
      CudaCheckErrors(
          cudaGetDeviceProperties(&device_properties, device_idx),
          "Failed to get device properties");
      const std::string device_name(device_properties.name);
      const std::string full_name = "CUDA - Device: [" + device_name + "]";

      std::map<std::string, int32_t> device_options;
      device_options["CUDA_DEVICE"] = device_idx;

      available_devices.push_back(AvailableDevice(full_name, device_options));
    }
  }
  catch (const std::runtime_error&) {}

  return available_devices;
}

std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeCudaVoxelizationHelper(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
{
  return std::unique_ptr<DeviceVoxelizationHelperInterface>(
      new CudaVoxelizationHelperInterface(options, logging_fn));
}
}  // namespace cuda_helpers
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
