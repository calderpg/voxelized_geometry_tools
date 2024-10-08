#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <voxelized_geometry_tools/cl.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
namespace
{
const char* kRaycastPointKernelCode = R"(
void kernel RaycastPoint(
    global const float* points, const float max_range,
    global const float* grid_pointcloud_transform,
    const float inverse_step_size, const float inverse_cell_size,
    const int stride1, const int stride2, const int num_x_cells,
    const int num_y_cells, const int num_z_cells, global int* tracking_grid,
    const int tracking_grid_starting_offset)
{
  const int point_index = get_global_id(0);
  // Point in pointcloud frame
  const float px = points[(point_index * 3) + 0];
  const float py = points[(point_index * 3) + 1];
  const float pz = points[(point_index * 3) + 2];
  // Skip invalid points marked with NaN or infinity
  if (isfinite(px) && isfinite(py) && isfinite(pz))
  {
    // Pointcloud origin in grid frame
    const float ox = grid_pointcloud_transform[12];
    const float oy = grid_pointcloud_transform[13];
    const float oz = grid_pointcloud_transform[14];
    // Point in grid frame
    const float gx = grid_pointcloud_transform[0] * px
                     + grid_pointcloud_transform[4] * py
                     + grid_pointcloud_transform[8] * pz
                     + grid_pointcloud_transform[12];
    const float gy = grid_pointcloud_transform[1] * px
                     + grid_pointcloud_transform[5] * py
                     + grid_pointcloud_transform[9] * pz
                     + grid_pointcloud_transform[13];
    const float gz = grid_pointcloud_transform[2] * px
                     + grid_pointcloud_transform[6] * py
                     + grid_pointcloud_transform[10] * pz
                     + grid_pointcloud_transform[14];
    const float rx = gx - ox;
    const float ry = gy - oy;
    const float rz = gz - oz;
    const float current_ray_length = sqrt((rx * rx) + (ry * ry) + (rz * rz));
    const float num_steps = floor(current_ray_length * inverse_step_size);
    int previous_x_cell = -1;
    int previous_y_cell = -1;
    int previous_z_cell = -1;
    bool ray_crossed_grid = false;
    for (float step = 0.0f; step < num_steps; step += 1.0f)
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
      const int x_cell = (int)floor(qx * inverse_cell_size);
      const int y_cell = (int)floor(qy * inverse_cell_size);
      const int z_cell = (int)floor(qz * inverse_cell_size);
      if (x_cell != previous_x_cell || y_cell != previous_y_cell
          || z_cell != previous_z_cell)
      {
        if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0
           && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)
        {
          ray_crossed_grid = true;
          const int cell_index =
              (x_cell * stride1) + (y_cell * stride2) + z_cell;
          const int tracking_grid_index =
              tracking_grid_starting_offset + (cell_index * 2);
          atomic_add(&(tracking_grid[tracking_grid_index]), 1);
        }
        else if (ray_crossed_grid)
        {
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
      const int x_cell = (int)floor(gx * inverse_cell_size);
      const int y_cell = (int)floor(gy * inverse_cell_size);
      const int z_cell = (int)floor(gz * inverse_cell_size);
      if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0
          && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)
      {
        const int cell_index = (x_cell * stride1) + (y_cell * stride2) + z_cell;
        const int tracking_grid_index =
            tracking_grid_starting_offset + (cell_index * 2);
        atomic_add(&(tracking_grid[tracking_grid_index + 1]), 1);
      }
    }
  }
}
)";

const char* kFilterGridsKernelCode = R"(
void kernel FilterGrids(
    const int num_cells, const int num_grids, global const int* tracking_grid,
    global float* filter_grid, const float percent_seen_free,
    const int outlier_points_threshold, const int num_cameras_seen_free)
{
  const int voxel_index = get_global_id(0);
  const int filter_grid_index = voxel_index * 2;
  const float current_occupancy = filter_grid[filter_grid_index];
  if (current_occupancy <= 0.5f)
  {
    int cameras_seen_filled = 0;
    int cameras_seen_free = 0;
    for (int idx = 0; idx < num_grids; idx++)
    {
      const int tracking_grid_offset = num_cells * 2 * idx;
      const int tracking_grid_index =
          tracking_grid_offset + filter_grid_index;
      const int free_count = tracking_grid[tracking_grid_index];
      const int filled_count = tracking_grid[tracking_grid_index + 1];
      const int filtered_filled_count =
          (filled_count >= outlier_points_threshold) ? filled_count : 0;
      if (free_count > 0 && filtered_filled_count > 0)
      {
        const float current_percent_seen_free =
            (float)(free_count) / (float)(free_count + filtered_filled_count);
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
      filter_grid[filter_grid_index] = 1.0f;
    }
    else if (cameras_seen_free >= num_cameras_seen_free)
    {
      filter_grid[filter_grid_index] = 0.0f;
    }
    else
    {
      filter_grid[filter_grid_index] = 0.5f;
    }
  }
}
)";

static std::string GetRaycastingKernelCode()
{
  return std::string(kRaycastPointKernelCode);
}

static std::string GetFilterKernelCode()
{
  return std::string(kFilterGridsKernelCode);
}

std::string LogOpenCLError(const cl_int error_code)
{
  return "[error code " + std::to_string(error_code) + "]";
}

class OpenCLTrackingGridsHandle : public TrackingGridsHandle
{
public:
  OpenCLTrackingGridsHandle(
      std::unique_ptr<cl::Buffer> tracking_grids_buffer,
      const std::vector<int64_t>& tracking_grid_starting_offsets,
      const int64_t num_cells_per_grid)
      : TrackingGridsHandle(tracking_grid_starting_offsets, num_cells_per_grid),
        tracking_grids_buffer_(std::move(tracking_grids_buffer))
  {
    if (!tracking_grids_buffer_)
    {
      throw std::invalid_argument(
          "Cannot create OpenCLTrackingGridsHandle with null buffer");
    }
  }

  cl::Buffer& GetBuffer() { return *tracking_grids_buffer_; }

  const cl::Buffer& GetBuffer() const { return *tracking_grids_buffer_; }

private:
  std::unique_ptr<cl::Buffer> tracking_grids_buffer_;
};

class OpenCLFilterGridHandle : public FilterGridHandle
{
public:
  OpenCLFilterGridHandle(
      std::unique_ptr<cl::Buffer> filter_grid_buffer,
      const int64_t num_cells)
      : FilterGridHandle(num_cells),
        filter_grid_buffer_(std::move(filter_grid_buffer))
  {
    if (!filter_grid_buffer_)
    {
      throw std::invalid_argument(
          "Cannot create OpenCLFilterGridHandle with null buffer");
    }
  }

  cl::Buffer& GetBuffer() { return *filter_grid_buffer_; }

  const cl::Buffer& GetBuffer() const { return *filter_grid_buffer_; }

private:
  std::unique_ptr<cl::Buffer> filter_grid_buffer_;
};

class OpenCLVoxelizationHelperInterface
    : public DeviceVoxelizationHelperInterface
{
public:
  OpenCLVoxelizationHelperInterface(
      const std::map<std::string, int32_t>& options,
      const LoggingFunction& logging_fn)
  {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    const int32_t platform_index = RetrieveOptionOrDefault(
        options, "OPENCL_PLATFORM_INDEX", 0, logging_fn);
    if (all_platforms.size() > 0 && platform_index >= 0
        && platform_index < static_cast<int32_t>(all_platforms.size()))
    {
      auto& opencl_platform =
          all_platforms.at(static_cast<size_t>(platform_index));

      std::string platform_name;
      opencl_platform.getInfo(CL_PLATFORM_NAME, &platform_name);
      std::string platform_vendor;
      opencl_platform.getInfo(CL_PLATFORM_VENDOR, &platform_vendor);

      if (logging_fn)
      {
        logging_fn(
            "Using OpenCL Platform [" + std::to_string(platform_index) +
            "] - Name: [" + platform_name + "], Vendor: [" + platform_vendor +
            "]");
      }

      std::vector<cl::Device> all_devices;
      opencl_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

      const int32_t device_index = RetrieveOptionOrDefault(
          options, "OPENCL_DEVICE_INDEX", 0, logging_fn);
      if (all_devices.size() > 0 && device_index >= 0
          && device_index < static_cast<int32_t>(all_devices.size()))
      {
        auto& opencl_device = all_devices.at(static_cast<size_t>(device_index));

        std::string device_name;
        opencl_device.getInfo(CL_DEVICE_NAME, &device_name);

        if (logging_fn)
        {
          logging_fn(
              "Using OpenCL Device [" + std::to_string(device_index) +
              "] - Name: [" + device_name + "]");
        }

        // Make context + queue
        context_ = std::unique_ptr<cl::Context>(
            new cl::Context({opencl_device}));
        queue_ = std::unique_ptr<cl::CommandQueue>(
            new cl::CommandQueue(*context_, opencl_device));
        // Make kernel programs
        const std::string build_options =
            "-Werror -cl-unsafe-math-optimizations";
        cl::Program::Sources raycasting_sources;
        const std::string raycasting_kernel_source = GetRaycastingKernelCode();
        raycasting_sources.push_back({raycasting_kernel_source.c_str(),
                                      raycasting_kernel_source.length()});
        raycasting_program_ = std::unique_ptr<cl::Program>(
            new cl::Program(*context_, raycasting_sources));
        cl::Program::Sources filter_sources;
        const std::string filter_kernel_source = GetFilterKernelCode();
        filter_sources.push_back({filter_kernel_source.c_str(),
                                  filter_kernel_source.length()});
        filter_program_ = std::unique_ptr<cl::Program>(
            new cl::Program(*context_, filter_sources));
        if (raycasting_program_->build({opencl_device}, build_options.c_str())
            != CL_SUCCESS)
        {
          if (logging_fn)
          {
            logging_fn(
                "Error building raycasting kernel: " +
                raycasting_program_->getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    opencl_device));
          }
          raycasting_program_.reset();
        }
        if (filter_program_->build({opencl_device}, build_options.c_str())
            != CL_SUCCESS)
        {
          if (logging_fn)
          {
            logging_fn(
                "Error building filter kernel: " +
                filter_program_->getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    opencl_device));
          }
          filter_program_.reset();
        }
      }
      else if (all_devices.size() > 0)
      {
        if (logging_fn)
        {
          logging_fn(
              "OPENCL_DEVICE_INDEX = " + std::to_string(device_index) +
              " out of range for " + std::to_string(all_devices.size()) +
              " devices");
        }
      }
      else
      {
        if (logging_fn)
        {
          logging_fn("No OpenCL device available");
        }
      }
    }
    else if (all_platforms.size() > 0)
    {
      if (logging_fn)
      {
        logging_fn(
            "OPENCL_PLATFORM_INDEX = " + std::to_string(platform_index) +
            " out of range for " + std::to_string(all_platforms.size()) +
            " platforms");
      }
    }
    else
    {
      if (logging_fn)
      {
        logging_fn("No OpenCL platform available");
      }
    }
  }

  bool IsAvailable() const override
  {
    return (context_ && queue_ && raycasting_program_ && filter_program_);
  }

  std::unique_ptr<TrackingGridsHandle> PrepareTrackingGrids(
      const int64_t num_cells, const int32_t num_grids) override
  {
    const size_t buffer_elements =
        static_cast<size_t>(num_cells * num_grids) * 2;
    const size_t buffer_size = sizeof(int32_t) * buffer_elements;
    cl_int err = 0;

    // enqueueFillBuffer behavior appears to be broken on macOS. Instead of the
    // allocate + fill approach taken on Linux, use an allocate + copy approach.
#if defined(__APPLE__)
    std::vector<int32_t> fill_data(buffer_elements, 0);
    std::unique_ptr<cl::Buffer> tracking_grids_buffer(new cl::Buffer(
        *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_size,
        const_cast<void*>(static_cast<const void*>(fill_data.data())), &err));
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to allocate tracking grid buffer: " + LogOpenCLError(err));
    }
#else
    std::unique_ptr<cl::Buffer> tracking_grids_buffer(new cl::Buffer(
        *context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err));
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to allocate tracking grid buffer: " + LogOpenCLError(err));
    }

    // This is how we zero the buffer
    err = queue_->enqueueFillBuffer<int32_t>(
        *tracking_grids_buffer, 0, 0, buffer_size, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to enqueueFillBuffer: " + LogOpenCLError(err));
    }

    err = queue_->finish();
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to complete enqueueFillBuffer: " + LogOpenCLError(err));
    }
#endif

    // Calculate offsets into the buffer
    std::vector<int64_t> tracking_grid_offsets(
        static_cast<size_t>(num_grids), 0);
    for (int32_t num_grid = 0; num_grid < num_grids; num_grid++)
    {
      tracking_grid_offsets.at(static_cast<size_t>(num_grid)) =
          num_grid * num_cells * 2;
    }
    return std::unique_ptr<TrackingGridsHandle>(new OpenCLTrackingGridsHandle(
        std::move(tracking_grids_buffer), tracking_grid_offsets, num_cells));
  }

  void RaycastPoints(
      const std::vector<float>& raw_points, const float max_range,
      const float* const grid_pointcloud_transform,
      const float inverse_step_size, const float inverse_cell_size,
      const int32_t num_x_cells, const int32_t num_y_cells,
      const int32_t num_z_cells, TrackingGridsHandle& tracking_grids,
      const size_t tracking_grid_index) override
  {
    OpenCLTrackingGridsHandle& real_tracking_grids =
        dynamic_cast<OpenCLTrackingGridsHandle&>(tracking_grids);

    cl_int err = 0;

    cl::Buffer device_points_buffer(
        *context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * raw_points.size(),
        const_cast<void*>(static_cast<const void*>(raw_points.data())),
        &err);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to allocate and copy pointcloud: " + LogOpenCLError(err));
    }

    cl::Buffer device_grid_pointcloud_transform_buffer(
        *context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 16,
        const_cast<void*>(static_cast<const void*>(grid_pointcloud_transform)),
        &err);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to allocate and copy grid pointcloud transform: "
          + LogOpenCLError(err));
    }

    const int32_t stride1 = num_y_cells * num_z_cells;
    const int32_t stride2 = num_z_cells;
    const int32_t starting_index = static_cast<int32_t>(
        real_tracking_grids.GetTrackingGridStartingOffset(tracking_grid_index));
    // Build kernel
    cl::Kernel raycasting_kernel(*raycasting_program_, "RaycastPoint");
    raycasting_kernel.setArg(0, device_points_buffer);
    raycasting_kernel.setArg(1, max_range);
    raycasting_kernel.setArg(2, device_grid_pointcloud_transform_buffer);
    raycasting_kernel.setArg(3, inverse_step_size);
    raycasting_kernel.setArg(4, inverse_cell_size);
    raycasting_kernel.setArg(5, stride1);
    raycasting_kernel.setArg(6, stride2);
    raycasting_kernel.setArg(7, num_x_cells);
    raycasting_kernel.setArg(8, num_y_cells);
    raycasting_kernel.setArg(9, num_z_cells);
    raycasting_kernel.setArg(10, real_tracking_grids.GetBuffer());
    raycasting_kernel.setArg(11, starting_index);
    err = queue_->enqueueNDRangeKernel(
        raycasting_kernel, cl::NullRange, cl::NDRange(raw_points.size() / 3),
        cl::NullRange);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to enqueue raycasting kernel: " + LogOpenCLError(err));
    }
  }

  std::unique_ptr<FilterGridHandle> PrepareFilterGrid(
      const int64_t num_cells, const void* host_data_ptr) override
  {
    const size_t filter_grid_buffer_size =
        sizeof(float) * static_cast<size_t>(num_cells) * 2;
    cl_int err = 0;
    std::unique_ptr<cl::Buffer> filter_grid_buffer(new cl::Buffer(
        *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        filter_grid_buffer_size, const_cast<void*>(host_data_ptr), &err));
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to allocate and copy filtered buffer: "
          + LogOpenCLError(err));
    }

    return std::unique_ptr<FilterGridHandle>(new OpenCLFilterGridHandle(
        std::move(filter_grid_buffer), num_cells));
  }

  void FilterTrackingGrids(
      const TrackingGridsHandle& tracking_grids, const float percent_seen_free,
      const int32_t outlier_points_threshold,
      const int32_t num_cameras_seen_free,
      FilterGridHandle& filter_grid) override
  {
    const OpenCLTrackingGridsHandle& real_tracking_grids =
        dynamic_cast<const OpenCLTrackingGridsHandle&>(tracking_grids);
    OpenCLFilterGridHandle& real_filter_grid =
        dynamic_cast<OpenCLFilterGridHandle&>(filter_grid);

    // Build kernel
    cl::Kernel filter_kernel(*filter_program_, "FilterGrids");
    filter_kernel.setArg(
        0, static_cast<int32_t>(real_tracking_grids.NumCellsPerGrid()));
    filter_kernel.setArg(
        1, static_cast<int32_t>(real_tracking_grids.GetNumTrackingGrids()));
    filter_kernel.setArg(2, real_tracking_grids.GetBuffer());
    filter_kernel.setArg(3, real_filter_grid.GetBuffer());
    filter_kernel.setArg(4, percent_seen_free);
    filter_kernel.setArg(5, outlier_points_threshold);
    filter_kernel.setArg(6, num_cameras_seen_free);
    const cl_int err = queue_->enqueueNDRangeKernel(
        filter_kernel, cl::NullRange,
        cl::NDRange(static_cast<size_t>(real_tracking_grids.NumCellsPerGrid())),
        cl::NullRange);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to enqueue filter kernel: " + LogOpenCLError(err));
    }
  }

  void RetrieveTrackingGrid(
      const TrackingGridsHandle& tracking_grids,
      const size_t tracking_grid_index, void* host_data_ptr) override
  {
    const OpenCLTrackingGridsHandle& real_tracking_grids =
        dynamic_cast<const OpenCLTrackingGridsHandle&>(tracking_grids);

    cl_int err = queue_->finish();
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "RetrieveTrackingGrid finish failed: " + LogOpenCLError(err));
    }

    const size_t item_size = sizeof(int32_t) * 2;
    const size_t tracking_grid_size =
        static_cast<size_t>(real_tracking_grids.NumCellsPerGrid()) * item_size;
    const size_t starting_offset =
        static_cast<size_t>(real_tracking_grids.GetTrackingGridStartingOffset(
            tracking_grid_index))
        * sizeof(int32_t);
    err = queue_->enqueueReadBuffer(
        real_tracking_grids.GetBuffer(), CL_TRUE, starting_offset,
        tracking_grid_size, host_data_ptr);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "RetrieveTrackingGrid enqueueReadBuffer failed: "
          + LogOpenCLError(err));
    }
  }

  void RetrieveFilteredGrid(
      const FilterGridHandle& filter_grid, void* host_data_ptr) override
  {
    const OpenCLFilterGridHandle& real_filter_grid =
        dynamic_cast<const OpenCLFilterGridHandle&>(filter_grid);

    cl_int err = queue_->finish();
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "RetrieveFilteredGrid finish failed: " + LogOpenCLError(err));
    }

    const size_t item_size = sizeof(float) * 2;
    const size_t buffer_size =
        static_cast<size_t>(real_filter_grid.NumCells()) * item_size;
    err = queue_->enqueueReadBuffer(
        real_filter_grid.GetBuffer(), CL_TRUE, 0, buffer_size, host_data_ptr);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "RetrieveFilteredGrid enqueueReadBuffer failed: "
          + LogOpenCLError(err));
    }
  }

private:
  std::unique_ptr<cl::Context> context_;
  std::unique_ptr<cl::CommandQueue> queue_;
  std::unique_ptr<cl::Program> raycasting_program_;
  std::unique_ptr<cl::Program> filter_program_;
};
}  // namespace

std::vector<AvailableDevice> GetAvailableDevices()
{
  std::vector<AvailableDevice> available_devices;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (size_t platform_idx = 0; platform_idx < platforms.size(); platform_idx++)
  {
    const auto& platform = platforms.at(platform_idx);
    std::string platform_name;
    platform.getInfo(CL_PLATFORM_NAME, &platform_name);
    std::string platform_vendor;
    platform.getInfo(CL_PLATFORM_VENDOR, &platform_vendor);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    for (size_t device_idx = 0; device_idx < devices.size(); device_idx++)
    {
      const auto& device = devices.at(device_idx);
      std::string device_name;
      device.getInfo(CL_DEVICE_NAME, &device_name);

      const std::string full_name =
          "OpenCL - Platform name: [" + platform_name + "] Vendor: ["
          + platform_vendor + "] Device: [" + device_name + "]";

      std::map<std::string, int32_t> device_options;
      device_options["OPENCL_PLATFORM_INDEX"] =
          static_cast<int32_t>(platform_idx);
      device_options["OPENCL_DEVICE_INDEX"] = static_cast<int32_t>(device_idx);

      available_devices.push_back(AvailableDevice(full_name, device_options));
    }
  }

  return available_devices;
}

std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeOpenCLVoxelizationHelper(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn)
{
  return std::unique_ptr<DeviceVoxelizationHelperInterface>(
      new OpenCLVoxelizationHelperInterface(options, logging_fn));
}
}  // namespace opencl_helpers
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
