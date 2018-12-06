#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Geometry>
#include <CL/cl.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
static std::string GetRaycastingKernelCode()
{
  const std::string kernel_code =
      "void kernel RaycastPoint("
      "    global const float* points,"
      "    global const float* pointcloud_origin_transform,"
      "    global const float* inverse_grid_origin_transform,"
      "    const float inverse_step_size, const float inverse_cell_size,"
      "    const int stride1, const int stride2, const int num_x_cells,"
      "    const int num_y_cells, const int num_z_cells,"
      "    global int* tracking_grid, const int tracking_grid_starting_offset)"
      "{"
      "  const int point_index = get_global_id(0);"
      "  const float ox = pointcloud_origin_transform[12];"
      "  const float oy = pointcloud_origin_transform[13];"
      "  const float oz = pointcloud_origin_transform[14];"
      "  const float px = points[(point_index * 3) + 0];"
      "  const float py = points[(point_index * 3) + 1];"
      "  const float pz = points[(point_index * 3) + 2];"
      "  const float wx = pointcloud_origin_transform[0] * px"
      "                   + pointcloud_origin_transform[4] * py"
      "                   + pointcloud_origin_transform[8] * pz"
      "                   + pointcloud_origin_transform[12];"
      "  const float wy = pointcloud_origin_transform[1] * px"
      "                   + pointcloud_origin_transform[5] * py"
      "                   + pointcloud_origin_transform[9] * pz"
      "                   + pointcloud_origin_transform[13];"
      "  const float wz = pointcloud_origin_transform[2] * px"
      "                   + pointcloud_origin_transform[6] * py"
      "                   + pointcloud_origin_transform[10] * pz"
      "                   + pointcloud_origin_transform[14];"
      "  const float rx = wx - ox;"
      "  const float ry = wy - oy;"
      "  const float rz = wz - oz;"
      "  const float current_ray_length ="
      "      sqrt((rx * rx) + (ry * ry) + (rz * rz));"
      "  const float num_steps = floor(current_ray_length * inverse_step_size);"
      "  int previous_x_cell = -1;"
      "  int previous_y_cell = -1;"
      "  int previous_z_cell = -1;"
      "  for (float step = 0.0; step < num_steps; step += 1.0)"
      "  {"
      "    bool in_grid = false;"
      "    const float elapsed_ratio = step / num_steps;"
      "    const float cx = (rx * elapsed_ratio) + ox;"
      "    const float cy = (ry * elapsed_ratio) + oy;"
      "    const float cz = (rz * elapsed_ratio) + oz;"
      "    const float gx ="
      "        inverse_grid_origin_transform[0] * cx"
      "        + inverse_grid_origin_transform[4] * cy"
      "        + inverse_grid_origin_transform[8] * cz"
      "        + inverse_grid_origin_transform[12];"
      "    const float gy ="
      "        inverse_grid_origin_transform[1] * cx"
      "        + inverse_grid_origin_transform[5] * cy"
      "        + inverse_grid_origin_transform[9] * cz"
      "        + inverse_grid_origin_transform[13];"
      "    const float gz ="
      "        inverse_grid_origin_transform[2] * cx"
      "        + inverse_grid_origin_transform[6] * cy"
      "        + inverse_grid_origin_transform[10] * cz"
      "        + inverse_grid_origin_transform[14];"
      "    const int x_cell = (int)(gx * inverse_cell_size);"
      "    const int y_cell = (int)(gy * inverse_cell_size);"
      "    const int z_cell = (int)(gz * inverse_cell_size);"
      "    if (x_cell != previous_x_cell || y_cell != previous_y_cell"
      "        || z_cell != previous_z_cell)"
      "    {"
      "      if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0"
      "         && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)"
      "      {"
      "        in_grid = true;"
      "        const int cell_index ="
      "            (x_cell * stride1) + (y_cell * stride2) + z_cell;"
      "        const int tracking_grid_index ="
      "            tracking_grid_starting_offset + (cell_index * 2);"
      "        atomic_add(&(tracking_grid[tracking_grid_index]), 1);"
      "      }"
      "      else if (in_grid)"
      "      {"
      "        break;"
      "      }"
      "    }"
      "    previous_x_cell = x_cell;"
      "    previous_y_cell = y_cell;"
      "    previous_z_cell = z_cell;"
      "  }"
      "  const float gx ="
      "      inverse_grid_origin_transform[0] * wx"
      "      + inverse_grid_origin_transform[4] * wy"
      "      + inverse_grid_origin_transform[8] * wz"
      "      + inverse_grid_origin_transform[12];"
      "  const float gy ="
      "      inverse_grid_origin_transform[1] * wx"
      "      + inverse_grid_origin_transform[5] * wy"
      "      + inverse_grid_origin_transform[9] * wz"
      "      + inverse_grid_origin_transform[13];"
      "  const float gz ="
      "      inverse_grid_origin_transform[2] * wx"
      "      + inverse_grid_origin_transform[6] * wy"
      "      + inverse_grid_origin_transform[10] * wz"
      "      + inverse_grid_origin_transform[14];"
      "  const int x_cell = (int)(gx * inverse_cell_size);"
      "  const int y_cell = (int)(gy * inverse_cell_size);"
      "  const int z_cell = (int)(gz * inverse_cell_size);"
      "  if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0"
      "      && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)"
      "  {"
      "    const int cell_index ="
      "        (x_cell * stride1) + (y_cell * stride2) + z_cell;"
      "    const int tracking_grid_index ="
      "        tracking_grid_starting_offset + (cell_index * 2);"
      "    atomic_add(&(tracking_grid[tracking_grid_index + 1]), 1);"
      "  }"
      "}";
  return kernel_code;
}

static std::string GetFilterKernelCode()
{
  const std::string kernel_code =
      "void kernel FilterGrids("
      "    const int num_cells, const int num_grids,"
      "    global const int* tracking_grid,"
      "    global float* filter_grid,"
      "    const float percent_seen_free, const int outlier_points_threshold,"
      "    const int num_cameras_seen_free)"
      "{"
      "  const int voxel_index = get_global_id(0);"
      "  const int filter_grid_index = voxel_index * 2;"
      "  const float current_occupancy = filter_grid[filter_grid_index];"
      "  if (current_occupancy <= 0.5)"
      "  {"
      "    int cameras_seen_filled = 0;"
      "    int cameras_seen_free = 0;"
      "    for (int idx = 0; idx < num_grids; idx++)"
      "    {"
      "      const int tracking_grid_offset = num_cells * 2 * idx;"
      "      const int tracking_grid_index ="
      "          tracking_grid_offset + filter_grid_index;"
      "      const int free_count = tracking_grid[tracking_grid_index];"
      "      const int filled_count ="
      "          tracking_grid[tracking_grid_index + 1];"
      "      const int filtered_filled_count ="
      "          (filled_count >= outlier_points_threshold) ? filled_count : 0;"
      "      if (free_count > 0 && filtered_filled_count > 0)"
      "      {"
      "        const float current_percent_seen_free ="
      "            (float)(free_count)"
      "            / (float)(free_count + filtered_filled_count);"
      "        if (current_percent_seen_free >= percent_seen_free)"
      "        {"
      "          cameras_seen_free += 1;"
      "        }"
      "        else"
      "        {"
      "          cameras_seen_filled += 1;"
      "        }"
      "      }"
      "      else if (free_count > 0)"
      "      {"
      "        cameras_seen_free += 1;"
      "      }"
      "      else if (filtered_filled_count > 0)"
      "      {"
      "        cameras_seen_filled += 1;"
      "      }"
      "    }"
      "    if (cameras_seen_filled > 0)"
      "    {"
      "      filter_grid[filter_grid_index] = 1.0;"
      "    }"
      "    else if (cameras_seen_free >= num_cameras_seen_free)"
      "    {"
      "      filter_grid[filter_grid_index] = 0.0;"
      "    }"
      "    else"
      "    {"
      "      filter_grid[filter_grid_index] = 0.5;"
      "    }"
      "  }"
      "}";
  return kernel_code;
}

class RealOpenCLVoxelizationHelperInterface
    : public OpenCLVoxelizationHelperInterface
{
public:
  RealOpenCLVoxelizationHelperInterface()
  {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() > 0)
    {
      auto& default_platform = all_platforms.front();
      std::vector<cl::Device> all_devices;
      default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
      if (all_devices.size() > 0)
      {
        auto& default_device = all_devices.front();
        // Make context + queue
        context_ = std::unique_ptr<cl::Context>(
            new cl::Context({default_device}));
        queue_ = std::unique_ptr<cl::CommandQueue>(
            new cl::CommandQueue(*context_, default_device));
        // Make kernel programs
        const std::string build_options = "-Werror -cl-fast-relaxed-math";
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
        if (raycasting_program_->build({default_device}, build_options.c_str())
            != CL_SUCCESS)
        {
          std::cerr << " Error building raycasting kernel: "
                    << raycasting_program_->getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                        default_device)
                    << std::endl;
          raycasting_program_.reset();
        }
        if (filter_program_->build({default_device}, build_options.c_str())
            != CL_SUCCESS)
        {
          std::cerr << " Error building filter kernel: "
                    << filter_program_->getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                        default_device)
                    << std::endl;
          filter_program_.reset();
        }
      }
      else
      {
        std::cerr << "No OpenCL device available" << std::endl;
      }
    }
    else
    {
      std::cerr << "No OpenCL platform available" << std::endl;
    }
  }

  bool IsAvailable() const override
  {
    return (context_ && queue_ && raycasting_program_ && filter_program_);
  }

  std::vector<int64_t> PrepareTrackingGrids(
      const int64_t num_cells, const int32_t num_grids) override
  {
    const size_t buffer_size = sizeof(int32_t) * 2 * num_cells * num_grids;
    cl_int err = 0;
    device_tracking_grid_buffer_ = std::unique_ptr<cl::Buffer>(new cl::Buffer(
        *context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err));
    if (err == 0)
    {
      // This is how we zero the buffer
      cl::Event event;
      err = queue_->enqueueFillBuffer<int32_t>(
          *device_tracking_grid_buffer_, 0, 0, buffer_size, nullptr, &event);
      if (err == CL_SUCCESS)
      {
        err = event.wait();
        if (err == CL_SUCCESS)
        {
          std::vector<int64_t> tracking_grid_offsets(num_grids, 0);
          for (int32_t num_grid = 0; num_grid < num_grids; num_grid++)
          {
            tracking_grid_offsets.at(num_grid) = num_grid * num_cells * 2;
          }
          return tracking_grid_offsets;
        }
        else
        {
          throw std::runtime_error("Failed to wait for event");
        }
      }
      else
      {
        throw std::runtime_error("Failed to enqueueFillBuffer");
      }
    }
    else
    {
      throw std::runtime_error("Failed to allocate tracking grid buffer");
    }
  }

  void RaycastPoints(
      const std::vector<float>& raw_points,
      const Eigen::Isometry3f& pointcloud_origin_transform,
      const Eigen::Isometry3f& inverse_grid_origin_transform,
      const float inverse_step_size, const float inverse_cell_size,
      const int32_t num_x_cells, const int32_t num_y_cells,
      const int32_t num_z_cells,
      const int64_t tracking_grid_starting_offset) override
  {
    cl_int err = 0;
    cl::Buffer device_points_buffer(
        *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * raw_points.size(),
        const_cast<void*>(static_cast<const void*>(raw_points.data())),
        &err);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Failed to allocate and copy pointcloud");
    }
    cl::Buffer device_pointcloud_origin_transform_buffer(
        *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 16,
        const_cast<void*>(static_cast<const void*>(
            pointcloud_origin_transform.data())),
        &err);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Failed to allocate and copy cloud transform");
    }
    cl::Buffer device_inverse_grid_origin_transform_buffer(
        *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 16,
        const_cast<void*>(static_cast<const void*>(
            inverse_grid_origin_transform.data())),
        &err);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Failed to allocate and copy grid transform");
    }
    const int32_t stride1 = num_y_cells * num_z_cells;
    const int32_t stride2 = num_z_cells;
    // Build kernel
    cl::Kernel raycasting_kernel(*raycasting_program_, "RaycastPoint");
    raycasting_kernel.setArg(0, device_points_buffer);
    raycasting_kernel.setArg(1, device_pointcloud_origin_transform_buffer);
    raycasting_kernel.setArg(2, device_inverse_grid_origin_transform_buffer);
    raycasting_kernel.setArg(3, inverse_step_size);
    raycasting_kernel.setArg(4, inverse_cell_size);
    raycasting_kernel.setArg(5, stride1);
    raycasting_kernel.setArg(6, stride2);
    raycasting_kernel.setArg(7, num_x_cells);
    raycasting_kernel.setArg(8, num_y_cells);
    raycasting_kernel.setArg(9, num_z_cells);
    raycasting_kernel.setArg(10, *device_tracking_grid_buffer_);
    raycasting_kernel.setArg(
        11, static_cast<int32_t>(tracking_grid_starting_offset));
    err = queue_->enqueueNDRangeKernel(
        raycasting_kernel, cl::NullRange, cl::NDRange(raw_points.size() / 3),
        cl::NullRange);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Failed to enqueue raycasting kernel");
    }
  }

  bool PrepareFilterGrid(
      const int64_t num_cells, const void* host_data_ptr) override
  {
    cl_int err = 0;
    device_filter_grid_buffer_ = std::unique_ptr<cl::Buffer>(new cl::Buffer(
        *context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * num_cells * 2, const_cast<void*>(host_data_ptr), &err));
    if (err == CL_SUCCESS)
    {
      return true;
    }
    else
    {
      throw std::runtime_error("Failed to allocate and copy filtered buffer");
    }
  }

  void FilterTrackingGrids(
       const int64_t num_cells, const int32_t num_grids,
       const float percent_seen_free, const int32_t outlier_points_threshold,
       const int32_t num_cameras_seen_free) override
  {
    // Build kernel
    cl::Kernel filter_kernel(*filter_program_, "FilterGrids");
    filter_kernel.setArg(0, static_cast<int32_t>(num_cells));
    filter_kernel.setArg(1, num_grids);
    filter_kernel.setArg(2, *device_tracking_grid_buffer_);
    filter_kernel.setArg(3, *device_filter_grid_buffer_);
    filter_kernel.setArg(4, percent_seen_free);
    filter_kernel.setArg(5, outlier_points_threshold);
    filter_kernel.setArg(6, num_cameras_seen_free);
    const cl_int err = queue_->enqueueNDRangeKernel(
        filter_kernel, cl::NullRange, cl::NDRange(num_cells), cl::NullRange);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error(
          "Failed to enqueue filter kernel [" + std::to_string(err) + "]");
    }
  }

  void RetrieveTrackingGrid(
      const int64_t num_cells,
      const int64_t device_tracking_grid_starting_index,
      void* host_data_ptr) override
  {
    queue_->finish();
    const size_t item_size = sizeof(int32_t) * 2;
    const size_t buffer_size = num_cells * item_size;
    const size_t starting_offset =
        device_tracking_grid_starting_index * sizeof(int32_t);
    const cl_int err = queue_->enqueueReadBuffer(
        *device_tracking_grid_buffer_, CL_TRUE, starting_offset, buffer_size,
        host_data_ptr);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Tracking buffer enqueueReadBuffer failed");
    }
  }

  void RetrieveFilteredGrid(
      const int64_t num_cells, void* host_data_ptr) override
  {
    queue_->finish();
    const size_t buffer_size = sizeof(float) * num_cells * 2;
    const cl_int err = queue_->enqueueReadBuffer(
        *device_filter_grid_buffer_, CL_TRUE, 0, buffer_size, host_data_ptr);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Filtered buffer enqueueReadBuffer failed");
    }
  }

  void CleanupAllocatedMemory() override
  {
    device_tracking_grid_buffer_.reset();
    device_filter_grid_buffer_.reset();
  }

private:
  std::unique_ptr<cl::Context> context_;
  std::unique_ptr<cl::CommandQueue> queue_;
  std::unique_ptr<cl::Program> raycasting_program_;
  std::unique_ptr<cl::Program> filter_program_;
  std::unique_ptr<cl::Buffer> device_tracking_grid_buffer_;
  std::unique_ptr<cl::Buffer> device_filter_grid_buffer_;
};

OpenCLVoxelizationHelperInterface* MakeHelperInterface()
{
  return new RealOpenCLVoxelizationHelperInterface();
}
}  // namespace opencl_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools

