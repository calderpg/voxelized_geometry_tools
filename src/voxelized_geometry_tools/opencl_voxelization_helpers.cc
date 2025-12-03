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
int GetStepFromDiff(const int diff)
{
  if (diff > 0)
  {
    return 1;
  }
  else if (diff < 0)
  {
    return -1;
  }
  else
  {
    return 0;
  }
}

float GetAxisTValue(
    const float point_axis, const float ray_axis,
    const float voxel_min_axis, const float voxel_max_axis)
{
  if (ray_axis > 0.0)
  {
    const float max_within_voxel = voxel_max_axis - point_axis;
    return fabs(max_within_voxel / ray_axis);
  }
  else if (ray_axis < -0.0)
  {
    const float max_within_voxel = point_axis - voxel_min_axis;
    return fabs(max_within_voxel / ray_axis);
  }
  else
  {
    return INFINITY;
  }
}

void kernel RaycastPoint(
    global const float* points, const float max_range,
    global const float* grid_pointcloud_transform,
    const float voxel_size, const float inverse_voxel_size,
    const float grid_x_size, const float grid_y_size, const float grid_z_size,
    const int num_x_voxels, const int num_y_voxels, const int num_z_voxels,
    const int stride1, const int stride2, global int* tracking_grid,
    const int tracking_grid_starting_offset)
{
  const int point_index = get_global_id(0);

  // Point in pointcloud frame.
  const float p_CP_x = points[(point_index * 3) + 0];
  const float p_CP_y = points[(point_index * 3) + 1];
  const float p_CP_z = points[(point_index * 3) + 2];

  // Bail if the point is non-finite.
  if (!isfinite(p_CP_x) || !isfinite(p_CP_y) || !isfinite(p_CP_z))
  {
    return;
  }

  // Transform point into grid frame.
  const float p_GP_x = grid_pointcloud_transform[0] * p_CP_x +
                       grid_pointcloud_transform[4] * p_CP_y +
                       grid_pointcloud_transform[8] * p_CP_z +
                       grid_pointcloud_transform[12];
  const float p_GP_y = grid_pointcloud_transform[1] * p_CP_x +
                       grid_pointcloud_transform[5] * p_CP_y +
                       grid_pointcloud_transform[9] * p_CP_z +
                       grid_pointcloud_transform[13];
  const float p_GP_z = grid_pointcloud_transform[2] * p_CP_x +
                       grid_pointcloud_transform[6] * p_CP_y +
                       grid_pointcloud_transform[10] * p_CP_z +
                       grid_pointcloud_transform[14];

  // Get pointcloud origin in grid frame.
  const float p_GCo_x = grid_pointcloud_transform[12];
  const float p_GCo_y = grid_pointcloud_transform[13];
  const float p_GCo_z = grid_pointcloud_transform[14];

  // Step 1: limit the final point to the provided maximum range.
  const float ray_x = p_GP_x - p_GCo_x;
  const float ray_y = p_GP_y - p_GCo_y;
  const float ray_z = p_GP_z - p_GCo_z;
  const float ray_length = sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
  const bool clipped = ray_length > max_range;

  float p_GFinal_x = p_GP_x;
  float p_GFinal_y = p_GP_y;
  float p_GFinal_z = p_GP_z;
  if (clipped)
  {
    p_GFinal_x = p_GCo_x + (ray_x * (max_range / ray_length));
    p_GFinal_y = p_GCo_y + (ray_y * (max_range / ray_length));
    p_GFinal_z = p_GCo_z + (ray_z * (max_range / ray_length));
  }

  // Step 2: get a starting point within the grid.
  const int p_GCo_idx = (int)floor(p_GCo_x * inverse_voxel_size);
  const int p_GCo_idy = (int)floor(p_GCo_y * inverse_voxel_size);
  const int p_GCo_idz = (int)floor(p_GCo_z * inverse_voxel_size);

  const bool origin_in_grid =
      p_GCo_idx >= 0 && p_GCo_idx < num_x_voxels &&
      p_GCo_idy >= 0 && p_GCo_idy < num_y_voxels &&
      p_GCo_idz >= 0 && p_GCo_idz < num_z_voxels;

  float p_GStart_x = p_GCo_x;
  float p_GStart_y = p_GCo_y;
  float p_GStart_z = p_GCo_z;
  if (!origin_in_grid)
  {
    float grid_sizes[3];
    grid_sizes[0] = grid_x_size;
    grid_sizes[1] = grid_y_size;
    grid_sizes[2] = grid_z_size;

    float p_GCo[3];
    p_GCo[0] = p_GCo_x;
    p_GCo[1] = p_GCo_y;
    p_GCo[2] = p_GCo_z;

    float tmin = 0.0f;
    float tmax = max_range;

    float direction[3];
    direction[0] = ray_x / ray_length;
    direction[1] = ray_y / ray_length;
    direction[2] = ray_z / ray_length;

    // Threshold for considering an axis direction as flat.
    const float flat_threshold = 1e-10;

    for (int axis = 0; axis < 3; axis++)
    {
      if (fabs(direction[axis]) < flat_threshold)
      {
        // If the direction verctor is nearly zero, make sure it is within the
        // axis range of the grid; if not, terminate.
        const bool in_slab =
            p_GCo[axis] >= 0.0f && p_GCo[axis] < grid_sizes[axis];

        if (!in_slab)
        {
          return;
        }
      }
      else
      {
        // Check against the low and high planes of the current axis.
        const float ood = 1.0f / direction[axis];

        const float tlow = (0.0 - p_GCo[axis]) * ood;
        const float thigh = (grid_sizes[axis] - p_GCo[axis]) * ood;

        const float t1 = (tlow <= thigh) ? tlow : thigh;
        const float t2 = (tlow <= thigh) ? thigh : tlow;

        if (t1 > tmin)
        {
          tmin = t1;
        }
        if (t2 > tmax)
        {
          tmax = t2;
        }

        if (tmin > tmax)
        {
          // Line segment does not interset the grid, terminate.
          return;
        }
      }
    }

    // Nudge the point slightly farther into the grid to avoid any edge cases.
    const float nudge = 1e-10;

    p_GStart_x = p_GCo_x + (direction[0] * (tmin + nudge));
    p_GStart_y = p_GCo_y + (direction[1] * (tmin + nudge));
    p_GStart_z = p_GCo_z + (direction[2] * (tmin + nudge));
  }

  // Step 3: grab indices for start and final points.
  const int p_GStart_idx = (int)floor(p_GStart_x * inverse_voxel_size);
  const int p_GStart_idy = (int)floor(p_GStart_y * inverse_voxel_size);
  const int p_GStart_idz = (int)floor(p_GStart_z * inverse_voxel_size);

  const int p_GFinal_idx = (int)floor(p_GFinal_x * inverse_voxel_size);
  const int p_GFinal_idy = (int)floor(p_GFinal_y * inverse_voxel_size);
  const int p_GFinal_idz = (int)floor(p_GFinal_z * inverse_voxel_size);

  // Step 4: get axis steps.
  const int x_step = GetStepFromDiff(p_GFinal_idx - p_GStart_idx);
  const int y_step = GetStepFromDiff(p_GFinal_idy - p_GStart_idy);
  const int z_step = GetStepFromDiff(p_GFinal_idz - p_GStart_idz);

  // Step 5: compute the control values.
  const float half_voxel_size = voxel_size * 0.5f;

  const float p_GStart_idcx = ((float)(p_GStart_idx) + 0.5f) * voxel_size;
  const float p_GStart_idcy = ((float)(p_GStart_idy) + 0.5f) * voxel_size;
  const float p_GStart_idcz = ((float)(p_GStart_idz) + 0.5f) * voxel_size;

  const float voxel_bottom_corner_x = p_GStart_idcx - half_voxel_size;
  const float voxel_bottom_corner_y = p_GStart_idcy - half_voxel_size;
  const float voxel_bottom_corner_z = p_GStart_idcz - half_voxel_size;

  const float voxel_top_corner_x = p_GStart_idcx + half_voxel_size;
  const float voxel_top_corner_y = p_GStart_idcy + half_voxel_size;
  const float voxel_top_corner_z = p_GStart_idcz + half_voxel_size;

  const float tx_initial = GetAxisTValue(
      p_GStart_x, ray_x, voxel_bottom_corner_x, voxel_top_corner_x);
  const float ty_initial = GetAxisTValue(
      p_GStart_y, ray_y, voxel_bottom_corner_y, voxel_top_corner_y);
  const float tz_initial = GetAxisTValue(
      p_GStart_z, ray_z, voxel_bottom_corner_z, voxel_top_corner_z);

  const float delta_tx = fabs(voxel_size / ray_x);
  const float delta_ty = fabs(voxel_size / ray_y);
  const float delta_tz = fabs(voxel_size / ray_z);

  // Step 6: set the final point.
  if (p_GFinal_idx >= 0 && p_GFinal_idx < num_x_voxels &&
      p_GFinal_idy >= 0 && p_GFinal_idy < num_y_voxels &&
      p_GFinal_idz >= 0 && p_GFinal_idz < num_z_voxels)
  {
    const int data_index =
        (p_GFinal_idx * stride1) + (p_GFinal_idy * stride2) + p_GFinal_idz;
    const int tracking_grid_index =
        tracking_grid_starting_offset + (data_index * 2);
    if (clipped)
    {
      // If the actual point was clipped, mark the final voxel as seen-free.
      atomic_add(&(tracking_grid[tracking_grid_index + 0]), 1);
    }
    else
    {
      // If the point was not clipped, mark the final voxel as seen-filled.
      atomic_add(&(tracking_grid[tracking_grid_index + 1]), 1);
    }
  }

  // Iterate along line.
  int current_idx = p_GStart_idx;
  int current_idy = p_GStart_idy;
  int current_idz = p_GStart_idz;

  float tx = tx_initial;
  float ty = ty_initial;
  float tz = tz_initial;

  while (current_idx != p_GFinal_idx ||
         current_idy != p_GFinal_idy ||
         current_idz != p_GFinal_idz)
  {
    // Update the current voxel.
    if (current_idx >= 0 && current_idx < num_x_voxels &&
        current_idy >= 0 && current_idy < num_y_voxels &&
        current_idz >= 0 && current_idz < num_z_voxels)
    {
      const int data_index =
          (current_idx * stride1) + (current_idy * stride2) + current_idz;
      const int tracking_grid_index =
          tracking_grid_starting_offset + (data_index * 2);
      // If the query is in bounds, update the seen-free count.
      atomic_add(&(tracking_grid[tracking_grid_index + 0]), 1);
    }
    else
    {
      // If the query is out of bounds, we are done.
      break;
    }

    // Step.
    if (tx <= ty && tx <= tz)
    {
      if (current_idx == p_GFinal_idx)
      {
        // If we would step out of range, we are done.
        break;
      }
      current_idx += x_step;
      tx += delta_tx;
    }
    else if (ty <= tx && ty <= tz)
    {
      if (current_idy == p_GFinal_idy)
      {
        // If we would step out of range, we are done.
        break;
      }
      current_idy += y_step;
      ty += delta_ty;
    }
    else
    {
      if (current_idz == p_GFinal_idz)
      {
        // If we would step out of range, we are done.
        break;
      }
      current_idz += z_step;
      tz += delta_tz;
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
  const float current_occupancy = filter_grid[voxel_index];
  if (current_occupancy <= 0.5f)
  {
    int cameras_seen_filled = 0;
    int cameras_seen_free = 0;
    for (int idx = 0; idx < num_grids; idx++)
    {
      const int tracking_grid_offset = num_cells * 2 * idx;
      const int tracking_grid_index =
          tracking_grid_offset + (voxel_index * 2);
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
      filter_grid[voxel_index] = 1.0f;
    }
    else if (cameras_seen_free >= num_cameras_seen_free)
    {
      filter_grid[voxel_index] = 0.0f;
    }
    else
    {
      filter_grid[voxel_index] = 0.5f;
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
      const float voxel_size, const float inverse_voxel_size,
      const float grid_x_size, const float grid_y_size, const float grid_z_size,
      const int32_t num_x_voxels, const int32_t num_y_voxels,
      const int32_t num_z_voxels, TrackingGridsHandle& tracking_grids,
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

    const int32_t stride1 = num_y_voxels * num_z_voxels;
    const int32_t stride2 = num_z_voxels;
    const int32_t starting_index = static_cast<int32_t>(
        real_tracking_grids.GetTrackingGridStartingOffset(tracking_grid_index));
    // Build kernel
    cl::Kernel raycasting_kernel(*raycasting_program_, "RaycastPoint");
    raycasting_kernel.setArg(0, device_points_buffer);
    raycasting_kernel.setArg(1, max_range);
    raycasting_kernel.setArg(2, device_grid_pointcloud_transform_buffer);
    raycasting_kernel.setArg(3, voxel_size);
    raycasting_kernel.setArg(4, inverse_voxel_size);
    raycasting_kernel.setArg(5, grid_x_size);
    raycasting_kernel.setArg(6, grid_y_size);
    raycasting_kernel.setArg(7, grid_z_size);
    raycasting_kernel.setArg(8, num_x_voxels);
    raycasting_kernel.setArg(9, num_y_voxels);
    raycasting_kernel.setArg(10, num_z_voxels);
    raycasting_kernel.setArg(11, stride1);
    raycasting_kernel.setArg(12, stride2);
    raycasting_kernel.setArg(13, real_tracking_grids.GetBuffer());
    raycasting_kernel.setArg(14, starting_index);
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
        sizeof(float) * static_cast<size_t>(num_cells);
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

    const size_t item_size = sizeof(float);
    const size_t buffer_size =
        static_cast<size_t>(real_filter_grid.NumVoxels()) * item_size;
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
