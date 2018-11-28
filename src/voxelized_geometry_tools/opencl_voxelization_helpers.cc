#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
bool IsAvailable() { return true; }

int32_t* PrepareTrackingGrid(const int64_t num_cells)
{
  int32_t* device_tracking_grid_ptr =
      static_cast<int32_t*>(calloc(num_cells * 2, sizeof(int32_t)));
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
  // Prepare for raycasting
  const int32_t stride1 = num_y_cells * num_z_cells;
  const int32_t stride2 = num_z_cells;
  // Raycast
  for (int32_t point_index = 0; point_index < num_points; point_index++)
  {
    const float ox = pointcloud_origin_transform[12];
    const float oy = pointcloud_origin_transform[13];
    const float oz = pointcloud_origin_transform[14];
    const float px = points[(point_index * 3) + 0];
    const float py = points[(point_index * 3) + 1];
    const float pz = points[(point_index * 3) + 2];
    const float wx = pointcloud_origin_transform[0] * px
                     + pointcloud_origin_transform[4] * py
                     + pointcloud_origin_transform[8] * pz
                     + pointcloud_origin_transform[12];
    const float wy = pointcloud_origin_transform[1] * px
                     + pointcloud_origin_transform[5] * py
                     + pointcloud_origin_transform[9] * pz
                     + pointcloud_origin_transform[13];
    const float wz = pointcloud_origin_transform[2] * px
                     + pointcloud_origin_transform[6] * py
                     + pointcloud_origin_transform[10] * pz
                     + pointcloud_origin_transform[14];
    const float step_size = (1.0f / inverse_cell_size) * 0.5f;
    const float rx = wx - ox;
    const float ry = wy - oy;
    const float rz = wz - oz;
    const float current_ray_length =
        std::sqrt((rx * rx) + (ry * ry) + (rz * rz));
    const float num_steps = std::floor(current_ray_length / step_size);
    int32_t previous_x_cell = -1;
    int32_t previous_y_cell = -1;
    int32_t previous_z_cell = -1;
    for (float step = 0.0f; step < num_steps; step += 1.0f)
    {
      bool in_grid = false;
      const float elapsed_ratio = step / num_steps;
      const float cx = (rx * elapsed_ratio) + ox;
      const float cy = (ry * elapsed_ratio) + oy;
      const float cz = (rz * elapsed_ratio) + oz;
      const float gx =
          inverse_grid_origin_transform[0] * cx
          + inverse_grid_origin_transform[4] * cy
          + inverse_grid_origin_transform[8] * cz
          + inverse_grid_origin_transform[12];
      const float gy =
          inverse_grid_origin_transform[1] * cx
          + inverse_grid_origin_transform[5] * cy
          + inverse_grid_origin_transform[9] * cz
          + inverse_grid_origin_transform[13];
      const float gz =
          inverse_grid_origin_transform[2] * cx
          + inverse_grid_origin_transform[6] * cy
          + inverse_grid_origin_transform[10] * cz
          + inverse_grid_origin_transform[14];
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
          device_tracking_grid_ptr[cell_index * 2] += 1;
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
        inverse_grid_origin_transform[0] * wx
        + inverse_grid_origin_transform[4] * wy
        + inverse_grid_origin_transform[8] * wz
        + inverse_grid_origin_transform[12];
    const float gy =
        inverse_grid_origin_transform[1] * wx
        + inverse_grid_origin_transform[5] * wy
        + inverse_grid_origin_transform[9] * wz
        + inverse_grid_origin_transform[13];
    const float gz =
        inverse_grid_origin_transform[2] * wx
        + inverse_grid_origin_transform[6] * wy
        + inverse_grid_origin_transform[10] * wz
        + inverse_grid_origin_transform[14];
    const int32_t x_cell = static_cast<int32_t>(gx * inverse_cell_size);
    const int32_t y_cell = static_cast<int32_t>(gy * inverse_cell_size);
    const int32_t z_cell = static_cast<int32_t>(gz * inverse_cell_size);
    if (x_cell >= 0 && x_cell < num_x_cells && y_cell >= 0
        && y_cell < num_y_cells && z_cell >= 0 && z_cell < num_z_cells)
    {
      const int32_t cell_index =
          (x_cell * stride1) + (y_cell * stride2) + z_cell;
      // Increase filled count
      device_tracking_grid_ptr[(cell_index * 2) + 1] += 1;
    }
  }
}

float* PrepareFilterGrid(const int64_t num_cells, const void* host_data_ptr)
{
  const size_t filter_grid_size = sizeof(float) * num_cells * 2;
  float* device_filter_grid_ptr =
      static_cast<float*>(calloc(num_cells * 2, sizeof(float)));
  if (device_filter_grid_ptr != nullptr)
  {
    memcpy(device_filter_grid_ptr, host_data_ptr, filter_grid_size);
  }
  return device_filter_grid_ptr;
}

void FilterTrackingGrids(
    const int64_t num_cells, const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs,
    float* const device_filter_grid_ptr, const float percent_seen_free,
    const int32_t outlier_points_threshold, const int32_t num_cameras_seen_free)
{
  // Filter
  for (int32_t voxel_index = 0; voxel_index < num_cells; voxel_index++)
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

void RetrieveTrackingGrid(
    const int64_t num_cells, const int32_t* const device_tracking_grid_ptr, 
    void* host_data_ptr)
{
  const size_t tracking_grid_size = sizeof(int32_t) * num_cells * 2;
  memcpy(host_data_ptr, device_tracking_grid_ptr, tracking_grid_size);
}

void RetrieveFilteredGrid(
    const int64_t num_cells, const float* const device_filter_grid_ptr, 
    void* host_data_ptr)
{
  const size_t filter_grid_size = sizeof(float) * num_cells * 2;
  memcpy(host_data_ptr, device_filter_grid_ptr, filter_grid_size);
}

void CleanupDeviceMemory(
    const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs, float* device_filter_grid_ptr)
{
  for (int32_t idx = 0; idx < num_device_tracking_grids; idx++)
  {
    auto device_tracking_grid_ptr = device_tracking_grid_ptrs[idx];
    // Free the device memory
    free(device_tracking_grid_ptr);
  }
  // Free the device memory
  free(device_filter_grid_ptr);
}
}
}
}
