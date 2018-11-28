#pragma once

#include <cstdint>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
bool IsAvailable();

int32_t* PrepareTrackingGrid(const int64_t num_cells);

void RaycastPoints(
    const float* points, const int32_t num_points,
    const float* const pointcloud_origin_transform,
    const float* const inverse_grid_origin_transform,
    const float inverse_cell_size, const int32_t num_x_cells,
    const int32_t num_y_cells, const int32_t num_z_cells,
    int32_t* const device_tracking_grid_ptr);

float* PrepareFilterGrid(
    const int64_t num_cells, const void* host_data_ptr);

void FilterTrackingGrids(
    const int64_t num_cells, const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs,
    float* const device_filter_grid_ptr, const float percent_seen_free,
    const int32_t outlier_points_threshold,
    const int32_t num_cameras_seen_free);

void RetrieveTrackingGrid(
    const int64_t num_cells, const int32_t* const device_tracking_grid_ptr,
    void* host_data_ptr);

void RetrieveFilteredGrid(
    const int64_t num_cells, const float* const device_filter_grid_ptr,
    void* host_data_ptr);

void CleanupDeviceMemory(
    const int32_t num_device_tracking_grids,
    int32_t* const* device_tracking_grid_ptrs, float* device_filter_grid_ptr);
}
}
}
