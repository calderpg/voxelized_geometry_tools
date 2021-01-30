#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
// Helper for common option retrieval in device voxelizers
inline int32_t RetrieveOptionOrDefault(
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

// Opaque handle type to tracking grids.
class TrackingGridsHandle
{
public:
  virtual ~TrackingGridsHandle() {}

  int64_t GetTrackingGridStartingOffset(const size_t index) const
  {
    return tracking_grid_starting_offsets_.at(index);
  }

  size_t GetNumTrackingGrids() const
  {
    return tracking_grid_starting_offsets_.size();
  }

  int64_t NumCellsPerGrid() const { return num_cells_per_grid_; }

protected:
  TrackingGridsHandle(
      const std::vector<int64_t>& tracking_grid_starting_offsets,
      const int64_t num_cells_per_grid)
      : tracking_grid_starting_offsets_(tracking_grid_starting_offsets),
        num_cells_per_grid_(num_cells_per_grid) {}

private:
  std::vector<int64_t> tracking_grid_starting_offsets_;
  int64_t num_cells_per_grid_ = 0;
};

// Opaque handle type to filter grid
class FilterGridHandle
{
public:
  virtual ~FilterGridHandle() {}

  int64_t NumCells() const { return num_cells_; }

protected:
  FilterGridHandle(const int64_t num_cells) : num_cells_(num_cells) {}

private:
  int64_t num_cells_ = 0;
};

class DeviceVoxelizationHelperInterface
{
public:
  virtual ~DeviceVoxelizationHelperInterface() {}

  virtual bool IsAvailable() const = 0;

  virtual std::unique_ptr<TrackingGridsHandle> PrepareTrackingGrids(
      const int64_t num_cells, const int32_t num_grids) = 0;

  virtual void RaycastPoints(
      const std::vector<float>& raw_points,
      const float* const pointcloud_origin_transform, const float max_range,
      const float* const inverse_grid_origin_transform,
      const float inverse_step_size, const float inverse_cell_size,
      const int32_t num_x_cells, const int32_t num_y_cells,
      const int32_t num_z_cells, TrackingGridsHandle& tracking_grids,
      const size_t tracking_grid_index) = 0;

  virtual std::unique_ptr<FilterGridHandle> PrepareFilterGrid(
      const int64_t num_cells, const void* host_data_ptr) = 0;

  virtual void FilterTrackingGrids(
      const TrackingGridsHandle& tracking_grids, const float percent_seen_free,
      const int32_t outlier_points_threshold,
      const int32_t num_cameras_seen_free, FilterGridHandle& filter_grid) = 0;

  virtual void RetrieveTrackingGrid(
      const TrackingGridsHandle& tracking_grids,
      const size_t tracking_grid_index, void* host_data_ptr) = 0;

  virtual void RetrieveFilteredGrid(
      const FilterGridHandle& filter_grid, void* host_data_ptr) = 0;
};
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
