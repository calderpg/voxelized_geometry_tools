#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
/// Struct to store free/filled counts in a voxel grid. Counts are stored in
/// std::atomic<int32_t> since a given cell may be manipulated by multiple
/// threads simultaneously and this gives us atomic fetch_add() to increment the
/// stored values.
struct CpuVoxelizationTrackingCell
{
  std::atomic<int32_t> seen_free_count;
  std::atomic<int32_t> seen_filled_count;

  CpuVoxelizationTrackingCell()
  {
    seen_free_count.store(0);
    seen_filled_count.store(0);
  }

  CpuVoxelizationTrackingCell(
      const int32_t seen_free, const int32_t seen_filled)
  {
    seen_free_count.store(seen_free);
    seen_filled_count.store(seen_filled);
  }

  /// We need copy constructor since std::atomics do not have copy constructors.
  CpuVoxelizationTrackingCell(const CpuVoxelizationTrackingCell& other)
  {
    seen_free_count.store(other.seen_free_count.load());
    seen_filled_count.store(other.seen_filled_count.load());
  }

  /// We need assignment operator since std::atomics do not have it.
  CpuVoxelizationTrackingCell& operator =
      (const CpuVoxelizationTrackingCell& other)
  {
    if (this != &other)
    {
      this->seen_free_count.store(other.seen_free_count.load());
      this->seen_filled_count.store(other.seen_filled_count.load());
    }
    return *this;
  }
};

using CpuVoxelizationTrackingGrid = common_robotics_utilities::voxel_grid
    ::VoxelGrid<CpuVoxelizationTrackingCell>;

class CpuPointCloudVoxelizer : public PointCloudVoxelizationInterface {
public:
  CpuPointCloudVoxelizer() {}

  voxelized_geometry_tools::CollisionMap VoxelizePointClouds(
      const CollisionMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperPtr>& pointclouds) const override;

private:
  void RaycastPointCloud(
      const PointCloudWrapper& cloud, const double step_size,
      CpuVoxelizationTrackingGrid& tracking_grid) const;

  voxelized_geometry_tools::CollisionMap CombineAndFilterGrids(
      const CollisionMap& static_environment,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<CpuVoxelizationTrackingGrid>& tracking_grids) const;
};
}
}
