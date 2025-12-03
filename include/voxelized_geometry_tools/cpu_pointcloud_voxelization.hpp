#pragma once

#include <memory>
#include <vector>

#include <common_robotics_utilities/parallelism.hpp>
#include <voxelized_geometry_tools/device_voxelization_interface.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
/// CPU-based (OpenMP) implementation of pointcloud voxelizer.
class CpuPointCloudVoxelizer : public PointCloudVoxelizationInterface {
public:
  /// Struct to store free/filled counts in a voxel grid. Counts are stored in
  /// std::atomic<int32_t> since a given cell may be manipulated by multiple
  /// threads simultaneously and this gives us atomic fetch_add() to increment
  /// the stored values.
  struct CpuVoxelizationTrackingCell
  {
    common_robotics_utilities::utility
        ::CopyableMoveableAtomic<int32_t, std::memory_order_relaxed>
            seen_free_count{0};
    common_robotics_utilities::utility
        ::CopyableMoveableAtomic<int32_t, std::memory_order_relaxed>
            seen_filled_count{0};
  };

  using CpuVoxelizationTrackingGrid = common_robotics_utilities::voxel_grid
      ::VoxelGrid<CpuVoxelizationTrackingCell>;
  using CpuVoxelizationTrackingGridAllocator =
      Eigen::aligned_allocator<CpuVoxelizationTrackingGrid>;
  using VectorCpuVoxelizationTrackingGrid =
      std::vector<CpuVoxelizationTrackingGrid,
                  CpuVoxelizationTrackingGridAllocator>;

  CpuPointCloudVoxelizer(
      const std::map<std::string, int32_t>& options,
      const LoggingFunction& logging_fn = {});

  // Access is provided to the internal implementations, but with (potentially
  // expensive) sanity checks on their inputs.

  void RaycastPointCloud(
      const PointCloudWrapper& cloud,
      CpuVoxelizationTrackingGrid& tracking_grid) const;

  void RaycastSinglePoint(
      const Eigen::Vector4d& p_GCo, const Eigen::Vector4d& p_GP,
      const double max_range, CpuVoxelizationTrackingGrid& tracking_grid) const;

  void CombineAndFilterGrids(
      const PointCloudVoxelizationFilterOptions& filter_options,
      const VectorCpuVoxelizationTrackingGrid& tracking_grids,
      OccupancyMap& filtered_grid) const;

private:
  VoxelizerRuntime DoVoxelizePointClouds(
      const OccupancyMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
      OccupancyMap& output_environment) const override;

  // Note: the private implementations do not perform sanity checks on their
  // inputs, as they can assume these are provided correctly.

  void DoRaycastPointCloud(
      const PointCloudWrapper& cloud,
      CpuVoxelizationTrackingGrid& tracking_grid) const;

  void DoRaycastSinglePoint(
      const Eigen::Vector4d& p_GCo,
      const common_robotics_utilities::voxel_grid::GridIndex& p_GCo_index,
      const Eigen::Vector4d& p_GP, const double max_range,
      CpuVoxelizationTrackingGrid& tracking_grid) const;

  void DoCombineAndFilterGrids(
      const PointCloudVoxelizationFilterOptions& filter_options,
      const VectorCpuVoxelizationTrackingGrid& tracking_grids,
      OccupancyMap& filtered_grid) const;

  const common_robotics_utilities::parallelism::DegreeOfParallelism&
  Parallelism() const { return parallelism_; }

  common_robotics_utilities::parallelism::DegreeOfParallelism parallelism_;
};
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
