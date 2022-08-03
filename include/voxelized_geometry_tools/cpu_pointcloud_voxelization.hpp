#pragma once

#include <memory>
#include <vector>

#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
/// CPU-based (OpenMP) implementation of pointcloud voxelizer.
class CpuPointCloudVoxelizer : public PointCloudVoxelizationInterface {
public:
  explicit CpuPointCloudVoxelizer(
      const std::map<std::string, int32_t>& options);

private:
  VoxelizerRuntime DoVoxelizePointClouds(
      const CollisionMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
      CollisionMap& output_environment) const override;

  bool use_parallel_ = true;
};
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
