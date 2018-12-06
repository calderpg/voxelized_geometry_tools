#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
class OpenCLPointCloudVoxelizer : public PointCloudVoxelizationInterface {
public:
  OpenCLPointCloudVoxelizer();

  voxelized_geometry_tools::CollisionMap VoxelizePointClouds(
      const CollisionMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperPtr>& pointclouds) const override;

private:
  std::unique_ptr<opencl_helpers::OpenCLVoxelizationHelperInterface> interface_;
};
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
