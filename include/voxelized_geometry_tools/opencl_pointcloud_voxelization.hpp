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
class OpenCLPointcloudVoxelizer : public PointcloudVoxelizationInterface {
public:
  OpenCLPointcloudVoxelizer();

  voxelized_geometry_tools::CollisionMap VoxelizePointclouds(
      const CollisionMap& static_environment,
      const PointcloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointcloudWrapperPtr>& pointclouds) const override;
};
}
}
