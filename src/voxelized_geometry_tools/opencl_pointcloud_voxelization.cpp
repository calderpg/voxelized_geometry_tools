#include <voxelized_geometry_tools/opencl_pointcloud_voxelization.hpp>

#include <atomic>
#include <cmath>
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
OpenCLPointcloudVoxelizer::OpenCLPointcloudVoxelizer()
{
  if (!opencl_helpers::IsAvailable())
  {
    throw std::runtime_error("OpenCLPointcloudVoxelizer not available");
  }
}

CollisionMap OpenCLPointcloudVoxelizer::VoxelizePointclouds(
    const CollisionMap& static_environment,
    const PointcloudVoxelizationFilterOptions& filter_options,
    const std::vector<PointcloudWrapperPtr>& pointclouds) const
{
  if (!static_environment.IsInitialized())
  {
    throw std::invalid_argument("!static_environment.IsInitialized()");
  }
  if (opencl_helpers::IsAvailable())
  {
    UNUSED(static_environment);
    UNUSED(filter_options);
    UNUSED(pointclouds);
    throw std::runtime_error("OpenCLPointcloudVoxelizer not implemented");
  }
  else
  {
    UNUSED(static_environment);
    UNUSED(filter_options);
    UNUSED(pointclouds);
    throw std::runtime_error("OpenCLPointcloudVoxelizer not available");
  }
}
}
}
