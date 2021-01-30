#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <voxelized_geometry_tools/device_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeCudaVoxelizationHelper(const std::map<std::string, int32_t>& options);
}  // namespace cuda_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
