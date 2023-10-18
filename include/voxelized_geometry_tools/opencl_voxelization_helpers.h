#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <voxelized_geometry_tools/device_voxelization_interface.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
std::vector<AvailableDevice> GetAvailableDevices();

std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeOpenCLVoxelizationHelper(
    const std::map<std::string, int32_t>& options,
    const LoggingFunction& logging_fn);
}  // namespace opencl_helpers
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
