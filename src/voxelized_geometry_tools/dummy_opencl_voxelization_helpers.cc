#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
std::vector<AvailableDevice> GetAvailableDevices() { return {}; }

std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeOpenCLVoxelizationHelper(
    const std::map<std::string, int32_t>&, const LoggingFunction&)
{
  return std::unique_ptr<DeviceVoxelizationHelperInterface>();
}
}  // namespace opencl_helpers
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
