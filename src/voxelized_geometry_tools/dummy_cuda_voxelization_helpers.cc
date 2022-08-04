#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
std::vector<AvailableDevice> GetAvailableDevices() { return {}; }

std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeCudaVoxelizationHelper(
    const std::map<std::string, int32_t>&, const LoggingFunction&)
{
  return std::unique_ptr<DeviceVoxelizationHelperInterface>();
}
}  // namespace cuda_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
