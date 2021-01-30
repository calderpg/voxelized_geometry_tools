#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

#include <cstdint>
#include <map>
#include <string>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
std::unique_ptr<DeviceVoxelizationHelperInterface>
MakeOpenCLVoxelizationHelper(const std::map<std::string, int32_t>&)
{
  return std::unique_ptr<DeviceVoxelizationHelperInterface>();
}
}  // namespace opencl_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
