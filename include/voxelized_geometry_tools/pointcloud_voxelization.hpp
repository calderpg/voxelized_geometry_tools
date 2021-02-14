#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <voxelized_geometry_tools/device_voxelization_interface.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
enum class VoxelizerOptions : uint8_t { BEST_AVAILABLE = 0x00,
                                        CPU = 0x01,
                                        OPENCL = 0x02,
                                        CUDA = 0x03 };

// Wrapper for an available voxelizer.
class AvailableVoxelizer
{
public:
  AvailableVoxelizer(
      const std::string& device_name,
      const std::map<std::string, int32_t>& device_options,
      const VoxelizerOptions voxelizer_option)
      : device_name_(device_name), device_options_(device_options),
        voxelizer_option_(voxelizer_option) {}

  AvailableVoxelizer(
      const AvailableDevice& device, const VoxelizerOptions voxelizer_option)
      : AvailableVoxelizer(
          device.DeviceName(), device.DeviceOptions(), voxelizer_option) {}

  const std::string& DeviceName() const { return device_name_; }

  const std::map<std::string, int32_t>& DeviceOptions() const
  {
    return device_options_;
  }

  VoxelizerOptions VoxelizerOption() const { return voxelizer_option_; }

private:
  std::string device_name_;
  std::map<std::string, int32_t> device_options_;
  VoxelizerOptions voxelizer_option_{};
};

std::vector<AvailableVoxelizer> GetAvailableVoxelizers();

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(
    const VoxelizerOptions voxelizer_option,
    const std::map<std::string, int32_t>& options);

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(const AvailableVoxelizer& voxelizer)
{
  return MakePointCloudVoxelizer(
      voxelizer.VoxelizerOption(), voxelizer.DeviceOptions());
}

std::unique_ptr<PointCloudVoxelizationInterface>
MakeBestAvailablePointCloudVoxelizer(
    const std::map<std::string, int32_t>& options);
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
