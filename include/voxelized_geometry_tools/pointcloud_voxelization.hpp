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
enum class BackendOptions : uint8_t { BEST_AVAILABLE = 0x00,
                                      CPU = 0x01,
                                      OPENCL = 0x02,
                                      CUDA = 0x03 };

// Wrapper for an available voxelizer backend.
class AvailableBackend
{
public:
  AvailableBackend(
      const std::string& device_name,
      const std::map<std::string, int32_t>& device_options,
      const BackendOptions backend_option)
      : device_name_(device_name), device_options_(device_options),
        backend_option_(backend_option) {}

  AvailableBackend(
      const AvailableDevice& device, const BackendOptions backend_option)
      : AvailableBackend(
          device.DeviceName(), device.DeviceOptions(), backend_option) {}

  const std::string& DeviceName() const { return device_name_; }

  const std::map<std::string, int32_t>& DeviceOptions() const
  {
    return device_options_;
  }

  BackendOptions BackendOption() const { return backend_option_; }

private:
  std::string device_name_;
  std::map<std::string, int32_t> device_options_;
  BackendOptions backend_option_{};
};

std::vector<AvailableBackend> GetAvailableBackends();

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(
    const BackendOptions backend_option,
    const std::map<std::string, int32_t>& device_options);

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(const AvailableBackend& backend)
{
  return MakePointCloudVoxelizer(
      backend.BackendOption(), backend.DeviceOptions());
}

std::unique_ptr<PointCloudVoxelizationInterface>
MakeBestAvailablePointCloudVoxelizer(
    const std::map<std::string, int32_t>& device_options);
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
