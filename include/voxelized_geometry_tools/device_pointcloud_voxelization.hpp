#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/parallelism.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/device_voxelization_interface.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace pointcloud_voxelization
{
/// Base class for device-accelerated pointcloud voxelizers.
class DevicePointCloudVoxelizer : public PointCloudVoxelizationInterface
{
public:
  virtual ~DevicePointCloudVoxelizer() {}

protected:
  DevicePointCloudVoxelizer(
      const std::map<std::string, int32_t>& options,
      const LoggingFunction& logging_fn);

  void EnforceAvailable() const
  {
    if (!helper_interface_)
    {
      throw std::runtime_error(
          device_name_ + " is not available (feature was not built)");
    }
    if (!helper_interface_->IsAvailable())
    {
      throw std::runtime_error(
          device_name_ + " is not available (device cannot be used)");
    }
  }

  std::unique_ptr<DeviceVoxelizationHelperInterface> helper_interface_;
  std::string device_name_;

private:
  VoxelizerRuntime DoVoxelizePointClouds(
      const CollisionMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
      CollisionMap& output_environment) const override;

  const common_robotics_utilities::parallelism::DegreeOfParallelism&
  DispatchParallelism() const { return dispatch_parallelism_; }

  common_robotics_utilities::parallelism::DegreeOfParallelism
      dispatch_parallelism_;
};

/// Implementation of device-accelerated pointcloud voxelization for CUDA
/// devices. See base classes for method documentation.
class CudaPointCloudVoxelizer : public DevicePointCloudVoxelizer
{
public:
  CudaPointCloudVoxelizer(
      const std::map<std::string, int32_t>& options,
      const LoggingFunction& logging_fn = {});
};

/// Implementation of device-accelerated pointcloud voxelization for OpenCL
/// devices. See base classes for method documentation.
class OpenCLPointCloudVoxelizer : public DevicePointCloudVoxelizer
{
public:
  OpenCLPointCloudVoxelizer(
      const std::map<std::string, int32_t>& options,
      const LoggingFunction& logging_fn = {});
};

}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
