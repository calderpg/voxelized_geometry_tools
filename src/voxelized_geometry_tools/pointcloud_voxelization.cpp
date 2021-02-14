#include <voxelized_geometry_tools/pointcloud_voxelization.hpp>

#include <memory>
#include <iostream>
#include <stdexcept>

#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>
#include <voxelized_geometry_tools/device_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
std::vector<AvailableVoxelizer> GetAvailableVoxelizers()
{
  std::vector<AvailableVoxelizer> available_voxelizers;

  const auto cuda_devices = cuda_helpers::GetAvailableDevices();
  for (const auto& cuda_device : cuda_devices)
  {
    available_voxelizers.push_back(AvailableVoxelizer(
        cuda_device, VoxelizerOptions::CUDA));
  }

  const auto opencl_devices = opencl_helpers::GetAvailableDevices();
  for (const auto& opencl_device : opencl_devices)
  {
    available_voxelizers.push_back(AvailableVoxelizer(
        opencl_device, VoxelizerOptions::OPENCL));
  }

  available_voxelizers.push_back(AvailableVoxelizer(
      "CPU", {}, VoxelizerOptions::CPU));

  return available_voxelizers;
}

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(
    const VoxelizerOptions voxelizer_option,
    const std::map<std::string, int32_t>& options)
{
  if (voxelizer_option == VoxelizerOptions::BEST_AVAILABLE)
  {
    return MakeBestAvailablePointCloudVoxelizer(options);
  }
  else if (voxelizer_option == VoxelizerOptions::CPU)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CpuPointCloudVoxelizer());
  }
  else if (voxelizer_option == VoxelizerOptions::OPENCL)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new OpenCLPointCloudVoxelizer(options));
  }
  else if (voxelizer_option == VoxelizerOptions::CUDA)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CudaPointCloudVoxelizer(options));
  }
  else
  {
    throw std::invalid_argument("Invalid VoxelizerOptions");
  }
}

std::unique_ptr<PointCloudVoxelizationInterface>
MakeBestAvailablePointCloudVoxelizer(
    const std::map<std::string, int32_t>& options)
{
  // Since not all voxelizers will be available on all platforms, we try them
  // in order of preference. If available (i.e. on NVIDIA platforms), CUDA is
  // always preferable to OpenCL, and likewise OpenCL is always preferable to
  // CPU-based (OpenMP). If you want a specific implementation, you should
  // directly construct the desired voxelizer.
  try
  {
    std::cout << "Trying CUDA PointCloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CudaPointCloudVoxelizer(options));
  }
  catch (std::runtime_error&)
  {
    std::cerr << "CUDA PointCloud Voxelizer is not available" << std::endl;
  }
  try
  {
    std::cout << "Trying OpenCL PointCloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new OpenCLPointCloudVoxelizer(options));
  }
  catch (std::runtime_error&)
  {
    std::cerr << "OpenCL PointCloud Voxelizer is not available" << std::endl;
  }
  try
  {
    std::cout << "Trying CPU PointCloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CpuPointCloudVoxelizer());
  }
  catch (std::runtime_error&)
  {
    throw std::runtime_error("No PointCloud Voxelizers available");
  }
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
