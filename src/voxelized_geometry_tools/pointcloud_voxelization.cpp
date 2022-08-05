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
std::vector<AvailableBackend> GetAvailableBackends()
{
  std::vector<AvailableBackend> available_backends;

  const auto cuda_devices = cuda_helpers::GetAvailableDevices();
  for (const auto& cuda_device : cuda_devices)
  {
    available_backends.push_back(AvailableBackend(
        cuda_device, BackendOptions::CUDA));
  }

  const auto opencl_devices = opencl_helpers::GetAvailableDevices();
  for (const auto& opencl_device : opencl_devices)
  {
    available_backends.push_back(AvailableBackend(
        opencl_device, BackendOptions::OPENCL));
  }

  available_backends.push_back(AvailableBackend(
      "CPU/OpenMP (parallel)", {{"CPU_PARALLELIZE", 1}}, BackendOptions::CPU));

  available_backends.push_back(AvailableBackend(
      "CPU/OpenMP (serial)", {{"CPU_PARALLELIZE", 0}}, BackendOptions::CPU));

  return available_backends;
}

std::unique_ptr<PointCloudVoxelizationInterface> MakePointCloudVoxelizer(
    const BackendOptions backend_option,
    const std::map<std::string, int32_t>& device_options,
    const LoggingFunction& logging_fn)
{
  if (backend_option == BackendOptions::BEST_AVAILABLE)
  {
    return MakeBestAvailablePointCloudVoxelizer(device_options, logging_fn);
  }
  else if (backend_option == BackendOptions::CPU)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CpuPointCloudVoxelizer(device_options, logging_fn));
  }
  else if (backend_option == BackendOptions::OPENCL)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new OpenCLPointCloudVoxelizer(device_options, logging_fn));
  }
  else if (backend_option == BackendOptions::CUDA)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CudaPointCloudVoxelizer(device_options, logging_fn));
  }
  else
  {
    throw std::invalid_argument("Invalid BackendOptions");
  }
}

std::unique_ptr<PointCloudVoxelizationInterface> MakePointCloudVoxelizer(
    const AvailableBackend& backend, const LoggingFunction& logging_fn)
{
  return MakePointCloudVoxelizer(
      backend.BackendOption(), backend.DeviceOptions(), logging_fn);
}

std::unique_ptr<PointCloudVoxelizationInterface>
MakeBestAvailablePointCloudVoxelizer(
    const std::map<std::string, int32_t>& device_options,
    const LoggingFunction& logging_fn)
{
  // Since not all voxelizers will be available on all platforms, we try them
  // in order of preference. If available (i.e. on NVIDIA platforms), CUDA is
  // always preferable to OpenCL, and likewise OpenCL is always preferable to
  // CPU-based (OpenMP). If you want a specific implementation, you should
  // directly construct the desired voxelizer.
  try
  {
    if (logging_fn)
    {
      logging_fn("Trying to construct CUDA PointCloud Voxelizer...");
    }
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CudaPointCloudVoxelizer(device_options, logging_fn));
  }
  catch (const std::runtime_error&)
  {
    if (logging_fn)
    {
      logging_fn("CUDA PointCloud Voxelizer is not available");
    }
  }
  try
  {
    if (logging_fn)
    {
      logging_fn("Trying to construct OpenCL PointCloud Voxelizer...");
    }
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new OpenCLPointCloudVoxelizer(device_options, logging_fn));
  }
  catch (const std::runtime_error&)
  {
    if (logging_fn)
    {
      logging_fn("OpenCL PointCloud Voxelizer is not available");
    }
  }
  try
  {
    if (logging_fn)
    {
      logging_fn("Trying to construct CPU PointCloud Voxelizer...");
    }
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CpuPointCloudVoxelizer(device_options, logging_fn));
  }
  catch (const std::runtime_error&)
  {
    throw std::runtime_error("No PointCloud Voxelizers available");
  }
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
