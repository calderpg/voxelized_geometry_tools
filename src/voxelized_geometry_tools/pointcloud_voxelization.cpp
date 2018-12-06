#include <voxelized_geometry_tools/pointcloud_voxelization.hpp>

#include <memory>
#include <iostream>
#include <stdexcept>

#include <voxelized_geometry_tools/cuda_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/opencl_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
std::unique_ptr<PointCloudVoxelizationInterface>
MakeBestAvailablePointCloudVoxelizer()
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
        new CudaPointCloudVoxelizer());
  }
  catch (std::runtime_error&)
  {
    std::cerr << "CUDA PointCloud Voxelizer is not available" << std::endl;
  }
  try
  {
    std::cout << "Trying OpenCL PointCloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new OpenCLPointCloudVoxelizer());
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

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(const VoxelizerOptions option)
{
  if (option == VoxelizerOptions::BEST_AVAILABLE)
  {
    return MakeBestAvailablePointCloudVoxelizer();
  }
  else if (option == VoxelizerOptions::CPU)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CpuPointCloudVoxelizer());
  }
  else if (option == VoxelizerOptions::OPENCL)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new OpenCLPointCloudVoxelizer());
  }
  else if (option == VoxelizerOptions::CUDA)
  {
    return std::unique_ptr<PointCloudVoxelizationInterface>(
        new CudaPointCloudVoxelizer());
  }
  else
  {
    throw std::invalid_argument("Invalid VoxelizerOptions");
  }
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
