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
std::unique_ptr<PointcloudVoxelizationInterface>
MakeBestAvailablePointcloudVoxelizer()
{
  // Since not all voxelizers will be available on all platforms, we try them
  // in order of preference. If available (i.e. on NVIDIA platforms), CUDA is
  // always preferable to OpenCL, and likewise OpenCL is always preferable to
  // CPU-based (OpenMP). If you want a specific implementation, you should
  // directly construct the desired voxelizer.
  try
  {
    std::cout << "Trying CUDA Pointcloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointcloudVoxelizationInterface>(
        new CudaPointcloudVoxelizer());
  }
  catch (std::runtime_error&)
  {
    std::cerr << "CUDA Pointcloud Voxelizer is not available" << std::endl;
  }
  try
  {
    std::cout << "Trying OpenCL Pointcloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointcloudVoxelizationInterface>(
        new OpenCLPointcloudVoxelizer());
  }
  catch (std::runtime_error&)
  {
    std::cerr << "OpenCL Pointcloud Voxelizer is not available" << std::endl;
  }
  try
  {
    std::cout << "Trying CPU Pointcloud Voxelizer..." << std::endl;
    return std::unique_ptr<PointcloudVoxelizationInterface>(
        new CpuPointcloudVoxelizer());
  }
  catch (std::runtime_error&)
  {
    throw std::runtime_error("No Pointcloud Voxelizers available");
  }
}
}
}
