#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
std::unique_ptr<PointCloudVoxelizationInterface>
MakeBestAvailablePointCloudVoxelizer(
    const std::map<std::string, int32_t>& options);

enum class VoxelizerOptions : uint8_t { BEST_AVAILABLE = 0x00,
                                        CPU = 0x01,
                                        OPENCL = 0x02,
                                        CUDA = 0x03 };

std::unique_ptr<PointCloudVoxelizationInterface>
MakePointCloudVoxelizer(
    const VoxelizerOptions voxelizer,
    const std::map<std::string, int32_t>& options);
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
