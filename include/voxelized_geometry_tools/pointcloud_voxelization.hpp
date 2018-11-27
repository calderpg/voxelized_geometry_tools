#pragma once

#include <memory>

#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
std::unique_ptr<PointcloudVoxelizationInterface>
MakeBestAvailablePointcloudVoxelizer();
}
}
