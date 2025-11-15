#pragma once

#include <voxelized_geometry_tools/occupancy_component_map.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_component_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_map.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN

/// Helpers to convert between occupancy map cell types.
inline OccupancyComponentCell ConvertToOccupancyComponentCell(
    const OccupancyCell& occupancy_cell)
{
  return OccupancyComponentCell(occupancy_cell.Occupancy());
}

inline OccupancyCell ConvertFromOccupancyComponentCell(
    const OccupancyComponentCell& occupancy_component_cell)
{
  return OccupancyCell(occupancy_component_cell.Occupancy());
}

inline TaggedObjectOccupancyComponentCell
ConvertToTaggedObjectOccupancyComponentCell(
    const TaggedObjectOccupancyCell& occupancy_cell)
{
  return TaggedObjectOccupancyComponentCell(
      occupancy_cell.Occupancy(), occupancy_cell.ObjectId());
}

inline TaggedObjectOccupancyCell ConvertFromTaggedObjectOccupancyComponentCell(
    const TaggedObjectOccupancyComponentCell& occupancy_component_cell)
{
  return TaggedObjectOccupancyCell(
      occupancy_component_cell.Occupancy(),
      occupancy_component_cell.ObjectId());
}

/// Helpers to convert between occupancy map types.
OccupancyComponentMap ConvertToOccupancyComponentMap(
    const OccupancyMap& occupancy_map);

OccupancyMap ConvertFromOccupancyComponentMap(
    const OccupancyComponentMap& occupancy_component_map);

TaggedObjectOccupancyComponentMap ConvertToTaggedObjectOccupancyComponentMap(
    const TaggedObjectOccupancyMap& occupancy_map);

TaggedObjectOccupancyMap ConvertFromTaggedObjectOccupancyComponentMap(
    const TaggedObjectOccupancyComponentMap& occupancy_component_map);

VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
