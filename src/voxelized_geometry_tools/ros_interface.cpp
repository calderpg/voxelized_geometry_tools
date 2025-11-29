#include <voxelized_geometry_tools/ros_interface.hpp>

#include <vector>

#include <Eigen/Geometry>
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <ros/ros.h>
#endif
#include <common_robotics_utilities/color_builder.hpp>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <voxelized_geometry_tools/dynamic_spatial_hashed_occupancy_map.hpp>
#include <voxelized_geometry_tools/occupancy_component_map.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_component_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_map.hpp>

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
#include <voxelized_geometry_tools/msg/dynamic_spatial_hashed_occupancy_map_message.hpp>
#include <voxelized_geometry_tools/msg/occupancy_component_map_message.hpp>
#include <voxelized_geometry_tools/msg/occupancy_map_message.hpp>
#include <voxelized_geometry_tools/msg/tagged_object_occupancy_component_map_message.hpp>
#include <voxelized_geometry_tools/msg/tagged_object_occupancy_map_message.hpp>
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <voxelized_geometry_tools/DynamicSpatialHashedOccupancyMapMessage.h>
#include <voxelized_geometry_tools/OccupancyComponentMapMessage.h>
#include <voxelized_geometry_tools/OccupancyMapMessage.h>
#include <voxelized_geometry_tools/TaggedObjectOccupancyComponentMapMessage.h>
#include <voxelized_geometry_tools/TaggedObjectOccupancyMapMessage.h>
#endif

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace ros_interface
{

Marker ExportForDisplay(
    const OccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const OccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() > 0.5)
    {
      return collision_color;
    }
    else if (cell.Occupancy() < 0.5)
    {
      return free_color;
    }
    else
    {
      return unknown_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<OccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportForSeparateDisplay(
    const OccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportForDisplay(occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_only";
  Marker free_only_marker
      = ExportForDisplay(occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_only";
  Marker unknown_only_marker
      = ExportForDisplay(occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportSurfacesForDisplay(
    const OccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const OccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (occupancy_map.IsSurfaceIndex(index))
    {
      if (cell.Occupancy() > 0.5)
      {
        return collision_color;
      }
      else if (cell.Occupancy() < 0.5)
      {
        return free_color;
      }
      else
      {
        return unknown_color;
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<OccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_surfaces";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportSurfacesForSeparateDisplay(
    const OccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_surfaces_only";
  Marker free_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_surfaces_only";
  Marker unknown_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_surfaces_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportIndexMapForDisplay(
    const OccupancyMap& occupancy_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const OccupancyCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndexMapToRViz<OccupancyCell>(
      occupancy_map, index_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportIndicesForDisplay(
    const OccupancyMap& occupancy_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const OccupancyCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndicesToRViz<OccupancyCell>(
      occupancy_map, indices, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

OccupancyMapMessage GetMessageRepresentation(const OccupancyMap& map)
{
  OccupancyMapMessage map_message;
  map_message.header.frame_id = map.Frame();
  std::vector<uint8_t> buffer;
  OccupancyMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

OccupancyMap LoadFromMessageRepresentation(const OccupancyMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return OccupancyMap::Deserialize(decompressed_map, 0).Value();
  }
  else
  {
    return OccupancyMap::Deserialize(message.serialized_map, 0).Value();
  }
}

Marker ExportForDisplay(
    const OccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const OccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() > 0.5)
    {
      return collision_color;
    }
    else if (cell.Occupancy() < 0.5)
    {
      return free_color;
    }
    else
    {
      return unknown_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<OccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportForSeparateDisplay(
    const OccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportForDisplay(occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_only";
  Marker free_only_marker
      = ExportForDisplay(occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_only";
  Marker unknown_only_marker
      = ExportForDisplay(occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportSurfacesForDisplay(
    const OccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const OccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (occupancy_map.IsSurfaceIndex(index))
    {
      if (cell.Occupancy() > 0.5)
      {
        return collision_color;
      }
      else if (cell.Occupancy() < 0.5)
      {
        return free_color;
      }
      else
      {
        return unknown_color;
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<OccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_surfaces";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportSurfacesForSeparateDisplay(
    const OccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_surfaces_only";
  Marker free_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_surfaces_only";
  Marker unknown_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_surfaces_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportConnectedComponentsForDisplay(
    const OccupancyComponentMap& occupancy_map,
    const bool color_unknown_components)
{
  const ColorRGBA unknown_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.5, 0.5, 0.5, 1.0);
  const auto color_fn
      = [&] (const OccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() != 0.5)
    {
      return LookupComponentColor(cell.Component());
    }
    else
    {
      if (color_unknown_components)
      {
        return LookupComponentColor(cell.Component());
      }
      else
      {
        return unknown_color;
      }
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<OccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "connected_components";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportIndexMapForDisplay(
    const OccupancyComponentMap& occupancy_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const OccupancyComponentCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndexMapToRViz<OccupancyComponentCell>(
      occupancy_map, index_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportIndicesForDisplay(
    const OccupancyComponentMap& occupancy_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const OccupancyComponentCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndicesToRViz<OccupancyComponentCell>(
      occupancy_map, indices, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

OccupancyComponentMapMessage GetMessageRepresentation(
    const OccupancyComponentMap& map)
{
  OccupancyComponentMapMessage map_message;
  map_message.header.frame_id = map.Frame();
  std::vector<uint8_t> buffer;
  OccupancyComponentMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

OccupancyComponentMap LoadFromMessageRepresentation(
    const OccupancyComponentMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return OccupancyComponentMap::Deserialize(decompressed_map, 0).Value();
  }
  else
  {
    return OccupancyComponentMap::Deserialize(
        message.serialized_map, 0).Value();
  }
}

Marker ExportForDisplay(
    const DynamicSpatialHashedOccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const OccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() > 0.5)
    {
      return collision_color;
    }
    else if (cell.Occupancy() < 0.5)
    {
      return free_color;
    }
    else
    {
      return unknown_color;
    }
  };
  auto display_rep
      = ExportDynamicSpatialHashedVoxelGridToRViz<OccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportForSeparateDisplay(
    const DynamicSpatialHashedOccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportForDisplay(occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_only";
  Marker free_only_marker
      = ExportForDisplay(occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_only";
  Marker unknown_only_marker
      = ExportForDisplay(occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

DynamicSpatialHashedOccupancyMapMessage GetMessageRepresentation(
    const DynamicSpatialHashedOccupancyMap& map)
{
  DynamicSpatialHashedOccupancyMapMessage map_message;
  map_message.header.frame_id = map.Frame();
  std::vector<uint8_t> buffer;
  DynamicSpatialHashedOccupancyMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

DynamicSpatialHashedOccupancyMap LoadFromMessageRepresentation(
    const DynamicSpatialHashedOccupancyMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return DynamicSpatialHashedOccupancyMap::Deserialize(
        decompressed_map, 0).Value();
  }
  else
  {
    return DynamicSpatialHashedOccupancyMap::Deserialize(
        message.serialized_map, 0).Value();
  }
}

Marker ExportForDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const TaggedObjectOccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() > 0.5)
    {
      return collision_color;
    }
    else if (cell.Occupancy() < 0.5)
    {
      return free_color;
    }
    else
    {
      return unknown_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportForDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const std::map<uint32_t, ColorRGBA>& object_color_map)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const bool specifies_colors = object_color_map.empty();
  const auto color_fn
      = [&] (const TaggedObjectOccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    const uint32_t object_id = cell.ObjectId();
    if (specifies_colors)
    {
      const auto found_itr = object_color_map.find(object_id);
      if (found_itr != object_color_map.end())
      {
        return found_itr->second;
      }
      else
      {
        return no_color;
      }
    }
    else
    {
      return LookupComponentColor(object_id, 1.0);
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_objects";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportForSeparateDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportForDisplay(occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_only";
  Marker free_only_marker
      = ExportForDisplay(occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_only";
  Marker unknown_only_marker
      = ExportForDisplay(occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportSurfacesForDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const TaggedObjectOccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (occupancy_map.IsSurfaceIndex(index))
    {
      if (cell.Occupancy() > 0.5)
      {
        return collision_color;
      }
      else if (cell.Occupancy() < 0.5)
      {
        return free_color;
      }
      else
      {
        return unknown_color;
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_surfaces";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportSurfacesForDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const std::map<uint32_t, ColorRGBA>& object_color_map)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const bool specifies_colors = object_color_map.empty();
  const auto color_fn
      = [&] (const TaggedObjectOccupancyCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (occupancy_map.IsSurfaceIndex(index))
    {
      const uint32_t object_id = cell.ObjectId();
      if (specifies_colors)
      {
        const auto found_itr = object_color_map.find(object_id);
        if (found_itr != object_color_map.end())
        {
          return found_itr->second;
        }
        else
        {
          return no_color;
        }
      }
      else
      {
        return LookupComponentColor(object_id, 1.0);
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_surfaces";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportSurfacesForSeparateDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_surfaces_only";
  Marker free_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_surfaces_only";
  Marker unknown_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_surfaces_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportIndexMapForDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const TaggedObjectOccupancyCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndexMapToRViz<TaggedObjectOccupancyCell>(
      occupancy_map, index_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportIndicesForDisplay(
    const TaggedObjectOccupancyMap& occupancy_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const TaggedObjectOccupancyCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndicesToRViz<TaggedObjectOccupancyCell>(
      occupancy_map, indices, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

TaggedObjectOccupancyMapMessage GetMessageRepresentation(
    const TaggedObjectOccupancyMap& map)
{
  TaggedObjectOccupancyMapMessage map_message;
  map_message.header.frame_id = map.Frame();
  std::vector<uint8_t> buffer;
  TaggedObjectOccupancyMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

TaggedObjectOccupancyMap LoadFromMessageRepresentation(
    const TaggedObjectOccupancyMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return TaggedObjectOccupancyMap::Deserialize(decompressed_map, 0).Value();
  }
  else
  {
    return TaggedObjectOccupancyMap::Deserialize(
        message.serialized_map, 0).Value();
  }
}

Marker ExportForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() > 0.5)
    {
      return collision_color;
    }
    else if (cell.Occupancy() < 0.5)
    {
      return free_color;
    }
    else
    {
      return unknown_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const std::map<uint32_t, ColorRGBA>& object_color_map)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const bool specifies_colors = object_color_map.empty();
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    const uint32_t object_id = cell.ObjectId();
    if (specifies_colors)
    {
      const auto found_itr = object_color_map.find(object_id);
      if (found_itr != object_color_map.end())
      {
        return found_itr->second;
      }
      else
      {
        return no_color;
      }
    }
    else
    {
      return LookupComponentColor(object_id, 1.0);
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_objects";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportForSeparateDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportForDisplay(occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_only";
  Marker free_only_marker
      = ExportForDisplay(occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_only";
  Marker unknown_only_marker
      = ExportForDisplay(occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportSurfacesForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (occupancy_map.IsSurfaceIndex(index))
    {
      if (cell.Occupancy() > 0.5)
      {
        return collision_color;
      }
      else if (cell.Occupancy() < 0.5)
      {
        return free_color;
      }
      else
      {
        return unknown_color;
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_surfaces";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportSurfacesForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const std::map<uint32_t, ColorRGBA>& object_color_map)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const bool specifies_colors = object_color_map.empty();
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (occupancy_map.IsSurfaceIndex(index))
    {
      const uint32_t object_id = cell.ObjectId();
      if (specifies_colors)
      {
        const auto found_itr = object_color_map.find(object_id);
        if (found_itr != object_color_map.end())
        {
          return found_itr->second;
        }
        else
        {
          return no_color;
        }
      }
      else
      {
        return LookupComponentColor(object_id, 1.0);
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_surfaces";
  display_rep.id = 1;
  return display_rep;
}

MarkerArray ExportSurfacesForSeparateDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color)
{
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  Marker collision_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_surfaces_only";
  Marker free_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_surfaces_only";
  Marker unknown_only_marker
      = ExportSurfacesForDisplay(
          occupancy_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_surfaces_only";
  MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

Marker ExportConnectedComponentsForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const bool color_unknown_components)
{
  const ColorRGBA unknown_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.5, 0.5, 0.5, 1.0);
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (cell.Occupancy() != 0.5)
    {
      return LookupComponentColor(cell.Component());
    }
    else
    {
      if (color_unknown_components)
      {
        return LookupComponentColor(cell.Component());
      }
      else
      {
        return unknown_color;
      }
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "connected_components";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportSpatialSegmentForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const uint32_t object_id, const uint32_t spatial_segment)
{
  const uint32_t number_of_segments
      = occupancy_map.NumSpatialSegments().Value();
  const ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if ((cell.ObjectId() == object_id)
        && (cell.SpatialSegment() == spatial_segment))
    {
      if (number_of_segments < 22)
      {
        return LookupComponentColor(cell.SpatialSegment());
      }
      else
      {
        return common_robotics_utilities::color_builder
            ::InterpolateHotToCold<ColorRGBA>(
                static_cast<double>(cell.SpatialSegment()),
                0.0,
                static_cast<double>(number_of_segments));
      }
    }
    else
    {
      return no_color;
    }
  };
  auto display_rep
      = ExportVoxelGridToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "object_"
                   + std::to_string(object_id)
                   + "_spatial_segment_"
                   + std::to_string(spatial_segment);
  display_rep.id = 1;
  return display_rep;
}

Marker ExportIndexMapForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep
      = ExportVoxelGridIndexMapToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, index_map, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

Marker ExportIndicesForDisplay(
    const TaggedObjectOccupancyComponentMap& occupancy_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const TaggedObjectOccupancyComponentCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep
      = ExportVoxelGridIndicesToRViz<TaggedObjectOccupancyComponentCell>(
          occupancy_map, indices, occupancy_map.Frame(), color_fn);
  display_rep.ns = "occupancy_map_surface";
  display_rep.id = 1;
  return display_rep;
}

TaggedObjectOccupancyComponentMapMessage GetMessageRepresentation(
    const TaggedObjectOccupancyComponentMap& map)
{
  TaggedObjectOccupancyComponentMapMessage map_message;
  map_message.header.frame_id = map.Frame();
  std::vector<uint8_t> buffer;
  TaggedObjectOccupancyComponentMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

TaggedObjectOccupancyComponentMap LoadFromMessageRepresentation(
    const TaggedObjectOccupancyComponentMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return TaggedObjectOccupancyComponentMap::Deserialize(
        decompressed_map, 0).Value();
  }
  else
  {
    return TaggedObjectOccupancyComponentMap::Deserialize(
        message.serialized_map, 0).Value();
  }
}
}  // namespace ros_interface
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
