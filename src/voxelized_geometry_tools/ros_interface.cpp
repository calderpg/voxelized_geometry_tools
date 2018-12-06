#include <voxelized_geometry_tools/ros_interface.hpp>

#include <vector>

#include <Eigen/Geometry>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

#include <common_robotics_utilities/color_builder.hpp>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/CollisionMapMessage.h>
#include <voxelized_geometry_tools/tagged_object_collision_map.hpp>
#include <voxelized_geometry_tools/TaggedObjectCollisionMapMessage.h>

namespace voxelized_geometry_tools
{
namespace ros_interface
{
visualization_msgs::Marker ExportForDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const CollisionCell& cell,
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
      = ExportVoxelGridToRViz<CollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "collision_map";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::MarkerArray ExportForSeparateDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  visualization_msgs::Marker collision_only_marker
      = ExportForDisplay(collision_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_only";
  visualization_msgs::Marker free_only_marker
      = ExportForDisplay(collision_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_only";
  visualization_msgs::Marker unknown_only_marker
      = ExportForDisplay(collision_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_only";
  visualization_msgs::MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

visualization_msgs::Marker ExportSurfacesForDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const CollisionCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (collision_map.IsSurfaceIndex(index))
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
      = ExportVoxelGridToRViz<CollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "collision_surfaces";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::MarkerArray ExportSurfacesForSeparateDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  visualization_msgs::Marker collision_only_marker
      = ExportSurfacesForDisplay(
          collision_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "collision_surfaces_only";
  visualization_msgs::Marker free_only_marker
      = ExportSurfacesForDisplay(
          collision_map, no_color, free_color, no_color);
  free_only_marker.ns = "free_surfaces_only";
  visualization_msgs::Marker unknown_only_marker
      = ExportSurfacesForDisplay(
          collision_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "unknown_surfaces_only";
  visualization_msgs::MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

visualization_msgs::Marker ExportConnectedComponentsForDisplay(
    const CollisionMap& collision_map,
    const bool color_unknown_components)
{
  const std_msgs::ColorRGBA unknown_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.5, 0.5, 0.5, 1.0);
  const auto color_fn
      = [&] (const CollisionCell& cell,
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
      = ExportVoxelGridToRViz<CollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "connected_components";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::Marker ExportIndexMapForDisplay(
    const CollisionMap& collision_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const std_msgs::ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const CollisionCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndexMapToRViz<CollisionCell>(
      collision_map, index_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "collision_map_surface";
  display_rep.id = 1;
  return display_rep;
}

CollisionMapMessage GetMessageRepresentation(const CollisionMap& map)
{
  CollisionMapMessage map_message;
  map_message.header.stamp = ros::Time::now();
  map_message.header.frame_id = map.GetFrame();
  std::vector<uint8_t> buffer;
  CollisionMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

CollisionMap LoadFromMessageRepresentation(const CollisionMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return CollisionMap::Deserialize(decompressed_map, 0).first;
  }
  else
  {
    return CollisionMap::Deserialize(message.serialized_map, 0).first;
  }
}

visualization_msgs::MarkerArray ExportForDisplay(
    const DynamicSpatialHashedCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const CollisionCell& cell,
             const Eigen::Vector4d&)
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
      = ExportDynamicSpatialHashedVoxelGridToRViz<CollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn, color_fn);
  display_rep.first.ns = "dsh_collision_map_chunks";
  display_rep.first.id = 1;
  display_rep.second.ns = "dsh_collision_map_cells";
  display_rep.second.id = 1;
  visualization_msgs::MarkerArray display_markers;
  display_markers.markers = {display_rep.first, display_rep.second};
  return display_markers;
}

visualization_msgs::MarkerArray ExportForSeparateDisplay(
    const DynamicSpatialHashedCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  visualization_msgs::MarkerArray collision_only_markers
      = ExportForDisplay(collision_map, collision_color, no_color, no_color);
  for (auto& collision_only_marker : collision_only_markers.markers)
  {
   collision_only_marker.ns += "_collision_only";
  }
  visualization_msgs::MarkerArray free_only_markers
      = ExportForDisplay(collision_map, no_color, free_color, no_color);
  for (auto& free_only_marker : free_only_markers.markers)
  {
    free_only_marker.ns += "_free_only";
  }
  visualization_msgs::MarkerArray unknown_only_markers
      = ExportForDisplay(collision_map, no_color, no_color, unknown_color);
  for (auto& unknown_only_marker : unknown_only_markers.markers)
  {
    unknown_only_marker.ns += "_unknown_only";
  }
  visualization_msgs::MarkerArray display_messages;
  display_messages.markers.insert(display_messages.markers.end(),
                                  collision_only_markers.markers.begin(),
                                  collision_only_markers.markers.end());
  display_messages.markers.insert(display_messages.markers.end(),
                                  free_only_markers.markers.begin(),
                                  free_only_markers.markers.end());
  display_messages.markers.insert(display_messages.markers.end(),
                                  unknown_only_markers.markers.begin(),
                                  unknown_only_markers.markers.end());
  return display_messages;
}

DynamicSpatialHashedCollisionMapMessage GetMessageRepresentation(
    const DynamicSpatialHashedCollisionMap& map)
{
  DynamicSpatialHashedCollisionMapMessage map_message;
  map_message.header.stamp = ros::Time::now();
  map_message.header.frame_id = map.GetFrame();
  std::vector<uint8_t> buffer;
  DynamicSpatialHashedCollisionMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

DynamicSpatialHashedCollisionMap LoadFromMessageRepresentation(
    const DynamicSpatialHashedCollisionMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return DynamicSpatialHashedCollisionMap::Deserialize(
        decompressed_map, 0).first;
  }
  else
  {
    return DynamicSpatialHashedCollisionMap::Deserialize(
        message.serialized_map, 0).first;
  }
}

visualization_msgs::Marker ExportForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell& cell,
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
      = ExportVoxelGridToRViz<TaggedObjectCollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_collision_map";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::Marker ExportForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::map<uint32_t, std_msgs::ColorRGBA>& object_color_map)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const bool specifies_colors = object_color_map.empty();
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell& cell,
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
      = ExportVoxelGridToRViz<TaggedObjectCollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_collision_map";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::MarkerArray ExportForSeparateDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  visualization_msgs::Marker collision_only_marker
      = ExportForDisplay(collision_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "tagged_object_collision_only";
  visualization_msgs::Marker free_only_marker
      = ExportForDisplay(collision_map, no_color, free_color, no_color);
  free_only_marker.ns = "tagged_object_free_only";
  visualization_msgs::Marker unknown_only_marker
      = ExportForDisplay(collision_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "tagged_object_unknown_only";
  visualization_msgs::MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

visualization_msgs::Marker ExportSurfacesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (collision_map.IsSurfaceIndex(index))
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
      = ExportVoxelGridToRViz<TaggedObjectCollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_collision_surfaces";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::Marker ExportSurfacesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::map<uint32_t, std_msgs::ColorRGBA>& object_color_map)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const bool specifies_colors = object_color_map.empty();
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell& cell,
             const common_robotics_utilities::voxel_grid::GridIndex& index)
  {
    if (collision_map.IsSurfaceIndex(index))
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
      = ExportVoxelGridToRViz<TaggedObjectCollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_collision_surfaces";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::MarkerArray ExportSurfacesForSeparateDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color)
{
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  visualization_msgs::Marker collision_only_marker
      = ExportSurfacesForDisplay(
          collision_map, collision_color, no_color, no_color);
  collision_only_marker.ns = "tagged_object_collision_surfaces_only";
  visualization_msgs::Marker free_only_marker
      = ExportSurfacesForDisplay(
          collision_map, no_color, free_color, no_color);
  free_only_marker.ns = "tagged_object_free_surfaces_only";
  visualization_msgs::Marker unknown_only_marker
      = ExportSurfacesForDisplay(
          collision_map, no_color, no_color, unknown_color);
  unknown_only_marker.ns = "tagged_object_unknown_surfaces_only";
  visualization_msgs::MarkerArray display_messages;
  display_messages.markers = {collision_only_marker,
                              free_only_marker,
                              unknown_only_marker};
  return display_messages;
}

visualization_msgs::Marker ExportConnectedComponentsForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const bool color_unknown_components)
{
  const std_msgs::ColorRGBA unknown_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.5, 0.5, 0.5, 1.0);
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell& cell,
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
      = ExportVoxelGridToRViz<TaggedObjectCollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_connected_components";
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::Marker ExportSpatialSegmentForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const uint32_t object_id, const uint32_t spatial_segment)
{
  const uint32_t number_of_segments
      = collision_map.GetNumSpatialSegments().Value();
  const std_msgs::ColorRGBA no_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell& cell,
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
            ::InterpolateHotToCold<std_msgs::ColorRGBA>(
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
      = ExportVoxelGridToRViz<TaggedObjectCollisionCell>(
          collision_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_"
                   + std::to_string(object_id)
                   + "_spatial_segment_"
                   + std::to_string(spatial_segment);
  display_rep.id = 1;
  return display_rep;
}

visualization_msgs::Marker ExportIndexMapForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const std_msgs::ColorRGBA& surface_color)
{
  const auto color_fn
      = [&] (const TaggedObjectCollisionCell&,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    return surface_color;
  };
  auto display_rep = ExportVoxelGridIndexMapToRViz<TaggedObjectCollisionCell>(
      collision_map, index_map, collision_map.GetFrame(), color_fn);
  display_rep.ns = "tagged_object_collision_map_surface";
  display_rep.id = 1;
  return display_rep;
}

TaggedObjectCollisionMapMessage GetMessageRepresentation(
    const TaggedObjectCollisionMap& map)
{
  TaggedObjectCollisionMapMessage map_message;
  map_message.header.stamp = ros::Time::now();
  map_message.header.frame_id = map.GetFrame();
  std::vector<uint8_t> buffer;
  TaggedObjectCollisionMap::Serialize(map, buffer);
  map_message.serialized_map
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  map_message.is_compressed = true;
  return map_message;
}

TaggedObjectCollisionMap LoadFromMessageRepresentation(
    const TaggedObjectCollisionMapMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_map
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_map);
    return TaggedObjectCollisionMap::Deserialize(decompressed_map, 0).first;
  }
  else
  {
    return TaggedObjectCollisionMap::Deserialize(
        message.serialized_map, 0).first;
  }
}
}  // namespace ros_interface
}  // namespace voxelized_geometry_tools

