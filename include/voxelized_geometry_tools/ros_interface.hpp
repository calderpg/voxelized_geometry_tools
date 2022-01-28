#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#else
#error "Undefined or unknown VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION"
#endif

#include <common_robotics_utilities/color_builder.hpp>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/ros_conversions.hpp>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/dynamic_spatial_hashed_collision_map.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/tagged_object_collision_map.hpp>

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
#include <voxelized_geometry_tools/msg/collision_map_message.hpp>
#include <voxelized_geometry_tools/msg/dynamic_spatial_hashed_collision_map_message.hpp>
#include <voxelized_geometry_tools/msg/signed_distance_field_message.hpp>
#include <voxelized_geometry_tools/msg/tagged_object_collision_map_message.hpp>
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <voxelized_geometry_tools/CollisionMapMessage.h>
#include <voxelized_geometry_tools/DynamicSpatialHashedCollisionMapMessage.h>
#include <voxelized_geometry_tools/SignedDistanceFieldMessage.h>
#include <voxelized_geometry_tools/TaggedObjectCollisionMapMessage.h>
#endif

namespace voxelized_geometry_tools
{
namespace ros_interface
{

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
using ColorRGBA = std_msgs::msg::ColorRGBA;
using Point = geometry_msgs::msg::Point;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;

using CollisionMapMessage = msg::CollisionMapMessage;
using DynamicSpatialHashedCollisionMapMessage = msg::DynamicSpatialHashedCollisionMapMessage;
using SignedDistanceFieldMessage = msg::SignedDistanceFieldMessage;
using TaggedObjectCollisionMapMessage = msg::TaggedObjectCollisionMapMessage;
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
using ColorRGBA = std_msgs::ColorRGBA;
using Point = geometry_msgs::Point;
using Marker = visualization_msgs::Marker;
using MarkerArray = visualization_msgs::MarkerArray;
#endif

inline ColorRGBA LookupComponentColor(
    const uint32_t component, const float alpha=1.0f)
{
  return common_robotics_utilities::color_builder
      ::LookupUniqueColor<ColorRGBA>(component, alpha);
}

template<typename T, typename BackingStore=std::vector<T>>
inline Marker ExportVoxelGridToRViz(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& voxel_grid,
    const std::string& frame,
    const std::function<ColorRGBA(
        const T&, const common_robotics_utilities
                      ::voxel_grid::GridIndex&)>& voxel_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  Marker display_rep;
  // Populate the header
  display_rep.header.frame_id = frame;
  // Populate the options
  display_rep.ns = "";
  display_rep.id = 0;
  display_rep.type = Marker::CUBE_LIST;
  display_rep.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  display_rep.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  display_rep.lifetime = ros::Duration(0.0);
#endif
  display_rep.frame_locked = false;
  display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(voxel_grid.GetOriginTransform());
  display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(voxel_grid.GetCellSizes());
  for (int64_t x_index = 0; x_index < voxel_grid.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < voxel_grid.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < voxel_grid.GetNumZCells(); z_index++)
      {
        const auto cell_value
            = voxel_grid.GetIndexImmutable(x_index, y_index, z_index).Value();
        const ColorRGBA cell_color
            = voxel_color_fn(cell_value, GridIndex(x_index, y_index, z_index));
        if (cell_color.a > 0.0f)
        {
          // Convert indices into a real-world location
          const Eigen::Vector4d location
              = voxel_grid.GridIndexToLocationInGridFrame(
                  x_index, y_index, z_index);
          const Point cell_point
              = common_robotics_utilities::ros_conversions
                  ::EigenVector4dToGeometryPoint(location);
          display_rep.points.push_back(cell_point);
          display_rep.colors.push_back(cell_color);
        }
      }
    }
  }
  return display_rep;
}

template<typename T, typename BackingStore=std::vector<T>>
inline Marker ExportVoxelGridIndexMapToRViz(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& voxel_grid,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const std::string& frame,
    const std::function<ColorRGBA(
        const T&, const common_robotics_utilities
                      ::voxel_grid::GridIndex&)>& voxel_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  Marker display_rep;
  // Populate the header
  display_rep.header.frame_id = frame;
  // Populate the options
  display_rep.ns = "";
  display_rep.id = 0;
  display_rep.type = Marker::CUBE_LIST;
  display_rep.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  display_rep.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  display_rep.lifetime = ros::Duration(0.0);
#endif
  display_rep.frame_locked = false;
  display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(voxel_grid.GetOriginTransform());
  display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(voxel_grid.GetCellSizes());
  for (auto surface_itr = index_map.begin(); surface_itr != index_map.end();
       ++surface_itr)
  {
    const GridIndex& index = surface_itr->first;
    const uint8_t valid = surface_itr->second;
    if (valid > 0)
    {
      const auto cell_value = voxel_grid.GetIndexImmutable(index).Value();
      const ColorRGBA cell_color = voxel_color_fn(cell_value, index);
      if (cell_color.a > 0.0f)
      {
        // Convert indices into a real-world location
        const Eigen::Vector4d location
            = voxel_grid.GridIndexToLocationInGridFrame(index);
        const Point cell_point
            = common_robotics_utilities::ros_conversions
                ::EigenVector4dToGeometryPoint(location);
        display_rep.points.push_back(cell_point);
        display_rep.colors.push_back(cell_color);
      }
    }
  }
  return display_rep;
}

template<typename T, typename BackingStore=std::vector<T>>
inline Marker ExportVoxelGridIndicesToRViz(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& voxel_grid,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const std::string& frame,
    const std::function<ColorRGBA(
        const T&, const common_robotics_utilities
                      ::voxel_grid::GridIndex&)>& voxel_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  Marker display_rep;
  // Populate the header
  display_rep.header.frame_id = frame;
  // Populate the options
  display_rep.ns = "";
  display_rep.id = 0;
  display_rep.type = Marker::CUBE_LIST;
  display_rep.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  display_rep.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  display_rep.lifetime = ros::Duration(0.0);
#endif
  display_rep.frame_locked = false;
  display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(voxel_grid.GetOriginTransform());
  display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(voxel_grid.GetCellSizes());
  for (const GridIndex& index : indices)
  {
    const auto cell_value = voxel_grid.GetIndexImmutable(index).Value();
    const ColorRGBA cell_color = voxel_color_fn(cell_value, index);
    if (cell_color.a > 0.0f)
    {
      // Convert indices into a real-world location
      const Eigen::Vector4d location
          = voxel_grid.GridIndexToLocationInGridFrame(index);
      const Point cell_point
          = common_robotics_utilities::ros_conversions
              ::EigenVector4dToGeometryPoint(location);
      display_rep.points.push_back(cell_point);
      display_rep.colors.push_back(cell_color);
    }
  }
  return display_rep;
}

template<typename T, typename BackingStore=std::vector<T>>
inline std::pair<Marker, Marker>
ExportDynamicSpatialHashedVoxelGridToRViz(
    const common_robotics_utilities::voxel_grid
        ::DynamicSpatialHashedVoxelGridBase<T, BackingStore>& dsh_voxel_grid,
    const std::string& frame,
    const std::function<ColorRGBA(
        const T&, const Eigen::Vector4d&)>& chunk_color_fn,
    const std::function<ColorRGBA(
        const T&, const Eigen::Vector4d&)>& cell_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  using common_robotics_utilities::voxel_grid::DSHVGFillStatus;
  Marker chunks_display_rep;
  // Populate the header
  chunks_display_rep.header.frame_id = frame;
  // Populate the options
  chunks_display_rep.ns = "";
  chunks_display_rep.id = 0;
  chunks_display_rep.type = Marker::CUBE_LIST;
  chunks_display_rep.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  chunks_display_rep.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  chunks_display_rep.lifetime = ros::Duration(0.0);
#endif
  chunks_display_rep.frame_locked = false;
  chunks_display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(dsh_voxel_grid.GetOriginTransform());
  chunks_display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(dsh_voxel_grid.GetChunkSizes());
  Marker cells_display_rep;
  // Populate the header
  cells_display_rep.header.frame_id = frame;
  // Populate the options
  cells_display_rep.ns = "";
  cells_display_rep.id = 0;
  cells_display_rep.type = Marker::CUBE_LIST;
  cells_display_rep.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  cells_display_rep.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  cells_display_rep.lifetime = ros::Duration(0.0);
#endif
  cells_display_rep.frame_locked = false;
  cells_display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(dsh_voxel_grid.GetOriginTransform());
  cells_display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(dsh_voxel_grid.GetCellSizes());
  // Go through the chunks in the DSH voxel grid
  const common_robotics_utilities::voxel_grid::GridSizes& chunk_grid_sizes
      = dsh_voxel_grid.GetChunkGridSizes();
  const auto& internal_chunks = dsh_voxel_grid.GetImmutableInternalChunks();
  for (auto internal_chunks_itr = internal_chunks.begin();
       internal_chunks_itr != internal_chunks.end();
       ++internal_chunks_itr)
  {
    const auto& current_chunk = internal_chunks_itr->second;
    const DSHVGFillStatus current_chunk_fill = current_chunk.FillStatus();
    if (current_chunk_fill == DSHVGFillStatus::CHUNK_FILLED)
    {
      const Eigen::Vector4d chunk_center
          = current_chunk.GetChunkCenterInGridFrame();
      const ColorRGBA chunk_color = chunk_color_fn(
          current_chunk.GetLocationImmutable4d(chunk_center).Value(),
          chunk_center);
      if (chunk_color.a > 0.0f)
      {
        const Point chunk_center_point
            = common_robotics_utilities::ros_conversions
                ::EigenVector4dToGeometryPoint(chunk_center);
        chunks_display_rep.points.push_back(chunk_center_point);
        chunks_display_rep.colors.push_back(chunk_color);
      }
    }
    else if (current_chunk_fill == DSHVGFillStatus::CELL_FILLED)
    {
      for (int64_t x_index = 0; x_index < chunk_grid_sizes.NumXCells();
           x_index++)
      {
        for (int64_t y_index = 0; y_index < chunk_grid_sizes.NumYCells();
             y_index++)
        {
          for (int64_t z_index = 0; z_index < chunk_grid_sizes.NumZCells();
               z_index++)
          {
            const GridIndex internal_index(x_index, y_index, z_index);
            const Eigen::Vector4d cell_center
                = current_chunk.GetCellLocationInGridFrame(internal_index);
            const ColorRGBA cell_color
                = cell_color_fn(
                    current_chunk.GetIndexImmutable(internal_index).Value(),
                    cell_center);
            if (cell_color.a > 0.0f)
            {
              const Point cell_center_point
                  = common_robotics_utilities::ros_conversions
                      ::EigenVector4dToGeometryPoint(cell_center);
              cells_display_rep.points.push_back(cell_center_point);
              cells_display_rep.colors.push_back(cell_color);
            }
          }
        }
      }
    }
  }
  return std::make_pair(chunks_display_rep, cells_display_rep);
}

/// Export SDF to RViz display.

template<typename ScalarType>
inline Marker ExportSDFForDisplay(
    const SignedDistanceField<ScalarType>& sdf, const float alpha = 0.01f)
{
  ScalarType min_distance = 0.0;
  ScalarType max_distance = 0.0;
  for (int64_t x_index = 0; x_index < sdf.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < sdf.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < sdf.GetNumZCells(); z_index++)
      {
        // Update minimum/maximum distance variables
        const ScalarType distance
            = sdf.GetIndexImmutable(x_index, y_index, z_index).Value();
        if (distance < min_distance)
        {
          min_distance = distance;
        }
        if (distance > max_distance)
        {
          max_distance = distance;
        }
      }
    }
  }
  const auto color_fn
      = [&] (const ScalarType& distance,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    ColorRGBA new_color;
    new_color.a
        = common_robotics_utilities::utility::ClampValue(alpha, 0.0f, 1.0f);
    if (distance > 0.0)
    {
      new_color.b = 0.0;
      new_color.g
          = static_cast<float>(std::abs(distance / max_distance) * 0.8) + 0.2f;
      new_color.r = 0.0;
    }
    else if (distance < 0.0)
    {
      new_color.b = 0.0;
      new_color.g = 0.0;
      new_color.r
          = static_cast<float>(std::abs(distance / min_distance) * 0.8) + 0.2f;
    }
    else
    {
      new_color.b = 1.0;
      new_color.g = 0.0;
      new_color.r = 0.0;
    }
    return new_color;
  };
  auto display_rep =
      ExportVoxelGridToRViz<ScalarType>(sdf, sdf.GetFrame(), color_fn);
  display_rep.ns = "sdf_distance";
  display_rep.id = 1;
  return display_rep;
}

template<typename ScalarType>
inline Marker ExportSDFForDisplayCollisionOnly(
    const SignedDistanceField<ScalarType>& sdf, const float alpha = 0.01f)
{
  const ColorRGBA filled_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(1.0, 0.0, 0.0, alpha);
  const ColorRGBA free_color
      = common_robotics_utilities::color_builder
        ::MakeFromFloatColors<ColorRGBA>(0.0, 0.0, 0.0, 0.0);
  const auto color_fn
      = [&] (const float& distance,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    if (distance <= 0.0)
    {
      return filled_color;
    }
    else
    {
      return free_color;
    }
  };
  auto display_rep =
      ExportVoxelGridToRViz<ScalarType>(sdf, sdf.GetFrame(), color_fn);
  display_rep.ns = "sdf_collision";
  display_rep.id = 1;
  return display_rep;
}

/// Convert SDF to and from ROS messages.

template<typename ScalarType>
inline SignedDistanceFieldMessage GetMessageRepresentation(
    const SignedDistanceField<ScalarType>& sdf)
{
  SignedDistanceFieldMessage sdf_message;
  sdf_message.header.frame_id = sdf.GetFrame();
  std::vector<uint8_t> buffer;
  SignedDistanceField<ScalarType>::Serialize(sdf, buffer);
  sdf_message.serialized_sdf
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  sdf_message.is_compressed = true;
  return sdf_message;
}

template<typename ScalarType>
inline SignedDistanceField<ScalarType> LoadFromMessageRepresentation(
    const SignedDistanceFieldMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_sdf
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_sdf);
    return SignedDistanceField<ScalarType>::Deserialize(
        decompressed_sdf, 0).Value();
  }
  else
  {
    return SignedDistanceField<ScalarType>::Deserialize(
        message.serialized_sdf, 0).Value();
  }
}

/// Export CollisionMap to RViz for display.

Marker ExportForDisplay(
    const CollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

MarkerArray ExportForSeparateDisplay(
    const CollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

Marker ExportSurfacesForDisplay(
    const CollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

MarkerArray ExportSurfacesForSeparateDisplay(
    const CollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

Marker ExportConnectedComponentsForDisplay(
    const CollisionMap& collision_map,
    const bool color_unknown_components);

Marker ExportIndexMapForDisplay(
    const CollisionMap& collision_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const ColorRGBA& surface_color);

Marker ExportIndicesForDisplay(
    const CollisionMap& collision_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const ColorRGBA& surface_color);

/// Convert CollisionMap to and from ROS messages.

CollisionMapMessage GetMessageRepresentation(const CollisionMap& map);

CollisionMap LoadFromMessageRepresentation(const CollisionMapMessage& message);

/// Export DynamicSpatialHashedCollisionMap to RViz for display.

MarkerArray ExportForDisplay(
    const DynamicSpatialHashedCollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

MarkerArray ExportForSeparateDisplay(
    const DynamicSpatialHashedCollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

DynamicSpatialHashedCollisionMapMessage GetMessageRepresentation(
    const DynamicSpatialHashedCollisionMap& map);

DynamicSpatialHashedCollisionMap LoadFromMessageRepresentation(
    const DynamicSpatialHashedCollisionMapMessage& message);

/// Export TaggedObjectCollisionMap to RViz for display.

Marker ExportForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

Marker ExportForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::map<uint32_t, ColorRGBA>& object_color_map
        =std::map<uint32_t, ColorRGBA>());

MarkerArray ExportForSeparateDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

Marker ExportSurfacesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

Marker ExportSurfacesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::map<uint32_t, ColorRGBA>& object_color_map
        =std::map<uint32_t, ColorRGBA>());

MarkerArray ExportSurfacesForSeparateDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const ColorRGBA& collision_color,
    const ColorRGBA& free_color,
    const ColorRGBA& unknown_color);

Marker ExportConnectedComponentsForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const bool color_unknown_components);

Marker ExportSpatialSegmentForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const uint32_t object_id, const uint32_t spatial_segment);

Marker ExportIndexMapForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const ColorRGBA& surface_color);

Marker ExportIndicesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const ColorRGBA& surface_color);

/// Convert TaggedObjectCollisionMap to and from ROS messages.

TaggedObjectCollisionMapMessage GetMessageRepresentation(
    const TaggedObjectCollisionMap& map);

TaggedObjectCollisionMap LoadFromMessageRepresentation(
    const TaggedObjectCollisionMapMessage& message);
}  // namespace ros_interface
}  // namespace voxelized_geometry_tools
