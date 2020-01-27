#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

#include <common_robotics_utilities/color_builder.hpp>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/ros_conversions.hpp>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/CollisionMapMessage.h>
#include <voxelized_geometry_tools/dynamic_spatial_hashed_collision_map.hpp>
#include <voxelized_geometry_tools/DynamicSpatialHashedCollisionMapMessage.h>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/SignedDistanceFieldMessage.h>
#include <voxelized_geometry_tools/tagged_object_collision_map.hpp>
#include <voxelized_geometry_tools/TaggedObjectCollisionMapMessage.h>

namespace voxelized_geometry_tools
{
namespace ros_interface
{
inline std_msgs::ColorRGBA LookupComponentColor(
    const uint32_t component, const float alpha=1.0f)
{
  return common_robotics_utilities::color_builder
      ::LookupUniqueColor<std_msgs::ColorRGBA>(component, alpha);
}

template<typename T, typename BackingStore=std::vector<T>>
inline visualization_msgs::Marker ExportVoxelGridToRViz(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& voxel_grid,
    const std::string& frame,
    const std::function<std_msgs::ColorRGBA(
        const T&, const common_robotics_utilities
                      ::voxel_grid::GridIndex&)>& voxel_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  visualization_msgs::Marker display_rep;
  // Populate the header
  display_rep.header.frame_id = frame;
  // Populate the options
  display_rep.ns = "";
  display_rep.id = 0;
  display_rep.type = visualization_msgs::Marker::CUBE_LIST;
  display_rep.action = visualization_msgs::Marker::ADD;
  display_rep.lifetime = ros::Duration(0.0);
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
            = voxel_grid.GetImmutable(x_index, y_index, z_index).Value();
        const std_msgs::ColorRGBA cell_color
            = voxel_color_fn(cell_value, GridIndex(x_index, y_index, z_index));
        if (cell_color.a > 0.0f)
        {
          // Convert indices into a real-world location
          const Eigen::Vector4d location
              = voxel_grid.GridIndexToLocationInGridFrame(
                  x_index, y_index, z_index);
          const geometry_msgs::Point cell_point
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
inline visualization_msgs::Marker ExportVoxelGridIndexMapToRViz(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& voxel_grid,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const std::string& frame,
    const std::function<std_msgs::ColorRGBA(
        const T&, const common_robotics_utilities
                      ::voxel_grid::GridIndex&)>& voxel_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  visualization_msgs::Marker display_rep;
  // Populate the header
  display_rep.header.frame_id = frame;
  // Populate the options
  display_rep.ns = "";
  display_rep.id = 0;
  display_rep.type = visualization_msgs::Marker::CUBE_LIST;
  display_rep.action = visualization_msgs::Marker::ADD;
  display_rep.lifetime = ros::Duration(0.0);
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
      const auto cell_value = voxel_grid.GetImmutable(index).Value();
      const std_msgs::ColorRGBA cell_color = voxel_color_fn(cell_value, index);
      if (cell_color.a > 0.0f)
      {
        // Convert indices into a real-world location
        const Eigen::Vector4d location
            = voxel_grid.GridIndexToLocationInGridFrame(index);
        const geometry_msgs::Point cell_point
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
inline visualization_msgs::Marker ExportVoxelGridIndicesToRViz(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& voxel_grid,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const std::string& frame,
    const std::function<std_msgs::ColorRGBA(
        const T&, const common_robotics_utilities
                      ::voxel_grid::GridIndex&)>& voxel_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  visualization_msgs::Marker display_rep;
  // Populate the header
  display_rep.header.frame_id = frame;
  // Populate the options
  display_rep.ns = "";
  display_rep.id = 0;
  display_rep.type = visualization_msgs::Marker::CUBE_LIST;
  display_rep.action = visualization_msgs::Marker::ADD;
  display_rep.lifetime = ros::Duration(0.0);
  display_rep.frame_locked = false;
  display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(voxel_grid.GetOriginTransform());
  display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(voxel_grid.GetCellSizes());
  for (const GridIndex& index : indices)
  {
    const auto cell_value = voxel_grid.GetImmutable(index).Value();
    const std_msgs::ColorRGBA cell_color = voxel_color_fn(cell_value, index);
    if (cell_color.a > 0.0f)
    {
      // Convert indices into a real-world location
      const Eigen::Vector4d location
          = voxel_grid.GridIndexToLocationInGridFrame(index);
      const geometry_msgs::Point cell_point
          = common_robotics_utilities::ros_conversions
              ::EigenVector4dToGeometryPoint(location);
      display_rep.points.push_back(cell_point);
      display_rep.colors.push_back(cell_color);
    }
  }
  return display_rep;
}

template<typename T, typename BackingStore=std::vector<T>>
inline std::pair<visualization_msgs::Marker, visualization_msgs::Marker>
ExportDynamicSpatialHashedVoxelGridToRViz(
    const common_robotics_utilities::voxel_grid
        ::DynamicSpatialHashedVoxelGridBase<T, BackingStore>& dsh_voxel_grid,
    const std::string& frame,
    const std::function<std_msgs::ColorRGBA(
        const T&, const Eigen::Vector4d&)>& chunk_color_fn,
    const std::function<std_msgs::ColorRGBA(
        const T&, const Eigen::Vector4d&)>& cell_color_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  using common_robotics_utilities::voxel_grid::DSHVGFillStatus;
  visualization_msgs::Marker chunks_display_rep;
  // Populate the header
  chunks_display_rep.header.frame_id = frame;
  // Populate the options
  chunks_display_rep.ns = "";
  chunks_display_rep.id = 0;
  chunks_display_rep.type = visualization_msgs::Marker::CUBE_LIST;
  chunks_display_rep.action = visualization_msgs::Marker::ADD;
  chunks_display_rep.lifetime = ros::Duration(0.0);
  chunks_display_rep.frame_locked = false;
  chunks_display_rep.pose
      = common_robotics_utilities::ros_conversions
          ::EigenIsometry3dToGeometryPose(dsh_voxel_grid.GetOriginTransform());
  chunks_display_rep.scale
      = common_robotics_utilities::ros_conversions
          ::EigenVector3dToGeometryVector3(dsh_voxel_grid.GetChunkSizes());
  visualization_msgs::Marker cells_display_rep;
  // Populate the header
  cells_display_rep.header.frame_id = frame;
  // Populate the options
  cells_display_rep.ns = "";
  cells_display_rep.id = 0;
  cells_display_rep.type = visualization_msgs::Marker::CUBE_LIST;
  cells_display_rep.action = visualization_msgs::Marker::ADD;
  cells_display_rep.lifetime = ros::Duration(0.0);
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
      const std_msgs::ColorRGBA chunk_color
          = chunk_color_fn(current_chunk.GetImmutable(chunk_center).Value(),
                           chunk_center);
      if (chunk_color.a > 0.0f)
      {
        const geometry_msgs::Point chunk_center_point
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
            const std_msgs::ColorRGBA cell_color
                = cell_color_fn(
                    current_chunk.GetImmutableInternal(internal_index).Value(),
                    cell_center);
            if (cell_color.a > 0.0f)
            {
              const geometry_msgs::Point cell_center_point
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

template<typename BackingStore=std::vector<float>>
inline visualization_msgs::Marker ExportSDFForDisplay(
    const SignedDistanceField<BackingStore>& sdf,
    const float alpha = 0.01f)
{
  float min_distance = 0.0;
  float max_distance = 0.0;
  for (int64_t x_index = 0; x_index < sdf.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < sdf.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < sdf.GetNumZCells(); z_index++)
      {
        // Update minimum/maximum distance variables
        const float distance
            = sdf.GetImmutable(x_index, y_index, z_index).Value();
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
      = [&] (const float& distance,
             const common_robotics_utilities::voxel_grid::GridIndex&)
  {
    std_msgs::ColorRGBA new_color;
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
  auto display_rep = ExportVoxelGridToRViz<float, BackingStore>(
      sdf, sdf.GetFrame(), color_fn);
  display_rep.ns = "sdf_distance";
  display_rep.id = 1;
  return display_rep;
}

template<typename BackingStore=std::vector<float>>
inline visualization_msgs::Marker ExportSDFForDisplayCollisionOnly(
    const SignedDistanceField<BackingStore>& sdf,
    const float alpha = 0.01f)
{
  const std_msgs::ColorRGBA filled_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<std_msgs::ColorRGBA>(1.0, 0.0, 0.0, alpha);
  const std_msgs::ColorRGBA free_color
      = common_robotics_utilities::color_builder
        ::MakeFromFloatColors<std_msgs::ColorRGBA>(0.0, 0.0, 0.0, 0.0);
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
  auto display_rep = ExportVoxelGridToRViz<float, BackingStore>(
      sdf, sdf.GetFrame(), color_fn);
  display_rep.ns = "sdf_collision";
  display_rep.id = 1;
  return display_rep;
}

/// Convert SDF to and from ROS messages.

template<typename BackingStore=std::vector<float>>
inline SignedDistanceFieldMessage GetMessageRepresentation(
    const SignedDistanceField<BackingStore>& sdf)
{
  SignedDistanceFieldMessage sdf_message;
  sdf_message.header.stamp = ros::Time::now();
  sdf_message.header.frame_id = sdf.GetFrame();
  std::vector<uint8_t> buffer;
  SignedDistanceField<BackingStore>::Serialize(sdf, buffer);
  sdf_message.serialized_sdf
      = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
  sdf_message.is_compressed = true;
  return sdf_message;
}

template<typename BackingStore=std::vector<float>>
inline SignedDistanceField<BackingStore> LoadFromMessageRepresentation(
    const SignedDistanceFieldMessage& message)
{
  if (message.is_compressed)
  {
    const std::vector<uint8_t> decompressed_sdf
        = common_robotics_utilities::zlib_helpers::DecompressBytes(
            message.serialized_sdf);
    return SignedDistanceField<BackingStore>::Deserialize(
        decompressed_sdf, 0).Value();
  }
  else
  {
    return SignedDistanceField<BackingStore>::Deserialize(
        message.serialized_sdf, 0).Value();
  }
}

/// Export CollisionMap to RViz for display.

visualization_msgs::Marker ExportForDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::MarkerArray ExportForSeparateDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::Marker ExportSurfacesForDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::MarkerArray ExportSurfacesForSeparateDisplay(
    const CollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::Marker ExportConnectedComponentsForDisplay(
    const CollisionMap& collision_map,
    const bool color_unknown_components);

visualization_msgs::Marker ExportIndexMapForDisplay(
    const CollisionMap& collision_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const std_msgs::ColorRGBA& surface_color);

visualization_msgs::Marker ExportIndicesForDisplay(
    const CollisionMap& collision_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const std_msgs::ColorRGBA& surface_color);

/// Convert CollisionMap to and from ROS messages.

CollisionMapMessage GetMessageRepresentation(const CollisionMap& map);

CollisionMap LoadFromMessageRepresentation(const CollisionMapMessage& message);

/// Export DynamicSpatialHashedCollisionMap to RViz for display.

visualization_msgs::MarkerArray ExportForDisplay(
    const DynamicSpatialHashedCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::MarkerArray ExportForSeparateDisplay(
    const DynamicSpatialHashedCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

DynamicSpatialHashedCollisionMapMessage GetMessageRepresentation(
    const DynamicSpatialHashedCollisionMap& map);

DynamicSpatialHashedCollisionMap LoadFromMessageRepresentation(
    const DynamicSpatialHashedCollisionMapMessage& message);

/// Export TaggedObjectCollisionMap to RViz for display.

visualization_msgs::Marker ExportForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::Marker ExportForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::map<uint32_t, std_msgs::ColorRGBA>& object_color_map
        =std::map<uint32_t, std_msgs::ColorRGBA>());

visualization_msgs::MarkerArray ExportForSeparateDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::Marker ExportSurfacesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::Marker ExportSurfacesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::map<uint32_t, std_msgs::ColorRGBA>& object_color_map
        =std::map<uint32_t, std_msgs::ColorRGBA>());

visualization_msgs::MarkerArray ExportSurfacesForSeparateDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std_msgs::ColorRGBA& collision_color,
    const std_msgs::ColorRGBA& free_color,
    const std_msgs::ColorRGBA& unknown_color);

visualization_msgs::Marker ExportConnectedComponentsForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const bool color_unknown_components);

visualization_msgs::Marker ExportSpatialSegmentForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const uint32_t object_id, const uint32_t spatial_segment);

visualization_msgs::Marker ExportIndexMapForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::unordered_map<
        common_robotics_utilities::voxel_grid::GridIndex, uint8_t>& index_map,
    const std_msgs::ColorRGBA& surface_color);

visualization_msgs::Marker ExportIndicesForDisplay(
    const TaggedObjectCollisionMap& collision_map,
    const std::vector<common_robotics_utilities::voxel_grid::GridIndex>&
        indices,
    const std_msgs::ColorRGBA& surface_color);

/// Convert TaggedObjectCollisionMap to and from ROS messages.

TaggedObjectCollisionMapMessage GetMessageRepresentation(
    const TaggedObjectCollisionMap& map);

TaggedObjectCollisionMap LoadFromMessageRepresentation(
    const TaggedObjectCollisionMapMessage& message);
}  // namespace ros_interface
}  // namespace voxelized_geometry_tools
