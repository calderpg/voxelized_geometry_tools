#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/maybe.hpp>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/signed_distance_field_generation.hpp>
#include <voxelized_geometry_tools/topology_computation.hpp>

namespace voxelized_geometry_tools
{
class TaggedObjectCollisionCell
{
private:
  float occupancy_ = 0.0;
  uint32_t object_id_ = 0u;
  uint32_t component_ = 0u;
  uint32_t spatial_segment_ = 0u;

public:
  TaggedObjectCollisionCell()
      : occupancy_(0.0), object_id_(0u), component_(0), spatial_segment_(0u) {}

  TaggedObjectCollisionCell(const float occupancy)
      : occupancy_(occupancy), object_id_(0u),
        component_(0), spatial_segment_(0u) {}

  TaggedObjectCollisionCell(const float occupancy, const uint32_t object_id)
      : occupancy_(occupancy), object_id_(object_id),
        component_(0), spatial_segment_(0u) {}

  TaggedObjectCollisionCell(
      const float occupancy, const uint32_t object_id,
      const uint32_t component, const uint32_t spatial_segment)
      : occupancy_(occupancy), object_id_(object_id),
        component_(component), spatial_segment_(spatial_segment) {}

  const float& Occupancy() const { return occupancy_; }

  float& Occupancy() { return occupancy_; }

  const uint32_t& ObjectId() const { return object_id_; }

  uint32_t& ObjectId() { return object_id_; }

  const uint32_t& Component() const { return component_; }

  uint32_t& Component() { return component_; }

  const uint32_t& SpatialSegment() const { return spatial_segment_; }

  uint32_t& SpatialSegment() { return spatial_segment_; }
};

class TaggedObjectCollisionMap
    : public common_robotics_utilities::voxel_grid
        ::VoxelGridBase<TaggedObjectCollisionCell,
                        std::vector<TaggedObjectCollisionCell>>
{
private:
  using common_robotics_utilities::serialization::Serializer;
  using common_robotics_utilities::serialization::Deserializer;
  using common_robotics_utilities::serialization::Deserialized;
  using common_robotics_utilities::serialization::MakeDeserialized;
  using signed_distance_field_generation::SignedDistanceFieldResult;
  using topology_computation::TopologicalInvariants;

  uint32_t number_of_components_ = 0u;
  uint32_t number_of_spatial_segments_ = 0u;
  std::string frame_;
  bool components_valid_ = false;
  bool spatial_segments_valid_ = 0u;

  /// Implement the VoxelGridBase interface.

  /// We need to implement cloning.
  common_robotics_utilities::voxel_grid
      ::VoxelGridBase<TaggedObjectCollisionCell,
                      std::vector<TaggedObjectCollisionCell>>*
  DoClone() const override;

  /// We need to serialize the frame and locked flag.
  uint64_t DerivedSerializeSelf(
      std::vector<uint8_t>& buffer,
      const Serializer<TaggedObjectCollisionCell>& value_serializer)
      const override;

  /// We need to deserialize the frame and locked flag.
  uint64_t DerivedDeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
      const Deserializer<TaggedObjectCollisionCell>& value_deserializer)
      override;

  /// Invalidate connected components on mutable access.
  bool OnMutableAccess(const int64_t x_index,
                       const int64_t y_index,
                       const int64_t z_index) override;

public:
  static uint64_t Serialize(
      const TaggedObjectCollisionMap& map, std::vector<uint8_t>& buffer);

  static Deserialized<TaggedObjectCollisionMap> Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  static void SaveToFile(const TaggedObjectCollisionMap& map,
                         const std::string& filepath,
                         const bool compress);

  static TaggedObjectCollisionMap LoadFromFile(const std::string& filepath);

  TaggedObjectCollisionMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const TaggedObjectCollisionCell& default_value)
      : TaggedObjectCollisionMap(
          origin_transform, frame, sizes, default_value, default_value) {}

  TaggedObjectCollisionMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const TaggedObjectCollisionCell& default_value)
      : TaggedObjectCollisionMap(frame, sizes, default_value, default_value) {}

  TaggedObjectCollisionMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const TaggedObjectCollisionCell& default_value,
      const TaggedObjectCollisionCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<TaggedObjectCollisionCell,
                          std::vector<TaggedObjectCollisionCell>>(
              origin_transform, sizes, default_value, oob_value),
        number_of_components_(0u), frame_(frame), components_valid_(false)
  {
    if (!HasUniformCellSize())
    {
      throw std::invalid_argument(
          "Tagged object collision map cannot have non-uniform cell sizes");
    }
  }

  TaggedObjectCollisionMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const TaggedObjectCollisionCell& default_value,
      const TaggedObjectCollisionCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<TaggedObjectCollisionCell,
                          std::vector<TaggedObjectCollisionCell>>(
              sizes, default_value, oob_value), number_of_components_(0u),
        frame_(frame), components_valid_(false)
  {
    if (!HasUniformCellSize())
    {
      throw std::invalid_argument(
          "Tagged object collision map cannot have non-uniform cell sizes");
    }
  }

  TaggedObjectCollisionMap()
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<TaggedObjectCollisionCell,
                          std::vector<TaggedObjectCollisionCell>>() {}

  bool AreComponentsValid() const { return components_valid_; }

  /// Use this with great care if you know the components are still/now valid.
  void ForceComponentsToBeValid() { components_valid_ = true; }

  /// Use this to invalidate the current components.
  void ForceComponentsToBeInvalid() { components_valid_ = false; }

  bool AreSpatialSegmentsValid() const { return spatial_segments_valid_; }

  /// Use this with great care if you know the spatial segments are still/now
  /// valid.
  void ForceSpatialSegmentsToBeValid() { spatial_segments_valid_ = true; }

  /// Use this to invalidate the current spatial segments.
  void ForceSpatialSegmentsToBeInvalid() { spatial_segments_valid_ = false; }

  double GetResolution() const { return GetCellSizes().x(); }

  const std::string& GetFrame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }

  uint32_t UpdateConnectedComponents(const bool connect_across_objects);

  common_robotics_utilities::OwningMaybe<uint32_t>
  GetNumConnectedComponents() const
  {
    if (components_valid_)
    {
      return common_robotics_utilities::OwningMaybe<uint32_t>(
          number_of_components_);
    }
    else
    {
      return common_robotics_utilities::OwningMaybe<uint32_t>();
    }
  }

  common_robotics_utilities::OwningMaybe<bool> IsSurfaceIndex(
      const common_robotics_utilities::voxel_grid::GridIndex& index) const;

  common_robotics_utilities::OwningMaybe<bool> IsSurfaceIndex(
      const int64_t x_index, const int64_t y_index,
      const int64_t z_index) const;

  common_robotics_utilities::OwningMaybe<bool> IsConnectedComponentSurfaceIndex(
      const common_robotics_utilities::voxel_grid::GridIndex& index) const;

  common_robotics_utilities::OwningMaybe<bool> IsConnectedComponentSurfaceIndex(
      const int64_t x_index, const int64_t y_index,
      const int64_t z_index) const;

  common_robotics_utilities::OwningMaybe<bool> CheckIfCandidateCorner(
      const double x, const double y, const double z) const;

  common_robotics_utilities::OwningMaybe<bool> CheckIfCandidateCorner3d(
      const Eigen::Vector3d& location) const;

  common_robotics_utilities::OwningMaybe<bool> CheckIfCandidateCorner4d(
      const Eigen::Vector4d& location) const;

  common_robotics_utilities::OwningMaybe<bool> CheckIfCandidateCorner(
      const common_robotics_utilities::voxel_grid::GridIndex& index) const;

  common_robotics_utilities::OwningMaybe<bool> CheckIfCandidateCorner(
      const int64_t x_index, const int64_t y_index,
      const int64_t z_index) const;

   /// Options for handling the edges of the grid in convex segmentation:
   ///
   /// add_virtual_border=false: Uses the current grid as-is. If the outside
   /// cells (for any axis more than one layer thick) use the special value
   /// occupancy>=0.5 and object_id=0u, all interior (non-edge-layer) free and
   /// filled areas will be completely segmented. If not, artifacts or
   /// incompletely-segmented areas will result.
   ///
   /// add_virtual_border=true: Adds the equivalent of an additional layer of
   /// cells with special value occupancy>=0.5 and object_id=0u around the grid.
   /// All free and filled areas will be completely segmented. This option is
   /// the most expensive in terms of memory and computation.
  uint32_t UpdateSpatialSegments(
      const double connected_threshold, const bool add_virtual_border,
      const bool use_parallel);

  common_robotics_utilities::OwningMaybe<uint32_t> GetNumSpatialSegments() const
  {
    if (spatial_segments_valid_)
    {
      return common_robotics_utilities::OwningMaybe<uint32_t>(
          spatial_segments_valid_);
    }
    else
    {
      return common_robotics_utilities::OwningMaybe<uint32_t>();
    }
  }

  enum COMPONENT_TYPES : uint8_t { FILLED_COMPONENTS=0x01,
                                   EMPTY_COMPONENTS=0x02,
                                   UNKNOWN_COMPONENTS=0x04 };

  std::map<uint32_t, std::unordered_map<
      common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
  ExtractComponentSurfaces(
      const COMPONENT_TYPES component_types_to_extract) const;

  std::map<uint32_t, std::unordered_map<
      common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
  ExtractFilledComponentSurfaces() const;

  std::map<uint32_t, std::unordered_map<
      common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
  ExtractUnknownComponentSurfaces() const;

  std::map<uint32_t, std::unordered_map<
      common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
  ExtractEmptyComponentSurfaces() const;

  TopologicalInvariants ComputeComponentTopology(
      const COMPONENT_TYPES component_types_to_use,
      const bool connect_across_objects, const bool verbose);

  template<typename BackingStore=std::vector<float>>
  SignedDistanceFieldResult<BackingStore>
  ExtractSignedDistanceField(const std::vector<uint32_t>& objects_to_use,
                             const float oob_value,
                             const bool unknown_is_filled,
                             const bool use_parallel,
                             const bool add_virtual_border) const
  {
    using common_robotics_utilities::voxel_grid::GridIndex;
    // To make this faster, we put the objects to use into a map
    std::map<uint32_t, int32_t> object_use_map;
    for (auto object_to_use : objects_to_use)
    {
      object_use_map[object_to_use] = 1;
    }
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        is_filled_fn = [&] (const GridIndex& index)
    {
      const auto query = GetImmutable(index);
      if (query)
      {
        // If it matches an object to use OR there are no objects supplied
        if ((object_use_map[query.Value().ObjectId()] == 1)
            || (objects_to_use.size() == 0))
        {
          const float occupancy = query.Value().Occupancy();
          if (occupancy > 0.5)
          {
            // Mark as filled
            return true;
          }
          else if (unknown_is_filled && (occupancy == 0.5))
          {
            // Mark as filled
            return true;
          }
        }
        // Mark as free
        return false;
      }
      else
      {
        throw std::runtime_error("index out of grid bounds");
      }
    };
    return signed_distance_field_generation
        ::ExtractSignedDistanceField<
            TaggedObjectCollisionCell,
            std::vector<TaggedObjectCollisionCell>,
            BackingStore>(
                *this, is_filled_fn, oob_value, GetFrame(), use_parallel,
                add_virtual_border);
  }

  template<typename BackingStore=std::vector<float>>
  std::map<uint32_t, SignedDistanceField<BackingStore>> MakeSeparateObjectSDFs(
      const std::vector<uint32_t>& object_ids,
      const float oob_value,
      const bool unknown_is_filled,
      const bool use_parallel,
      const bool add_virtual_border) const
  {
    std::map<uint32_t, SignedDistanceField<BackingStore>> per_object_sdfs;
    for (size_t idx = 0; idx < object_ids.size(); idx++)
    {
      const uint32_t object_id = object_ids[idx];
      per_object_sdfs[object_id]
          = ExtractSignedDistanceField<BackingStore>(
              std::vector<uint32_t>{object_id}, oob_value,
              unknown_is_filled, use_parallel, add_virtual_border)
              .SignedDistanceField();
    }
    return per_object_sdfs;
  }

  template<typename BackingStore=std::vector<float>>
  std::map<uint32_t, SignedDistanceField<BackingStore>> MakeAllObjectSDFs(
      const float oob_value, const bool unknown_is_filled,
      const bool use_parallel, const bool add_virtual_border) const
  {
    std::map<uint32_t, int32_t> object_id_map;
    for (int64_t x_index = 0; x_index < GetNumXCells(); x_index++)
    {
      for (int64_t y_index = 0; y_index < GetNumYCells(); y_index++)
      {
        for (int64_t z_index = 0; z_index < GetNumZCells(); z_index++)
        {
          const TaggedObjectCollisionCell& cell
              = GetImmutable(x_index, y_index, z_index).Value();
          const uint32_t cell_object_id = cell.ObjectId();
          if (cell_object_id > 0)
          {
            object_id_map[cell_object_id] = 1;
          }
        }
      }
    }
    return MakeSeparateObjectSDFs<BackingStore>(
        common_robotics_utilities::utility::GetKeys(object_id_map),
        oob_value, unknown_is_filled, use_parallel, add_virtual_border);
  }

  template<typename BackingStore=std::vector<float>>
  SignedDistanceFieldResult<BackingStore>
  ExtractFreeAndNamedObjectsSignedDistanceField(
      const float oob_value, const bool unknown_is_filled,
      const bool use_parallel) const
  {
    using common_robotics_utilities::voxel_grid::GridIndex;
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        free_sdf_filled_fn = [&] (const GridIndex& index)
    {
      const auto stored = GetImmutable(index).Value();
      if (stored.Occupancy() > 0.5)
      {
        // Mark as filled
        return true;
      }
      else if (unknown_is_filled && (stored.Occupancy() == 0.5))
      {
        // Mark as filled
        return true;
      }
      return false;
    };
    auto free_sdf_result =
        signed_distance_field_generation::ExtractSignedDistanceField
            <TaggedObjectCollisionCell, std::vector<TaggedObjectCollisionCell>,
             BackingStore>(
                *this, free_sdf_filled_fn, oob_value, GetFrame(), use_parallel,
                false);
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        object_filled_fn = [&] (const GridIndex& index)
    {
      const auto stored = GetImmutable(index).Value();
      // If it matches a named object (i.e. object_id >= 1)
      if (stored.ObjectId() > 0u)
      {
        if (stored.Occupancy() > 0.5)
        {
          // Mark as filled
          return true;
        }
        else if (unknown_is_filled && (stored.Occupancy() == 0.5))
        {
          // Mark as filled
          return true;
        }
      }
      return false;
    };
    auto named_objects_sdf_result =
        signed_distance_field_generation::ExtractSignedDistanceField
            <TaggedObjectCollisionCell, std::vector<TaggedObjectCollisionCell>,
             BackingStore>(
                *this, object_filled_fn, oob_value, GetFrame(), use_parallel,
                false);
    SignedDistanceField<BackingStore> combined_sdf
        = free_sdf_result.SignedDistanceField();
    for (int64_t x_idx = 0; x_idx < combined_sdf.GetNumXCells(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.GetNumYCells(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < combined_sdf.GetNumZCells(); z_idx++)
        {
          const float free_sdf_value
              = free_sdf_result.SignedDistanceField().GetImmutable(
                  x_idx, y_idx, z_idx).Value();
          const float named_objects_sdf_value
              = named_objects_sdf_result.SignedDistanceField().GetImmutable(
                  x_idx, y_idx, z_idx).Value();
          if (free_sdf_value >= 0.0)
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, free_sdf_value);
          }
          else if (named_objects_sdf_value <= -0.0)
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, named_objects_sdf_value);
          }
          else
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, 0.0f);
          }
        }
      }
    }
    // Get the combined max/min values
    return SignedDistanceFieldResult<BackingStore>(
        combined_sdf, free_sdf_result.Maximum(),
        named_objects_sdf_result.Minimim());
  }

  SignedDistanceFieldResult<std::vector<float>>
  ExtractSignedDistanceField(const std::vector<uint32_t>& objects_to_use,
                             const float oob_value,
                             const bool unknown_is_filled,
                             const bool use_parallel,
                             const bool add_virtual_border) const;

  std::map<uint32_t, SignedDistanceField<std::vector<float>>>
  MakeSeparateObjectSDFs(const std::vector<uint32_t>& object_ids,
                         const float oob_value,
                         const bool unknown_is_filled,
                         const bool use_parallel,
                         const bool add_virtual_border) const;

  std::map<uint32_t, SignedDistanceField<std::vector<float>>> MakeAllObjectSDFs(
      const float oob_value, const bool unknown_is_filled,
      const bool use_parallel, const bool add_virtual_border) const;

  SignedDistanceFieldResult<std::vector<float>>
  ExtractFreeAndNamedObjectsSignedDistanceField(
      const float oob_value, const bool unknown_is_filled,
      const bool use_parallel) const;
};
}  // namespace voxelized_geometry_tools
