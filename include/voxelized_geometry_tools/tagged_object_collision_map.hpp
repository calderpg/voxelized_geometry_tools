#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/maybe.hpp>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/signed_distance_field_generation.hpp>
#include <voxelized_geometry_tools/topology_computation.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
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

using TaggedObjectCollisionCellSerializer
    = common_robotics_utilities::serialization
        ::Serializer<TaggedObjectCollisionCell>;
using TaggedObjectCollisionCellDeserializer
    = common_robotics_utilities::serialization
        ::Deserializer<TaggedObjectCollisionCell>;

class TaggedObjectCollisionMap
    : public common_robotics_utilities::voxel_grid
        ::VoxelGridBase<TaggedObjectCollisionCell,
                        std::vector<TaggedObjectCollisionCell>>
{
private:
  using DeserializedTaggedObjectCollisionMap
      = common_robotics_utilities::serialization
          ::Deserialized<TaggedObjectCollisionMap>;

  uint32_t number_of_components_ = 0u;
  uint32_t number_of_spatial_segments_ = 0u;
  std::string frame_;
  common_robotics_utilities::utility::CopyableMoveableAtomic<bool>
      components_valid_{false};
  common_robotics_utilities::utility::CopyableMoveableAtomic<bool>
      spatial_segments_valid_{false};

  /// Implement the VoxelGridBase interface.

  /// We need to implement cloning.
  std::unique_ptr<common_robotics_utilities::voxel_grid
      ::VoxelGridBase<TaggedObjectCollisionCell,
                      std::vector<TaggedObjectCollisionCell>>>
  DoClone() const override;

  /// We need to serialize the frame and locked flag.
  uint64_t DerivedSerializeSelf(
      std::vector<uint8_t>& buffer,
      const TaggedObjectCollisionCellSerializer& value_serializer)
      const override;

  /// We need to deserialize the frame and locked flag.
  uint64_t DerivedDeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
      const TaggedObjectCollisionCellDeserializer& value_deserializer)
      override;

  /// Invalidate connected components on mutable access.
  bool OnMutableAccess(const int64_t x_index,
                       const int64_t y_index,
                       const int64_t z_index) override;

public:
  static uint64_t Serialize(
      const TaggedObjectCollisionMap& map, std::vector<uint8_t>& buffer);

  static DeserializedTaggedObjectCollisionMap Deserialize(
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
        frame_(frame)
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
              sizes, default_value, oob_value),
        frame_(frame)
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

  bool AreComponentsValid() const { return components_valid_.load(); }

  /// Use this with great care if you know the components are still/now valid.
  void ForceComponentsToBeValid() { components_valid_.store(true); }

  /// Use this to invalidate the current components.
  void ForceComponentsToBeInvalid() { components_valid_.store(false); }

  bool AreSpatialSegmentsValid() const
  {
    return spatial_segments_valid_.load();
  }

  /// Use this with great care if you know the spatial segments are still/now
  /// valid.
  void ForceSpatialSegmentsToBeValid() { spatial_segments_valid_.store(true); }

  /// Use this to invalidate the current spatial segments.
  void ForceSpatialSegmentsToBeInvalid()
  {
    spatial_segments_valid_.store(false);
  }

  double GetResolution() const { return GetCellSizes().x(); }

  const std::string& GetFrame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }

  uint32_t UpdateConnectedComponents(const bool connect_across_objects);

  common_robotics_utilities::OwningMaybe<uint32_t>
  GetNumConnectedComponents() const
  {
    if (components_valid_.load())
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
      const double connected_threshold,
      const SignedDistanceFieldGenerationParameters<float>& sdf_parameters);

  common_robotics_utilities::OwningMaybe<uint32_t> GetNumSpatialSegments() const
  {
    if (spatial_segments_valid_.load())
    {
      return common_robotics_utilities::OwningMaybe<uint32_t>(
          number_of_spatial_segments_);
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

  topology_computation::TopologicalInvariants ComputeComponentTopology(
      const COMPONENT_TYPES component_types_to_use,
      const bool connect_across_objects,
      const common_robotics_utilities::utility::LoggingFunction&
          logging_fn = {});

  template<typename ScalarType>
  SignedDistanceField<ScalarType> ExtractSignedDistanceField(
      const std::vector<uint32_t>& objects_to_use,
      const SignedDistanceFieldGenerationParameters<ScalarType>& parameters)
      const
  {
    using common_robotics_utilities::voxel_grid::GridIndex;
    // To make this faster, we put the objects to use into a set
    std::set<uint32_t> objects_to_use_set;
    for (auto object_to_use : objects_to_use)
    {
      objects_to_use_set.insert(object_to_use);
    }
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        is_filled_fn = [&] (const GridIndex& index)
    {
      const auto query = GetIndexImmutable(index);
      if (query)
      {
        // If it matches an object to use OR there are no objects supplied
        if ((objects_to_use_set.count(query.Value().ObjectId()) == 1)
            || (objects_to_use.size() == 0))
        {
          const float occupancy = query.Value().Occupancy();
          if (occupancy > 0.5)
          {
            // Mark as filled
            return true;
          }
          else if (parameters.UnknownIsFilled() && (occupancy == 0.5))
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
    return
        signed_distance_field_generation::internal::ExtractSignedDistanceField
            <TaggedObjectCollisionCell, std::vector<TaggedObjectCollisionCell>,
             ScalarType>(*this, is_filled_fn, GetFrame(), parameters);
  }

  template<typename ScalarType>
  std::map<uint32_t, SignedDistanceField<ScalarType>> MakeSeparateObjectSDFs(
      const std::vector<uint32_t>& object_ids,
      const SignedDistanceFieldGenerationParameters<ScalarType>& parameters)
      const
  {
    std::map<uint32_t, SignedDistanceField<ScalarType>> per_object_sdfs;
    for (auto object_id : object_ids)
    {
      per_object_sdfs[object_id] = ExtractSignedDistanceField<ScalarType>(
          std::vector<uint32_t>{object_id}, parameters);
    }
    return per_object_sdfs;
  }

  template<typename ScalarType>
  std::map<uint32_t, SignedDistanceField<ScalarType>> MakeAllObjectSDFs(
      const SignedDistanceFieldGenerationParameters<ScalarType>& parameters)
      const
  {
    std::set<uint32_t> object_ids_set;
    for (int64_t x_index = 0; x_index < GetNumXCells(); x_index++)
    {
      for (int64_t y_index = 0; y_index < GetNumYCells(); y_index++)
      {
        for (int64_t z_index = 0; z_index < GetNumZCells(); z_index++)
        {
          const auto query = GetIndexImmutable(x_index, y_index, z_index);
          const TaggedObjectCollisionCell& cell = query.Value();
          const uint32_t cell_object_id = cell.ObjectId();
          if (cell_object_id > 0)
          {
            object_ids_set.insert(cell_object_id);
          }
        }
      }
    }
    const std::vector<uint32_t> object_ids =
        common_robotics_utilities::utility::GetKeysFromSetLike<uint32_t>(
            object_ids_set);
    return MakeSeparateObjectSDFs<ScalarType>(object_ids, parameters);
  }

  template<typename ScalarType>
  SignedDistanceField<ScalarType> ExtractFreeAndNamedObjectsSignedDistanceField(
      const SignedDistanceFieldGenerationParameters<ScalarType>& parameters)
      const
  {
    using common_robotics_utilities::voxel_grid::GridIndex;
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        free_sdf_filled_fn = [&] (const GridIndex& index)
    {
      const auto stored = GetIndexImmutable(index).Value();
      if (stored.Occupancy() > 0.5)
      {
        // Mark as filled
        return true;
      }
      else if (parameters.UnknownIsFilled() && (stored.Occupancy() == 0.5))
      {
        // Mark as filled
        return true;
      }
      return false;
    };
    const auto free_sdf =
        signed_distance_field_generation::internal::ExtractSignedDistanceField
            <TaggedObjectCollisionCell, std::vector<TaggedObjectCollisionCell>,
             ScalarType>(*this, free_sdf_filled_fn, GetFrame(), parameters);

    // Make the helper function
    const std::function<bool(const GridIndex&)>
        object_filled_fn = [&] (const GridIndex& index)
    {
      const auto stored = GetIndexImmutable(index).Value();
      // If it matches a named object (i.e. object_id >= 1)
      if (stored.ObjectId() > 0u)
      {
        if (stored.Occupancy() > 0.5)
        {
          // Mark as filled
          return true;
        }
        else if (parameters.UnknownIsFilled() && (stored.Occupancy() == 0.5))
        {
          // Mark as filled
          return true;
        }
      }
      return false;
    };
    const auto named_objects_sdf =
        signed_distance_field_generation::internal::ExtractSignedDistanceField
            <TaggedObjectCollisionCell, std::vector<TaggedObjectCollisionCell>,
             ScalarType>(*this, object_filled_fn, GetFrame(), parameters);

    SignedDistanceField<ScalarType> combined_sdf = free_sdf;
    combined_sdf.Unlock();
    for (int64_t x_idx = 0; x_idx < combined_sdf.GetNumXCells(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.GetNumYCells(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < combined_sdf.GetNumZCells(); z_idx++)
        {
          const ScalarType free_sdf_value =
              free_sdf.GetIndexImmutable(x_idx, y_idx, z_idx).Value();
          const ScalarType named_objects_sdf_value =
              named_objects_sdf.GetIndexImmutable(x_idx, y_idx, z_idx).Value();
          if (free_sdf_value >= 0.0)
          {
            combined_sdf.SetIndex(x_idx, y_idx, z_idx, free_sdf_value);
          }
          else if (named_objects_sdf_value <= -0.0)
          {
            combined_sdf.SetIndex(x_idx, y_idx, z_idx, named_objects_sdf_value);
          }
          else
          {
            combined_sdf.SetIndex(
                x_idx, y_idx, z_idx, static_cast<ScalarType>(0.0));
          }
        }
      }
    }

    // Lock & update min/max values.
    combined_sdf.Lock();
    return combined_sdf;
  }

  SignedDistanceField<double> ExtractSignedDistanceFieldDouble(
      const std::vector<uint32_t>& objects_to_use,
      const SignedDistanceFieldGenerationParameters<double>& parameters) const;

  SignedDistanceField<float> ExtractSignedDistanceFieldFloat(
      const std::vector<uint32_t>& objects_to_use,
      const SignedDistanceFieldGenerationParameters<float>& parameters) const;

  std::map<uint32_t, SignedDistanceField<double>>
  MakeSeparateObjectSDFsDouble(
      const std::vector<uint32_t>& object_ids,
      const SignedDistanceFieldGenerationParameters<double>& parameters) const;

  std::map<uint32_t, SignedDistanceField<float>>
  MakeSeparateObjectSDFsFloat(
      const std::vector<uint32_t>& object_ids,
      const SignedDistanceFieldGenerationParameters<float>& parameters) const;

  std::map<uint32_t, SignedDistanceField<double>> MakeAllObjectSDFsDouble(
      const SignedDistanceFieldGenerationParameters<double>& parameters) const;

  std::map<uint32_t, SignedDistanceField<float>> MakeAllObjectSDFsFloat(
      const SignedDistanceFieldGenerationParameters<float>& parameters) const;

  SignedDistanceField<double>
  ExtractFreeAndNamedObjectsSignedDistanceFieldDouble(
      const SignedDistanceFieldGenerationParameters<double>& parameters) const;

  SignedDistanceField<float>
  ExtractFreeAndNamedObjectsSignedDistanceFieldFloat(
      const SignedDistanceFieldGenerationParameters<float>& parameters) const;
};
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
