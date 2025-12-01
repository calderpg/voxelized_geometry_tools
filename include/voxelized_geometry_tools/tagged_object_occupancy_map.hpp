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
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
class TaggedObjectOccupancyCell
{
private:
  using DeserializedTaggedObjectOccupancyCell
      = common_robotics_utilities::serialization
          ::Deserialized<TaggedObjectOccupancyCell>;

  common_robotics_utilities::utility
      ::CopyableMoveableAtomic<float, std::memory_order_relaxed>
          occupancy_{0.0f};
  common_robotics_utilities::utility
      ::CopyableMoveableAtomic<uint32_t, std::memory_order_relaxed>
          object_id_{0u};

public:
  static uint64_t Serialize(
      const TaggedObjectOccupancyCell& cell, std::vector<uint8_t>& buffer);

  static DeserializedTaggedObjectOccupancyCell Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  TaggedObjectOccupancyCell() : occupancy_(0.0f), object_id_(0u) {}

  explicit TaggedObjectOccupancyCell(const float occupancy)
      : occupancy_(occupancy), object_id_(0u) {}

  TaggedObjectOccupancyCell(const float occupancy, const uint32_t object_id)
      : occupancy_(occupancy), object_id_(object_id) {}

  float Occupancy() const { return occupancy_.load(); }

  uint32_t ObjectId() const { return object_id_.load(); }

  void SetOccupancy(const float occupancy) { occupancy_.store(occupancy); }

  void SetObjectId(const uint32_t object_id) { object_id_.store(object_id); }
};

// Enforce that despite its members being std::atomics,
// TaggedObjectOccupancyCell has the expected size.
static_assert(
    sizeof(TaggedObjectOccupancyCell) == (sizeof(float) * 2),
    "TaggedObjectOccupancyCell is larger than expected.");

using TaggedObjectOccupancyCellSerializer
    = common_robotics_utilities::serialization
        ::Serializer<TaggedObjectOccupancyCell>;
using TaggedObjectOccupancyCellDeserializer
    = common_robotics_utilities::serialization
        ::Deserializer<TaggedObjectOccupancyCell>;

class TaggedObjectOccupancyMap
    : public common_robotics_utilities::voxel_grid
        ::VoxelGridBase<TaggedObjectOccupancyCell,
                        std::vector<TaggedObjectOccupancyCell>>
{
private:
  using DeserializedTaggedObjectOccupancyMap
      = common_robotics_utilities::serialization
          ::Deserialized<TaggedObjectOccupancyMap>;

  std::string frame_;

  /// Implement the VoxelGridBase interface.

  /// We need to implement cloning.
  std::unique_ptr<common_robotics_utilities::voxel_grid
      ::VoxelGridBase<TaggedObjectOccupancyCell,
                      std::vector<TaggedObjectOccupancyCell>>>
  DoClone() const override;

  /// We need to serialize the frame and locked flag.
  uint64_t DerivedSerializeSelf(
      std::vector<uint8_t>& buffer,
      const TaggedObjectOccupancyCellSerializer& value_serializer)
      const override;

  /// We need to deserialize the frame and locked flag.
  uint64_t DerivedDeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
      const TaggedObjectOccupancyCellDeserializer& value_deserializer)
      override;

  bool OnMutableAccess(const int64_t x_index,
                       const int64_t y_index,
                       const int64_t z_index) override;

  bool OnMutableRawAccess() override;

  void EnforceUniformVoxelSize() const
  {
    if (!HasUniformVoxelSize())
    {
      throw std::invalid_argument(
          "Tagged object occupancy map cannot have non-uniform voxel sizes");
    }
  }

public:
  static uint64_t Serialize(
      const TaggedObjectOccupancyMap& map, std::vector<uint8_t>& buffer);

  static DeserializedTaggedObjectOccupancyMap Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  static void SaveToFile(const TaggedObjectOccupancyMap& map,
                         const std::string& filepath,
                         const bool compress);

  static TaggedObjectOccupancyMap LoadFromFile(const std::string& filepath);

  TaggedObjectOccupancyMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const TaggedObjectOccupancyCell& default_value)
      : TaggedObjectOccupancyMap(
          origin_transform, frame, sizes, default_value, default_value) {}

  TaggedObjectOccupancyMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const TaggedObjectOccupancyCell& default_value)
      : TaggedObjectOccupancyMap(frame, sizes, default_value, default_value) {}

  TaggedObjectOccupancyMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const TaggedObjectOccupancyCell& default_value,
      const TaggedObjectOccupancyCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<TaggedObjectOccupancyCell,
                          std::vector<TaggedObjectOccupancyCell>>(
              origin_transform, sizes, default_value, oob_value),
        frame_(frame)
  {
    EnforceUniformVoxelSize();
  }

  TaggedObjectOccupancyMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const TaggedObjectOccupancyCell& default_value,
      const TaggedObjectOccupancyCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<TaggedObjectOccupancyCell,
                          std::vector<TaggedObjectOccupancyCell>>(
              sizes, default_value, oob_value),
        frame_(frame)
  {
    EnforceUniformVoxelSize();
  }

  TaggedObjectOccupancyMap()
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<TaggedObjectOccupancyCell,
                          std::vector<TaggedObjectOccupancyCell>>() {}

  double Resolution() const { return VoxelXSize(); }

  const std::string& Frame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }

  common_robotics_utilities::OwningMaybe<bool> IsSurfaceIndex(
      const common_robotics_utilities::voxel_grid::GridIndex& index) const;

  common_robotics_utilities::OwningMaybe<bool> IsSurfaceIndex(
      const int64_t x_index, const int64_t y_index,
      const int64_t z_index) const;

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
            <TaggedObjectOccupancyCell, std::vector<TaggedObjectOccupancyCell>,
             ScalarType>(*this, is_filled_fn, Frame(), parameters);
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
    for (int64_t x_index = 0; x_index < NumXVoxels(); x_index++)
    {
      for (int64_t y_index = 0; y_index < NumYVoxels(); y_index++)
      {
        for (int64_t z_index = 0; z_index < NumZVoxels(); z_index++)
        {
          const auto query = GetIndexImmutable(x_index, y_index, z_index);
          const TaggedObjectOccupancyCell& cell = query.Value();
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
            <TaggedObjectOccupancyCell, std::vector<TaggedObjectOccupancyCell>,
             ScalarType>(*this, free_sdf_filled_fn, Frame(), parameters);

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
            <TaggedObjectOccupancyCell, std::vector<TaggedObjectOccupancyCell>,
             ScalarType>(*this, object_filled_fn, Frame(), parameters);

    SignedDistanceField<ScalarType> combined_sdf = free_sdf;
    combined_sdf.Unlock();
    for (int64_t x_idx = 0; x_idx < combined_sdf.NumXVoxels(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.NumYVoxels(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < combined_sdf.NumZVoxels(); z_idx++)
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
