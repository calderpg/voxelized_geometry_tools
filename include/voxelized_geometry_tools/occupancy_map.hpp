#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
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
class OccupancyCell
{
private:
  using DeserializedOccupancyCell
      = common_robotics_utilities::serialization::Deserialized<OccupancyCell>;

  common_robotics_utilities::utility
      ::CopyableMoveableAtomic<float, std::memory_order_relaxed>
          occupancy_{0.0f};

public:
  static uint64_t Serialize(
      const OccupancyCell& cell, std::vector<uint8_t>& buffer);

  static DeserializedOccupancyCell Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  OccupancyCell() : occupancy_(0.0f) {}

  explicit OccupancyCell(const float occupancy) : occupancy_(occupancy) {}

  float Occupancy() const { return occupancy_.load(); }

  void SetOccupancy(const float occupancy) { occupancy_.store(occupancy); }
};

// Enforce that despite its members being std::atomics, OccupancyCell has the
// expected size.
static_assert(
    sizeof(OccupancyCell) == sizeof(float),
    "OccupancyCell is larger than expected.");

using OccupancyCellSerializer
    = common_robotics_utilities::serialization::Serializer<OccupancyCell>;
using OccupancyCellDeserializer
    = common_robotics_utilities::serialization::Deserializer<OccupancyCell>;

class OccupancyMap
    : public common_robotics_utilities::voxel_grid
        ::VoxelGridBase<OccupancyCell, std::vector<OccupancyCell>>
{
private:
  using DeserializedOccupancyMap
      = common_robotics_utilities::serialization::Deserialized<OccupancyMap>;

  std::string frame_;

  /// Implement the VoxelGridBase interface.

  /// We need to implement cloning.
  std::unique_ptr<common_robotics_utilities::voxel_grid
      ::VoxelGridBase<OccupancyCell, std::vector<OccupancyCell>>>
  DoClone() const override;

  /// We need to serialize the frame and locked flag.
  uint64_t DerivedSerializeSelf(
      std::vector<uint8_t>& buffer,
      const OccupancyCellSerializer& value_serializer) const override;

  /// We need to deserialize the frame and locked flag.
  uint64_t DerivedDeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
      const OccupancyCellDeserializer& value_deserializer) override;

  bool OnMutableAccess(const int64_t x_index,
                       const int64_t y_index,
                       const int64_t z_index) override;

  bool OnMutableRawAccess() override;

  void EnforceUniformVoxelSize() const
  {
    if (!HasUniformVoxelSize())
    {
      throw std::invalid_argument(
          "Occupancy map cannot have non-uniform voxel sizes");
    }
  }

public:
  static uint64_t Serialize(
      const OccupancyMap& map, std::vector<uint8_t>& buffer);

  static DeserializedOccupancyMap Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  static void SaveToFile(const OccupancyMap& map,
                         const std::string& filepath,
                         const bool compress);

  static OccupancyMap LoadFromFile(const std::string& filepath);

  OccupancyMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const OccupancyCell& default_value)
      : OccupancyMap(
          origin_transform, frame, sizes, default_value, default_value) {}

  OccupancyMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const OccupancyCell& default_value)
      : OccupancyMap(frame, sizes, default_value, default_value) {}

  OccupancyMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const OccupancyCell& default_value, const OccupancyCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<OccupancyCell, std::vector<OccupancyCell>>(
              origin_transform, sizes, default_value, oob_value),
        frame_(frame)
  {
    EnforceUniformVoxelSize();
  }

  OccupancyMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::VoxelGridSizes& sizes,
      const OccupancyCell& default_value, const OccupancyCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<OccupancyCell, std::vector<OccupancyCell>>(
              sizes, default_value, oob_value),
        frame_(frame)
  {
    EnforceUniformVoxelSize();
  }

  OccupancyMap()
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<OccupancyCell, std::vector<OccupancyCell>>() {}

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
      const SignedDistanceFieldGenerationParameters<ScalarType>& parameters)
      const
  {
    using common_robotics_utilities::voxel_grid::GridIndex;
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        is_filled_fn = [&] (const GridIndex& index)
    {
      const auto query = GetIndexImmutable(index);
      if (query)
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
            <OccupancyCell, std::vector<OccupancyCell>, ScalarType>(
                *this, is_filled_fn, Frame(), parameters);
  }

  SignedDistanceField<double> ExtractSignedDistanceFieldDouble(
      const SignedDistanceFieldGenerationParameters<double>& parameters) const;

  SignedDistanceField<float> ExtractSignedDistanceFieldFloat(
      const SignedDistanceFieldGenerationParameters<float>& parameters) const;
};
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
