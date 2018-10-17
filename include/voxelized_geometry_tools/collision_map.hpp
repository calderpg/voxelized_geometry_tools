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
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/signed_distance_field_generation.hpp>

namespace voxelized_geometry_tools
{
class CollisionCell
{
private:
  float occupancy_ = 0.0;
  uint32_t component_ = 0u;

public:
  CollisionCell() : occupancy_(0.0), component_(0u) {}

  CollisionCell(const float occupancy)
      : occupancy_(occupancy), component_(0u) {}

  CollisionCell(const float occupancy, const uint32_t component)
      : occupancy_(occupancy), component_(component) {}

  const float& Occupancy() const { return occupancy_; }

  float& Occupancy() { return occupancy_; }

  const uint32_t& Component() const { return component_; }

  uint32_t& Component() { return component_; }
};

class CollisionMap
    : public common_robotics_utilities::voxel_grid
        ::VoxelGridBase<CollisionCell, std::vector<CollisionCell>>
{
private:
  uint32_t number_of_components_ = 0u;
  std::string frame_;
  bool components_valid_ = false;

  /// Implement the VoxelGridBase interface.

  /// We need to implement cloning.
  common_robotics_utilities::voxel_grid
      ::VoxelGridBase<CollisionCell, std::vector<CollisionCell>>*
  DoClone() const override;

  /// We need to serialize the frame and locked flag.
  uint64_t DerivedSerializeSelf(
      std::vector<uint8_t>& buffer,
      const std::function<uint64_t(
          const CollisionCell&,
          std::vector<uint8_t>&)>& value_serializer) const override;

  /// We need to deserialize the frame and locked flag.
  uint64_t DerivedDeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
      const std::function<std::pair<CollisionCell, uint64_t>(
          const std::vector<uint8_t>&,
          const uint64_t)>& value_deserializer) override;

  /// Invalidate connected components on mutable access.
  bool OnMutableAccess(const int64_t x_index,
                       const int64_t y_index,
                       const int64_t z_index) override;

public:
  static uint64_t Serialize(
      const CollisionMap& map, std::vector<uint8_t>& buffer);

  static std::pair<CollisionMap, uint64_t> Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  static void SaveToFile(const CollisionMap& map,
                         const std::string& filepath,
                         const bool compress);

  static CollisionMap LoadFromFile(const std::string& filepath);

  CollisionMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const CollisionCell& default_value)
      : CollisionMap(
          origin_transform, frame, sizes, default_value, default_value) {}

  CollisionMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const CollisionCell& default_value)
      : CollisionMap(frame, sizes, default_value, default_value) {}

  CollisionMap(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const CollisionCell& default_value, const CollisionCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<CollisionCell, std::vector<CollisionCell>>(
              origin_transform, sizes, default_value, oob_value),
        number_of_components_(0u), frame_(frame), components_valid_(false)
  {
    if (!HasUniformCellSize())
    {
      throw std::invalid_argument(
          "Collision map cannot have non-uniform cell sizes");
    }
  }

  CollisionMap(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const CollisionCell& default_value, const CollisionCell& oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<CollisionCell, std::vector<CollisionCell>>(
              sizes, default_value, oob_value), number_of_components_(0u),
        frame_(frame), components_valid_(false)
  {
    if (!HasUniformCellSize())
    {
      throw std::invalid_argument(
          "Collision map cannot have non-uniform cell sizes");
    }
  }

  CollisionMap()
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<CollisionCell, std::vector<CollisionCell>>() {}

  bool AreComponentsValid() const { return components_valid_; }

  /// Use this with great care if you know the components are still/now valid.
  void ForceComponentsToBeValid() { components_valid_ = true; }

  /// Use this to invalidate the current components.
  void ForceComponentsToBeInvalid() { components_valid_ = false; }

  double GetResolution() const { return GetCellSizes().x(); }

  const std::string& GetFrame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }

  uint32_t UpdateConnectedComponents();

  common_robotics_utilities::OwningMaybe<uint32_t> GetNumConnectedComponents() const
  {
    if (components_valid_)
    {
      return common_robotics_utilities::OwningMaybe<uint32_t>(number_of_components_);
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

  std::map<uint32_t, std::pair<int32_t, int32_t>>
  ComputeComponentTopology(
      const COMPONENT_TYPES component_types_to_use, const bool verbose);

  template<typename BackingStore=std::vector<float>>
  std::pair<SignedDistanceField<BackingStore>, std::pair<double, double>>
  ExtractSignedDistanceField(const float oob_value,
                             const bool unknown_is_filled,
                             const bool use_parallel,
                             const bool add_virtual_border) const
  {
    using common_robotics_utilities::voxel_grid::GridIndex;
    // Make the helper function
    const std::function<bool(const GridIndex&)>
        is_filled_fn = [&] (const GridIndex& index)
    {
      const auto query = GetImmutable(index);
      if (query)
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
        // Mark as free
        return false;
      }
      else
      {
        throw std::runtime_error("index out of grid bounds");
      }
    };
    return signed_distance_field_generation::ExtractSignedDistanceField
        <CollisionCell, std::vector<CollisionCell>, BackingStore>(
            *this, is_filled_fn, oob_value, GetFrame(), use_parallel,
            add_virtual_border);
  }

  std::pair<SignedDistanceField<std::vector<float>>, std::pair<double, double>>
  ExtractSignedDistanceField(const float oob_value,
                             const bool unknown_is_filled,
                             const bool use_parallel,
                             const bool add_virtual_border) const;
};
}  // namespace voxelized_geometry_tools
