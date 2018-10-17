#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>

namespace voxelized_geometry_tools
{
class DynamicSpatialHashedCollisionMap final
    : public common_robotics_utilities::voxel_grid
        ::DynamicSpatialHashedVoxelGridBase<
            CollisionCell, std::vector<CollisionCell>>
{
private:
  std::string frame_;

  /// Implement the DynamicSpatialHashedVoxelGridBase interface.

  /// We need to implement cloning.
  common_robotics_utilities::voxel_grid
      ::DynamicSpatialHashedVoxelGridBase<
          CollisionCell, std::vector<CollisionCell>>*
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

  bool OnMutableAccess(const Eigen::Vector4d& location) override;

public:
  static uint64_t Serialize(
      const DynamicSpatialHashedCollisionMap& grid,
      std::vector<uint8_t>& buffer);

  static std::pair<DynamicSpatialHashedCollisionMap, uint64_t>
  Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  static void SaveToFile(const DynamicSpatialHashedCollisionMap& map,
                         const std::string& filepath,
                         const bool compress);

  static DynamicSpatialHashedCollisionMap LoadFromFile(
      const std::string& filepath);

  DynamicSpatialHashedCollisionMap(
      const common_robotics_utilities::voxel_grid::GridSizes& chunk_sizes,
      const CollisionCell& default_value, const std::string& frame)
      : DynamicSpatialHashedVoxelGridBase<
          CollisionCell, std::vector<CollisionCell>>(
              Eigen::Isometry3d::Identity(), chunk_sizes, default_value),
        frame_(frame)
  {
    if (!HasUniformCellSize())
    {
      throw std::invalid_argument(
          "DSH collision map cannot have non-uniform cell sizes");
    }
  }

  DynamicSpatialHashedCollisionMap(
      const Eigen::Isometry3d& origin_transform,
      const common_robotics_utilities::voxel_grid::GridSizes& chunk_sizes,
      const CollisionCell& default_value, const std::string& frame)
      : DynamicSpatialHashedVoxelGridBase<
          CollisionCell, std::vector<CollisionCell>>(
              origin_transform, chunk_sizes, default_value), frame_(frame)
  {
    if (!HasUniformCellSize())
    {
      throw std::invalid_argument(
          "DSH collision map cannot have non-uniform cell sizes");
    }
  }

  DynamicSpatialHashedCollisionMap()
      : DynamicSpatialHashedVoxelGridBase<
          CollisionCell, std::vector<CollisionCell>>() {}

  double GetResolution() const { return GetCellSizes().x(); }

  const std::string& GetFrame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }
};
}  // namespace voxelized_geometry_tools
