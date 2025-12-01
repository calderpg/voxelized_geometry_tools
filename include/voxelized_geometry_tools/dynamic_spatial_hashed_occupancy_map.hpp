#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
class DynamicSpatialHashedOccupancyMap final
    : public common_robotics_utilities::voxel_grid
        ::DynamicSpatialHashedVoxelGridBase<
            OccupancyCell, std::vector<OccupancyCell>>
{
private:
  using DeserializedDynamicSpatialHashedOccupancyMap
      = common_robotics_utilities::serialization
          ::Deserialized<DynamicSpatialHashedOccupancyMap>;

  std::string frame_;

  /// Implement the DynamicSpatialHashedVoxelGridBase interface.

  /// We need to implement cloning.
  std::unique_ptr<common_robotics_utilities::voxel_grid
      ::DynamicSpatialHashedVoxelGridBase<
          OccupancyCell, std::vector<OccupancyCell>>>
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
          "DSH occupancy map cannot have non-uniform voxel sizes");
    }
  }

public:
  static uint64_t Serialize(
      const DynamicSpatialHashedOccupancyMap& grid,
      std::vector<uint8_t>& buffer);

  static DeserializedDynamicSpatialHashedOccupancyMap Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset);

  static void SaveToFile(const DynamicSpatialHashedOccupancyMap& map,
                         const std::string& filepath,
                         const bool compress);

  static DynamicSpatialHashedOccupancyMap LoadFromFile(
      const std::string& filepath);

  DynamicSpatialHashedOccupancyMap(
      const common_robotics_utilities::voxel_grid
          ::DynamicSpatialHashedVoxelGridSizes& voxel_grid_sizes,
      const OccupancyCell& default_value, const size_t expected_chunks,
      const std::string& frame)
      : DynamicSpatialHashedVoxelGridBase<
          OccupancyCell, std::vector<OccupancyCell>>(
              voxel_grid_sizes, default_value, expected_chunks),
        frame_(frame)
  {
    EnforceUniformVoxelSize();
  }

  DynamicSpatialHashedOccupancyMap(
      const Eigen::Isometry3d& origin_transform,
      const common_robotics_utilities::voxel_grid
          ::DynamicSpatialHashedVoxelGridSizes& voxel_grid_sizes,
      const OccupancyCell& default_value, const size_t expected_chunks,
      const std::string& frame)
      : DynamicSpatialHashedVoxelGridBase<
          OccupancyCell, std::vector<OccupancyCell>>(
              origin_transform, voxel_grid_sizes, default_value,
              expected_chunks),
        frame_(frame)
  {
    EnforceUniformVoxelSize();
  }

  DynamicSpatialHashedOccupancyMap()
      : DynamicSpatialHashedVoxelGridBase<
          OccupancyCell, std::vector<OccupancyCell>>() {}

  double Resolution() const { return VoxelXSize(); }

  const std::string& Frame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }
};
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
