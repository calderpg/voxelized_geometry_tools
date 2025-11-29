#include <voxelized_geometry_tools/occupancy_map_conversions.hpp>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/occupancy_component_map.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_component_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_map.hpp>

using common_robotics_utilities::voxel_grid::Vector3i64;
using common_robotics_utilities::voxel_grid::VoxelGridSizes;

namespace voxelized_geometry_tools
{
namespace
{

bool TransformsExactlyEqual(
    const Eigen::Isometry3d& t1, const Eigen::Isometry3d& t2)
{
  return t1.matrix().cwiseEqual(t2.matrix()).all();
}

bool CheckCellsEquivalent(
    const OccupancyCell& from_cell, const OccupancyCell& to_cell)
{
  return from_cell.Occupancy() == to_cell.Occupancy();
}

bool CheckCellsEquivalent(
    const OccupancyCell& from_cell, const OccupancyComponentCell& to_cell)
{
  return from_cell.Occupancy() == to_cell.Occupancy();
}

bool CheckCellsEquivalent(
    const OccupancyComponentCell& from_cell,
    const OccupancyComponentCell& to_cell)
{
  return (from_cell.Occupancy() == to_cell.Occupancy()) &&
         (from_cell.Component() == to_cell.Component());
}

bool CheckCellsEquivalent(
    const TaggedObjectOccupancyCell& from_cell,
    const TaggedObjectOccupancyCell& to_cell)
{
  return (from_cell.Occupancy() == to_cell.Occupancy()) &&
         (from_cell.ObjectId() == to_cell.ObjectId());
}

bool CheckCellsEquivalent(
    const TaggedObjectOccupancyCell& from_cell,
    const TaggedObjectOccupancyComponentCell& to_cell)
{
  return (from_cell.Occupancy() == to_cell.Occupancy()) &&
         (from_cell.ObjectId() == to_cell.ObjectId());
}

bool CheckCellsEquivalent(
    const TaggedObjectOccupancyComponentCell& from_cell,
    const TaggedObjectOccupancyComponentCell& to_cell)
{
  return (from_cell.Occupancy() == to_cell.Occupancy()) &&
         (from_cell.ObjectId() == to_cell.ObjectId()) &&
         (from_cell.Component() == to_cell.Component()) &&
         (from_cell.SpatialSegment() == to_cell.SpatialSegment());
}

template<typename FromMapType, typename ToMapType>
void CheckOccupancyMapsEquivalent(
    const FromMapType& from_map, const ToMapType& to_map)
{
  const bool from_map_initialized = from_map.IsInitialized();
  const bool to_map_initialized = to_map.IsInitialized();

  ASSERT_EQ(from_map_initialized, to_map_initialized);

  if (from_map_initialized)
  {
    const auto& from_map_origin_transform = from_map.OriginTransform();
    const auto& to_map_origin_transform = to_map.OriginTransform();

    EXPECT_TRUE(TransformsExactlyEqual(
        from_map_origin_transform, to_map_origin_transform));

    const auto& from_map_grid_sizes = from_map.ControlSizes();
    const auto& to_map_grid_sizes = to_map.ControlSizes();

    EXPECT_EQ(from_map_grid_sizes, to_map_grid_sizes);

    const auto& from_map_frame = from_map.Frame();
    const auto& to_map_frame = to_map.Frame();

    EXPECT_EQ(from_map_frame, to_map_frame);

    const auto& from_map_default_cell = from_map.DefaultValue();
    const auto& to_map_default_cell = to_map.DefaultValue();

    EXPECT_TRUE(
        CheckCellsEquivalent(from_map_default_cell, to_map_default_cell));

    const auto& from_map_oob_cell = from_map.OOBValue();
    const auto& to_map_oob_cell = to_map.OOBValue();

    EXPECT_TRUE(CheckCellsEquivalent(from_map_oob_cell, to_map_oob_cell));

    const auto& from_map_backing_store = from_map.GetImmutableRawData();
    const auto& to_map_backing_store = to_map.GetImmutableRawData();

    const size_t num_from_map_cells = from_map_backing_store.size();
    const size_t num_to_map_cells = to_map_backing_store.size();

    ASSERT_EQ(num_from_map_cells, num_to_map_cells);

    for (size_t index = 0; index < num_from_map_cells; index++)
    {
      const auto& from_map_cell = from_map_backing_store.at(index);
      const auto& to_map_cell = to_map_backing_store.at(index);

      EXPECT_TRUE(CheckCellsEquivalent(from_map_cell, to_map_cell));
    }
  }
}

GTEST_TEST(OccupancyMapConversionsTest, DefaultOccupancyCellConversions)
{
  {
    const OccupancyCell default_occupancy_cell;
    const OccupancyComponentCell default_occupancy_component_cell;

    const OccupancyComponentCell converted_default_occupancy_component_cell =
        ConvertToOccupancyComponentCell(default_occupancy_cell);
    const OccupancyCell converted_default_occupancy_cell =
        ConvertFromOccupancyComponentCell(default_occupancy_component_cell);

    EXPECT_EQ(default_occupancy_cell.Occupancy(), 0.0f);

    EXPECT_EQ(default_occupancy_component_cell.Occupancy(), 0.0f);
    EXPECT_EQ(default_occupancy_component_cell.Component(), 0u);

    EXPECT_EQ(converted_default_occupancy_cell.Occupancy(), 0.0f);

    EXPECT_EQ(converted_default_occupancy_component_cell.Occupancy(), 0.0f);
    EXPECT_EQ(converted_default_occupancy_component_cell.Component(), 0u);
  }

  {
    const TaggedObjectOccupancyCell default_occupancy_cell;
    const TaggedObjectOccupancyComponentCell default_occupancy_component_cell;

    const TaggedObjectOccupancyComponentCell
        converted_default_occupancy_component_cell =
            ConvertToTaggedObjectOccupancyComponentCell(default_occupancy_cell);
    const TaggedObjectOccupancyCell converted_default_occupancy_cell =
        ConvertFromTaggedObjectOccupancyComponentCell(
            default_occupancy_component_cell);

    EXPECT_EQ(default_occupancy_cell.Occupancy(), 0.0f);
    EXPECT_EQ(default_occupancy_cell.ObjectId(), 0u);

    EXPECT_EQ(default_occupancy_component_cell.Occupancy(), 0.0f);
    EXPECT_EQ(default_occupancy_component_cell.ObjectId(), 0u);
    EXPECT_EQ(default_occupancy_component_cell.Component(), 0u);
    EXPECT_EQ(default_occupancy_component_cell.SpatialSegment(), 0u);

    EXPECT_EQ(converted_default_occupancy_cell.Occupancy(), 0.0f);
    EXPECT_EQ(converted_default_occupancy_cell.ObjectId(), 0u);

    EXPECT_EQ(converted_default_occupancy_component_cell.Occupancy(), 0.0f);
    EXPECT_EQ(converted_default_occupancy_component_cell.ObjectId(), 0u);
    EXPECT_EQ(converted_default_occupancy_component_cell.Component(), 0u);
    EXPECT_EQ(converted_default_occupancy_component_cell.SpatialSegment(), 0u);
  }
}

GTEST_TEST(OccupancyMapConversionsTest, EmptyOccupancyCellConversions)
{
  {
    const OccupancyCell empty_occupancy_cell(0.25f);
    const OccupancyComponentCell empty_occupancy_component_cell(0.25f);

    const OccupancyComponentCell converted_empty_occupancy_component_cell =
        ConvertToOccupancyComponentCell(empty_occupancy_cell);
    const OccupancyCell converted_empty_occupancy_cell =
        ConvertFromOccupancyComponentCell(empty_occupancy_component_cell);

    EXPECT_EQ(empty_occupancy_cell.Occupancy(), 0.25f);

    EXPECT_EQ(empty_occupancy_component_cell.Occupancy(), 0.25f);
    EXPECT_EQ(empty_occupancy_component_cell.Component(), 0u);

    EXPECT_EQ(converted_empty_occupancy_cell.Occupancy(), 0.25f);

    EXPECT_EQ(converted_empty_occupancy_component_cell.Occupancy(), 0.25f);
    EXPECT_EQ(converted_empty_occupancy_component_cell.Component(), 0u);
  }

  {
    const TaggedObjectOccupancyCell empty_occupancy_cell(0.25);
    const TaggedObjectOccupancyComponentCell
        empty_occupancy_component_cell(0.25);

    const TaggedObjectOccupancyComponentCell
        converted_empty_occupancy_component_cell =
            ConvertToTaggedObjectOccupancyComponentCell(empty_occupancy_cell);
    const TaggedObjectOccupancyCell converted_empty_occupancy_cell =
        ConvertFromTaggedObjectOccupancyComponentCell(
            empty_occupancy_component_cell);

    EXPECT_EQ(empty_occupancy_cell.Occupancy(), 0.25f);
    EXPECT_EQ(empty_occupancy_cell.ObjectId(), 0u);

    EXPECT_EQ(empty_occupancy_component_cell.Occupancy(), 0.25f);
    EXPECT_EQ(empty_occupancy_component_cell.ObjectId(), 0u);
    EXPECT_EQ(empty_occupancy_component_cell.Component(), 0u);
    EXPECT_EQ(empty_occupancy_component_cell.SpatialSegment(), 0u);

    EXPECT_EQ(converted_empty_occupancy_cell.Occupancy(), 0.25f);
    EXPECT_EQ(converted_empty_occupancy_cell.ObjectId(), 0u);

    EXPECT_EQ(converted_empty_occupancy_component_cell.Occupancy(), 0.25f);
    EXPECT_EQ(converted_empty_occupancy_component_cell.ObjectId(), 0u);
    EXPECT_EQ(converted_empty_occupancy_component_cell.Component(), 0u);
    EXPECT_EQ(converted_empty_occupancy_component_cell.SpatialSegment(), 0u);
  }
}

GTEST_TEST(OccupancyMapConversionsTest, FilledOccupancyCellConversions)
{
  {
    const OccupancyCell filled_occupancy_cell(0.75f);
    const OccupancyComponentCell filled_occupancy_component_cell(0.75f);

    const OccupancyComponentCell converted_filled_occupancy_component_cell =
        ConvertToOccupancyComponentCell(filled_occupancy_cell);
    const OccupancyCell converted_filled_occupancy_cell =
        ConvertFromOccupancyComponentCell(filled_occupancy_component_cell);

    EXPECT_EQ(filled_occupancy_cell.Occupancy(), 0.75f);

    EXPECT_EQ(filled_occupancy_component_cell.Occupancy(), 0.75f);
    EXPECT_EQ(filled_occupancy_component_cell.Component(), 0u);

    EXPECT_EQ(converted_filled_occupancy_cell.Occupancy(), 0.75f);

    EXPECT_EQ(converted_filled_occupancy_component_cell.Occupancy(), 0.75f);
    EXPECT_EQ(converted_filled_occupancy_component_cell.Component(), 0u);
  }

  {
    const TaggedObjectOccupancyCell filled_occupancy_cell(0.75, 75u);
    const TaggedObjectOccupancyComponentCell
        filled_occupancy_component_cell(0.75, 75u);

    const TaggedObjectOccupancyComponentCell
        converted_filled_occupancy_component_cell =
            ConvertToTaggedObjectOccupancyComponentCell(filled_occupancy_cell);
    const TaggedObjectOccupancyCell converted_filled_occupancy_cell =
        ConvertFromTaggedObjectOccupancyComponentCell(
            filled_occupancy_component_cell);

    EXPECT_EQ(filled_occupancy_cell.Occupancy(), 0.75f);
    EXPECT_EQ(filled_occupancy_cell.ObjectId(), 75u);

    EXPECT_EQ(filled_occupancy_component_cell.Occupancy(), 0.75f);
    EXPECT_EQ(filled_occupancy_component_cell.ObjectId(), 75u);
    EXPECT_EQ(filled_occupancy_component_cell.Component(), 0u);
    EXPECT_EQ(filled_occupancy_component_cell.SpatialSegment(), 0u);

    EXPECT_EQ(converted_filled_occupancy_cell.Occupancy(), 0.75f);
    EXPECT_EQ(converted_filled_occupancy_cell.ObjectId(), 75u);

    EXPECT_EQ(converted_filled_occupancy_component_cell.Occupancy(), 0.75f);
    EXPECT_EQ(converted_filled_occupancy_component_cell.ObjectId(), 75u);
    EXPECT_EQ(converted_filled_occupancy_component_cell.Component(), 0u);
    EXPECT_EQ(converted_filled_occupancy_component_cell.SpatialSegment(), 0u);
  }
}

GTEST_TEST(OccupancyMapConversionsTest, DefaultOccupancyMapConversions)
{
  {
    const OccupancyMap default_occupancy_map;
    const OccupancyComponentMap default_occupancy_component_map;

    const OccupancyComponentMap converted_default_occupancy_component_map =
        ConvertToOccupancyComponentMap(default_occupancy_map);
    const OccupancyMap converted_default_occupancy_map =
        ConvertFromOccupancyComponentMap(default_occupancy_component_map);

    CheckOccupancyMapsEquivalent(
        default_occupancy_map, default_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        default_occupancy_map, converted_default_occupancy_map);
    CheckOccupancyMapsEquivalent(
        default_occupancy_component_map,
        converted_default_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        converted_default_occupancy_map,
        converted_default_occupancy_component_map);
  }

  {
    const TaggedObjectOccupancyMap default_occupancy_map;
    const TaggedObjectOccupancyComponentMap default_occupancy_component_map;

    const TaggedObjectOccupancyComponentMap
        converted_default_occupancy_component_map =
            ConvertToTaggedObjectOccupancyComponentMap(default_occupancy_map);
    const TaggedObjectOccupancyMap converted_default_occupancy_map =
        ConvertFromTaggedObjectOccupancyComponentMap(
            default_occupancy_component_map);

    CheckOccupancyMapsEquivalent(
        default_occupancy_map, default_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        default_occupancy_map, converted_default_occupancy_map);
    CheckOccupancyMapsEquivalent(
        default_occupancy_component_map,
        converted_default_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        converted_default_occupancy_map,
        converted_default_occupancy_component_map);
  }
}

GTEST_TEST(OccupancyMapConversionsTest, FilledOccupancyMapConversions)
{
  const Eigen::Isometry3d origin_transform =
      Eigen::Translation3d(1.0, 2.0, 3.0) *
      Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitZ());

  constexpr double resolution = 0.25;
  constexpr int64_t num_x_cells = 5;
  constexpr int64_t num_y_cells = 10;
  constexpr int64_t num_z_cells = 20;
  const auto grid_sizes = VoxelGridSizes::FromVoxelCounts(
      resolution, Vector3i64(num_x_cells, num_y_cells, num_z_cells));

  const std::string frame = "test_frame";

  {
    const OccupancyCell filled_occupancy_cell(0.75f);
    const OccupancyComponentCell filled_occupancy_component_cell(0.75f);

    const OccupancyMap filled_occupancy_map(
        origin_transform, frame, grid_sizes, filled_occupancy_cell);
    const OccupancyComponentMap filled_occupancy_component_map(
        origin_transform, frame, grid_sizes, filled_occupancy_component_cell);

    const OccupancyComponentMap converted_filled_occupancy_component_map =
        ConvertToOccupancyComponentMap(filled_occupancy_map);
    const OccupancyMap converted_filled_occupancy_map =
        ConvertFromOccupancyComponentMap(filled_occupancy_component_map);

    CheckOccupancyMapsEquivalent(
        filled_occupancy_map, filled_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        filled_occupancy_map, converted_filled_occupancy_map);
    CheckOccupancyMapsEquivalent(
        filled_occupancy_component_map,
        converted_filled_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        converted_filled_occupancy_map,
        converted_filled_occupancy_component_map);
  }

  {
    const TaggedObjectOccupancyCell filled_occupancy_cell(0.75f, 75u);
    const TaggedObjectOccupancyComponentCell filled_occupancy_component_cell(
        0.75f, 75u);

    const TaggedObjectOccupancyMap filled_occupancy_map(
        origin_transform, frame, grid_sizes, filled_occupancy_cell);
    const TaggedObjectOccupancyComponentMap filled_occupancy_component_map(
        origin_transform, frame, grid_sizes, filled_occupancy_component_cell);

    const TaggedObjectOccupancyComponentMap
        converted_filled_occupancy_component_map =
            ConvertToTaggedObjectOccupancyComponentMap(filled_occupancy_map);
    const TaggedObjectOccupancyMap converted_filled_occupancy_map =
        ConvertFromTaggedObjectOccupancyComponentMap(
            filled_occupancy_component_map);

    CheckOccupancyMapsEquivalent(
        filled_occupancy_map, filled_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        filled_occupancy_map, converted_filled_occupancy_map);
    CheckOccupancyMapsEquivalent(
        filled_occupancy_component_map,
        converted_filled_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        converted_filled_occupancy_map,
        converted_filled_occupancy_component_map);
  }
}

GTEST_TEST(OccupancyMapConversionsTest, PatternOccupancyMapConversions)
{
  const Eigen::Isometry3d origin_transform =
      Eigen::Translation3d(1.0, 2.0, 3.0) *
      Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitZ());

  constexpr double resolution = 0.25;
  constexpr int64_t num_x_cells = 5;
  constexpr int64_t num_y_cells = 10;
  constexpr int64_t num_z_cells = 20;
  const auto grid_sizes = VoxelGridSizes::FromVoxelCounts(
      resolution, Vector3i64(num_x_cells, num_y_cells, num_z_cells));

  const std::string frame = "test_frame";

  {
    const OccupancyCell empty_occupancy_cell(0.0f);
    const OccupancyComponentCell empty_occupancy_component_cell(0.0f);
    const OccupancyCell filled_occupancy_cell(0.75f);
    const OccupancyComponentCell filled_occupancy_component_cell(0.75f);

    OccupancyMap pattern_occupancy_map(
        origin_transform, frame, grid_sizes, filled_occupancy_cell);
    OccupancyComponentMap pattern_occupancy_component_map(
        origin_transform, frame, grid_sizes, filled_occupancy_component_cell);

    auto& occupancy_backing_store = pattern_occupancy_map.GetMutableRawData();
    auto& occupancy_component_backing_store =
        pattern_occupancy_component_map.GetMutableRawData();

    ASSERT_EQ(
        occupancy_backing_store.size(),
        occupancy_component_backing_store.size());

    for (size_t index = 0; index < occupancy_backing_store.size(); index++)
    {
      if ((index % 3) == 0)
      {
        occupancy_backing_store.at(index) = empty_occupancy_cell;
        occupancy_component_backing_store.at(index) =
            empty_occupancy_component_cell;
      }
    }

    const OccupancyComponentMap converted_pattern_occupancy_component_map =
        ConvertToOccupancyComponentMap(pattern_occupancy_map);
    const OccupancyMap converted_pattern_occupancy_map =
        ConvertFromOccupancyComponentMap(pattern_occupancy_component_map);

    CheckOccupancyMapsEquivalent(
        pattern_occupancy_map, pattern_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        pattern_occupancy_map, converted_pattern_occupancy_map);
    CheckOccupancyMapsEquivalent(
        pattern_occupancy_component_map,
        converted_pattern_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        converted_pattern_occupancy_map,
        converted_pattern_occupancy_component_map);
  }

  {
    const TaggedObjectOccupancyCell empty_occupancy_cell(0.25f);
    const TaggedObjectOccupancyComponentCell empty_occupancy_component_cell(
        0.25f);
    const TaggedObjectOccupancyCell filled_occupancy_cell(0.75f, 75u);
    const TaggedObjectOccupancyComponentCell filled_occupancy_component_cell(
        0.75f, 75u);

    TaggedObjectOccupancyMap pattern_occupancy_map(
        origin_transform, frame, grid_sizes, filled_occupancy_cell);
    TaggedObjectOccupancyComponentMap pattern_occupancy_component_map(
        origin_transform, frame, grid_sizes, filled_occupancy_component_cell);

    auto& occupancy_backing_store = pattern_occupancy_map.GetMutableRawData();
    auto& occupancy_component_backing_store =
        pattern_occupancy_component_map.GetMutableRawData();

    ASSERT_EQ(
        occupancy_backing_store.size(),
        occupancy_component_backing_store.size());

    for (size_t index = 0; index < occupancy_backing_store.size(); index++)
    {
      if ((index % 3) == 0)
      {
        occupancy_backing_store.at(index) = empty_occupancy_cell;
        occupancy_component_backing_store.at(index) =
            empty_occupancy_component_cell;
      }
    }

    const TaggedObjectOccupancyComponentMap
        converted_pattern_occupancy_component_map =
            ConvertToTaggedObjectOccupancyComponentMap(pattern_occupancy_map);
    const TaggedObjectOccupancyMap converted_pattern_occupancy_map =
        ConvertFromTaggedObjectOccupancyComponentMap(
            pattern_occupancy_component_map);

    CheckOccupancyMapsEquivalent(
        pattern_occupancy_map, pattern_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        pattern_occupancy_map, converted_pattern_occupancy_map);
    CheckOccupancyMapsEquivalent(
        pattern_occupancy_component_map,
        converted_pattern_occupancy_component_map);
    CheckOccupancyMapsEquivalent(
        converted_pattern_occupancy_map,
        converted_pattern_occupancy_component_map);
  }
}

}  // namespace
}  // namespace voxelized_geometry_tools

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
