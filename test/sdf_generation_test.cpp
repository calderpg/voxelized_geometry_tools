#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <voxelized_geometry_tools/occupancy_component_map.hpp>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_component_map.hpp>
#include <voxelized_geometry_tools/tagged_object_occupancy_map.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::voxel_grid::GridIndex;
using common_robotics_utilities::voxel_grid::GridSizes;

namespace voxelized_geometry_tools
{
namespace
{
constexpr double kExtremaTolerance = 0.0001;

template<typename ScalarType>
SignedDistanceFieldGenerationParameters<ScalarType> SDFGenerationParams(
    const DegreeOfParallelism& parallelism)
{
  return SignedDistanceFieldGenerationParameters<ScalarType>(
      std::numeric_limits<ScalarType>::infinity(), parallelism, true, false);
}

template<typename ScalarType>
struct GeneratedSignedDistanceFields
{
  SignedDistanceField<ScalarType> occupancy_map_sdf;
  SignedDistanceField<ScalarType> occupancy_component_map_sdf;
  SignedDistanceField<ScalarType> tagged_object_occupancy_map_sdf;
  SignedDistanceField<ScalarType> tagged_object_occupancy_component_map_sdf;
};

template<typename ScalarType>
GeneratedSignedDistanceFields<ScalarType> GenerateSignedDistanceFields(
    const OccupancyMap& occupancy_map,
    const OccupancyComponentMap& occupancy_component_map,
    const TaggedObjectOccupancyMap& tagged_object_occupancy_map,
    const TaggedObjectOccupancyComponentMap&
        tagged_object_occupancy_component_map,
    const DegreeOfParallelism& parallelism)
{
  // Enforce occupancy map sizes match
  const int64_t num_x_cells = occupancy_map.GetNumXCells();
  const int64_t num_y_cells = occupancy_map.GetNumYCells();
  const int64_t num_z_cells = occupancy_map.GetNumZCells();

  EXPECT_EQ(num_x_cells, occupancy_component_map.GetNumXCells());
  EXPECT_EQ(num_y_cells, occupancy_component_map.GetNumYCells());
  EXPECT_EQ(num_z_cells, occupancy_component_map.GetNumZCells());

  EXPECT_EQ(num_x_cells, tagged_object_occupancy_map.GetNumXCells());
  EXPECT_EQ(num_y_cells, tagged_object_occupancy_map.GetNumYCells());
  EXPECT_EQ(num_z_cells, tagged_object_occupancy_map.GetNumZCells());

  EXPECT_EQ(num_x_cells, tagged_object_occupancy_component_map.GetNumXCells());
  EXPECT_EQ(num_y_cells, tagged_object_occupancy_component_map.GetNumYCells());
  EXPECT_EQ(num_z_cells, tagged_object_occupancy_component_map.GetNumZCells());

  // Make SDFs
  const auto occupancy_map_sdf =
      occupancy_map.ExtractSignedDistanceField<ScalarType>(
          SDFGenerationParams<ScalarType>(parallelism));
  const auto occupancy_component_map_sdf =
      occupancy_component_map.ExtractSignedDistanceField<ScalarType>(
          SDFGenerationParams<ScalarType>(parallelism));
  const auto tagged_object_occupancy_map_sdf =
      tagged_object_occupancy_map.ExtractSignedDistanceField<ScalarType>(
          {}, SDFGenerationParams<ScalarType>(parallelism));
  const auto tagged_object_occupancy_component_map_sdf =
      tagged_object_occupancy_component_map
          .ExtractSignedDistanceField<ScalarType>(
              {}, SDFGenerationParams<ScalarType>(parallelism));

  // Enforce SDF sizes match
  EXPECT_EQ(num_x_cells, occupancy_map_sdf.GetNumXCells());
  EXPECT_EQ(num_y_cells, occupancy_map_sdf.GetNumYCells());
  EXPECT_EQ(num_z_cells, occupancy_map_sdf.GetNumZCells());

  EXPECT_EQ(num_x_cells, occupancy_component_map_sdf.GetNumXCells());
  EXPECT_EQ(num_y_cells, occupancy_component_map_sdf.GetNumYCells());
  EXPECT_EQ(num_z_cells, occupancy_component_map_sdf.GetNumZCells());

  EXPECT_EQ(num_x_cells, tagged_object_occupancy_map_sdf.GetNumXCells());
  EXPECT_EQ(num_y_cells, tagged_object_occupancy_map_sdf.GetNumYCells());
  EXPECT_EQ(num_z_cells, tagged_object_occupancy_map_sdf.GetNumZCells());

  EXPECT_EQ(
      num_x_cells, tagged_object_occupancy_component_map_sdf.GetNumXCells());
  EXPECT_EQ(
      num_y_cells, tagged_object_occupancy_component_map_sdf.GetNumYCells());
  EXPECT_EQ(
      num_z_cells, tagged_object_occupancy_component_map_sdf.GetNumZCells());

  return GeneratedSignedDistanceFields<ScalarType>{
      occupancy_map_sdf, occupancy_component_map_sdf,
      tagged_object_occupancy_map_sdf,
      tagged_object_occupancy_component_map_sdf};
}

template<typename ScalarType>
bool CloseEnough(const ScalarType a, const ScalarType b)
{
  if (a == b)
  {
    return true;
  }
  else
  {
    constexpr ScalarType threshold = static_cast<ScalarType>(kExtremaTolerance);
    const ScalarType delta = std::abs(a -b);
    if (delta <= threshold)
    {
      return true;
    }
    else
    {
      std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "a: " << a << std::endl;
      std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "b: " << b << std::endl;
      std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "threshold: " << threshold << std::endl;
      std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "delta: " << delta << std::endl;
      return false;
    }
  }
}

class SDFGenerationTestSuite
    : public testing::TestWithParam<DegreeOfParallelism> {};

template<typename ScalarType>
void TestSDFGeneration(
    const OccupancyMap& occupancy_map,
    const OccupancyComponentMap& occupancy_component_map,
    const TaggedObjectOccupancyMap& tagged_object_occupancy_map,
    const TaggedObjectOccupancyComponentMap&
        tagged_object_occupancy_component_map,
    const DegreeOfParallelism& parallelism,
    const ScalarType expected_sdf_minimum,
    const ScalarType expected_sdf_maximum)
{
  const auto generated_sdfs = GenerateSignedDistanceFields<ScalarType>(
      occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
      tagged_object_occupancy_component_map, parallelism);

  const auto& occupancy_map_sdf = generated_sdfs.occupancy_map_sdf;
  const auto& occupancy_component_map_sdf =
      generated_sdfs.occupancy_component_map_sdf;
  const auto& tagged_object_occupancy_map_sdf =
      generated_sdfs.tagged_object_occupancy_map_sdf;
  const auto& tagged_object_occupancy_component_map_sdf =
      generated_sdfs.tagged_object_occupancy_component_map_sdf;

  // Check expected extrema
  EXPECT_TRUE(CloseEnough(
      occupancy_map_sdf.GetMinimumMaximum().Minimum(), expected_sdf_minimum));
  EXPECT_TRUE(CloseEnough(
      occupancy_map_sdf.GetMinimumMaximum().Maximum(), expected_sdf_maximum));

  EXPECT_TRUE(CloseEnough(
      occupancy_component_map_sdf.GetMinimumMaximum().Minimum(),
      expected_sdf_minimum));
  EXPECT_TRUE(CloseEnough(
      occupancy_component_map_sdf.GetMinimumMaximum().Maximum(),
      expected_sdf_maximum));

  EXPECT_TRUE(CloseEnough(
      tagged_object_occupancy_map_sdf.GetMinimumMaximum().Minimum(),
      expected_sdf_minimum));
  EXPECT_TRUE(CloseEnough(
      tagged_object_occupancy_map_sdf.GetMinimumMaximum().Maximum(),
      expected_sdf_maximum));

  EXPECT_TRUE(CloseEnough(
      tagged_object_occupancy_component_map_sdf.GetMinimumMaximum().Minimum(),
      expected_sdf_minimum));
  EXPECT_TRUE(CloseEnough(
      tagged_object_occupancy_component_map_sdf.GetMinimumMaximum().Maximum(),
      expected_sdf_maximum));

  // Check elements
  const ScalarType zero = static_cast<ScalarType>(0);

  const auto get_sdf_index_value = [](
      const SignedDistanceField<ScalarType>& sdf, const GridIndex& index)
      -> ScalarType {
    return sdf.GetIndexImmutable(index).Value();
  };

  const int64_t num_x_cells = occupancy_map.GetNumXCells();
  const int64_t num_y_cells = occupancy_map.GetNumYCells();
  const int64_t num_z_cells = occupancy_map.GetNumZCells();

  for (int64_t x_index = 0; x_index < num_x_cells; x_index++)
  {
    for (int64_t y_index = 0; y_index < num_y_cells; y_index++)
    {
      for (int64_t z_index = 0; z_index < num_z_cells; z_index++)
      {
        const GridIndex index(x_index, y_index, z_index);

        const float occupancy_map_occupancy =
            occupancy_map.GetIndexImmutable(index).Value().Occupancy();
        const float occupancy_component_map_occupancy =
            occupancy_component_map.GetIndexImmutable(
                index).Value().Occupancy();
        const float tagged_object_occupancy_map_occupancy =
            tagged_object_occupancy_map.GetIndexImmutable(
                index).Value().Occupancy();
        const float tagged_object_occupancy_component_map_occupancy =
            tagged_object_occupancy_component_map.GetIndexImmutable(
                index).Value().Occupancy();

        EXPECT_EQ(occupancy_map_occupancy, occupancy_component_map_occupancy);
        EXPECT_EQ(
            occupancy_map_occupancy, tagged_object_occupancy_map_occupancy);
        EXPECT_EQ(
            occupancy_map_occupancy,
            tagged_object_occupancy_component_map_occupancy);

        if (occupancy_map_occupancy >= 0.5f)
        {
          EXPECT_LT(get_sdf_index_value(occupancy_map_sdf, index), zero);
          EXPECT_LT(
              get_sdf_index_value(occupancy_component_map_sdf, index), zero);
          EXPECT_LT(
              get_sdf_index_value(tagged_object_occupancy_map_sdf, index),
              zero);
          EXPECT_LT(
              get_sdf_index_value(
                  tagged_object_occupancy_component_map_sdf, index),
              zero);
        }
        else
        {
          EXPECT_GT(get_sdf_index_value(occupancy_map_sdf, index), zero);
          EXPECT_GT(
              get_sdf_index_value(occupancy_component_map_sdf, index), zero);
          EXPECT_GT(
              get_sdf_index_value(tagged_object_occupancy_map_sdf, index),
              zero);
          EXPECT_GT(
              get_sdf_index_value(
                  tagged_object_occupancy_component_map_sdf, index),
              zero);
        }
      }
    }
  }
}

TEST_P(SDFGenerationTestSuite, FullyFilledTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 0.25;
  const double x_size = 1.0;
  const double y_size = 2.0;
  const double z_size = 3.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);
  // Center the grid around the origin
  const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make filled occupancy map types
  const OccupancyMap filled_occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(1.0f));

  const OccupancyComponentMap filled_occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(1.0f));

  const TaggedObjectOccupancyMap filled_tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(1.0f, 1u));

  const TaggedObjectOccupancyComponentMap
      filled_tagged_object_occupancy_component_map(
          origin_transform, frame, grid_sizes,
          TaggedObjectOccupancyComponentCell(1.0f, 1u));

  // Test generation of float SDFs
  {
    const float negative_inf = -std::numeric_limits<float>::infinity();
    TestSDFGeneration<float>(
        filled_occupancy_map, filled_occupancy_component_map,
        filled_tagged_object_occupancy_map,
        filled_tagged_object_occupancy_component_map, parallelism, negative_inf,
        negative_inf);
  }

  // Test generation of double SDFs
  {
    const double negative_inf = -std::numeric_limits<double>::infinity();
    TestSDFGeneration<double>(
        filled_occupancy_map, filled_occupancy_component_map,
        filled_tagged_object_occupancy_map,
        filled_tagged_object_occupancy_component_map, parallelism, negative_inf,
        negative_inf);
  }
}

TEST_P(SDFGenerationTestSuite, FullyEmptyTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 0.25;
  const double x_size = 1.0;
  const double y_size = 2.0;
  const double z_size = 3.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);
  // Center the grid around the origin
  const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make empty occupancy map types
  const OccupancyMap empty_occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  const OccupancyComponentMap empty_occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  const TaggedObjectOccupancyMap empty_tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  const TaggedObjectOccupancyComponentMap
      empty_tagged_object_occupancy_component_map(
          origin_transform, frame, grid_sizes,
          TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Test generation of float SDFs
  {
    const float positive_inf = std::numeric_limits<float>::infinity();
    TestSDFGeneration<float>(
        empty_occupancy_map, empty_occupancy_component_map,
        empty_tagged_object_occupancy_map,
        empty_tagged_object_occupancy_component_map, parallelism, positive_inf,
        positive_inf);
  }

  // Test generation of double SDFs
  {
    const double positive_inf = std::numeric_limits<double>::infinity();
    TestSDFGeneration<double>(
        empty_occupancy_map, empty_occupancy_component_map,
        empty_tagged_object_occupancy_map,
        empty_tagged_object_occupancy_component_map, parallelism, positive_inf,
        positive_inf);
  }
}

TEST_P(SDFGenerationTestSuite, CenterObstacleTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 0.25;
  const double x_size = 1.0;
  const double y_size = 2.0;
  const double z_size = 3.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);
  // Center the grid around the origin
  const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make occupancy map types
  OccupancyMap occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  OccupancyComponentMap occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  TaggedObjectOccupancyMap tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  TaggedObjectOccupancyComponentMap tagged_object_occupancy_component_map(
      origin_transform, frame, grid_sizes,
      TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Fill an obstacle in the center of the grid
  for (int64_t x_index = 1; x_index < 3; x_index++)
  {
    for (int64_t y_index = 2; y_index < 6; y_index++)
    {
      for (int64_t z_index = 3; z_index < 9; z_index++)
      {
        occupancy_map.SetIndex(x_index, y_index, z_index, OccupancyCell(1.0f));
        occupancy_component_map.SetIndex(
            x_index, y_index, z_index, OccupancyComponentCell(1.0f));
        tagged_object_occupancy_map.SetIndex(
            x_index, y_index, z_index, TaggedObjectOccupancyCell(1.0f, 1u));
        tagged_object_occupancy_component_map.SetIndex(
            x_index, y_index, z_index,
            TaggedObjectOccupancyComponentCell(1.0f, 1u));
      }
    }
  }

  const double nominal_maximum =
      std::sqrt(std::pow(resolution, 2.0) +
                std::pow(2.0 * resolution, 2.0) +
                std::pow(3.0* resolution, 2.0));

  // Test generation of float SDFs
  {
    const float minimum = -0.25f;
    const float maximum = static_cast<float>(nominal_maximum);
    TestSDFGeneration<float>(
        occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
        tagged_object_occupancy_component_map, parallelism, minimum, maximum);
  }

  // Test generation of double SDFs
  {
    const double minimum = -0.25;
    const double maximum = nominal_maximum;
    TestSDFGeneration<double>(
        occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
        tagged_object_occupancy_component_map, parallelism, minimum, maximum);
  }
}

TEST_P(SDFGenerationTestSuite, CornerObstacleTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 0.25;
  const double x_size = 1.0;
  const double y_size = 2.0;
  const double z_size = 3.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);
  // Center the grid around the origin
  const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make occupancy map types
  OccupancyMap occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  OccupancyComponentMap occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  TaggedObjectOccupancyMap tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  TaggedObjectOccupancyComponentMap tagged_object_occupancy_component_map(
      origin_transform, frame, grid_sizes,
      TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Fill an obstacle in a corner of the grid
  for (int64_t x_index = 0; x_index < 2; x_index++)
  {
    for (int64_t y_index = 0; y_index < 4; y_index++)
    {
      for (int64_t z_index = 0; z_index < 6; z_index++)
      {
        occupancy_map.SetIndex(x_index, y_index, z_index, OccupancyCell(1.0f));
        occupancy_component_map.SetIndex(
            x_index, y_index, z_index, OccupancyComponentCell(1.0f));
        tagged_object_occupancy_map.SetIndex(
            x_index, y_index, z_index, TaggedObjectOccupancyCell(1.0f, 1u));
        tagged_object_occupancy_component_map.SetIndex(
            x_index, y_index, z_index,
            TaggedObjectOccupancyComponentCell(1.0f, 1u));
      }
    }
  }

  // Test generation of float SDFs
  {
    const float minimum = -0.5f;
    const float maximum = 1.8708f;
    TestSDFGeneration<float>(
        occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
        tagged_object_occupancy_component_map, parallelism, minimum, maximum);
  }

  // Test generation of double SDFs
  {
    const double minimum = -0.5;
    const double maximum = 1.8708;
    TestSDFGeneration<double>(
        occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
        tagged_object_occupancy_component_map, parallelism, minimum, maximum);
  }
}

TEST_P(SDFGenerationTestSuite, FaceObstacleTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 0.25;
  const double x_size = 1.0;
  const double y_size = 2.0;
  const double z_size = 3.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);
  // Center the grid around the origin
  const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make occupancy map types
  OccupancyMap occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  OccupancyComponentMap occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  TaggedObjectOccupancyMap tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  TaggedObjectOccupancyComponentMap tagged_object_occupancy_component_map(
      origin_transform, frame, grid_sizes,
      TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Fill an obstacle in a face of the grid
  const int64_t num_x_cells = occupancy_map.GetNumXCells();
  const int64_t num_y_cells = occupancy_map.GetNumYCells();

  constexpr int64_t z_index = 0;

  for (int64_t x_index = 0; x_index < num_x_cells; x_index++)
  {
    for (int64_t y_index = 0; y_index < num_y_cells; y_index++)
    {
      occupancy_map.SetIndex(x_index, y_index, z_index, OccupancyCell(1.0f));
      occupancy_component_map.SetIndex(
          x_index, y_index, z_index, OccupancyComponentCell(1.0f));
      tagged_object_occupancy_map.SetIndex(
          x_index, y_index, z_index, TaggedObjectOccupancyCell(1.0f, 1u));
      tagged_object_occupancy_component_map.SetIndex(
          x_index, y_index, z_index,
          TaggedObjectOccupancyComponentCell(1.0f, 1u));
    }
  }

  // Test generation of float SDFs
  {
    const float minimum = -0.25f;
    const float maximum = 2.75f;
    TestSDFGeneration<float>(
        occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
        tagged_object_occupancy_component_map, parallelism, minimum, maximum);
  }

  // Test generation of double SDFs
  {
    const double minimum = -0.25;
    const double maximum = 2.75;
    TestSDFGeneration<double>(
        occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
        tagged_object_occupancy_component_map, parallelism, minimum, maximum);
  }
}

TEST_P(SDFGenerationTestSuite, LinearExactTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 1.0;
  const double x_size = 1.0;
  const double y_size = 1.0;
  const double z_size = 4.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);

  const Eigen::Translation3d origin_translation(0.0, 0.0, 0.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make occupancy map types
  OccupancyMap occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  OccupancyComponentMap occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  TaggedObjectOccupancyMap tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  TaggedObjectOccupancyComponentMap tagged_object_occupancy_component_map(
      origin_transform, frame, grid_sizes,
      TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Fill an obstacle in a corner of the grid
  for (int64_t x_index = 0; x_index < 1; x_index++)
  {
    for (int64_t y_index = 0; y_index < 1; y_index++)
    {
      for (int64_t z_index = 0; z_index < 2; z_index++)
      {
        occupancy_map.SetIndex(x_index, y_index, z_index, OccupancyCell(1.0f));
        occupancy_component_map.SetIndex(
            x_index, y_index, z_index, OccupancyComponentCell(1.0f));
        tagged_object_occupancy_map.SetIndex(
            x_index, y_index, z_index, TaggedObjectOccupancyCell(1.0f, 1u));
        tagged_object_occupancy_component_map.SetIndex(
            x_index, y_index, z_index,
            TaggedObjectOccupancyComponentCell(1.0f, 1u));
      }
    }
  }

  // Make SDFs
  const auto generated_sdfs = GenerateSignedDistanceFields<float>(
      occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
      tagged_object_occupancy_component_map, parallelism);

  const auto& occupancy_map_sdf = generated_sdfs.occupancy_map_sdf;
  const auto& occupancy_component_map_sdf =
      generated_sdfs.occupancy_component_map_sdf;
  const auto& tagged_object_occupancy_map_sdf =
      generated_sdfs.tagged_object_occupancy_map_sdf;
  const auto& tagged_object_occupancy_component_map_sdf =
      generated_sdfs.tagged_object_occupancy_component_map_sdf;

  const auto get_occupancy_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return occupancy_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_occupancy_component_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return occupancy_component_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_tagged_object_occupancy_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return tagged_object_occupancy_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_tagged_object_occupancy_component_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return tagged_object_occupancy_component_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 0), -2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 0), -2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 0), -2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 0), -2.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 1), -1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 1), -1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 1), -1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 1), -1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 2), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 2), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 2), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 2), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 3), 2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 3), 2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 3), 2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 3), 2.0f);
}

TEST_P(SDFGenerationTestSuite, PlanarExactTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 1.0;
  const double x_size = 1.0;
  const double y_size = 4.0;
  const double z_size = 4.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);

  const Eigen::Translation3d origin_translation(0.0, 0.0, 0.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make occupancy map types
  OccupancyMap occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  OccupancyComponentMap occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  TaggedObjectOccupancyMap tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  TaggedObjectOccupancyComponentMap tagged_object_occupancy_component_map(
      origin_transform, frame, grid_sizes,
      TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Fill an obstacle in a corner of the grid
  for (int64_t x_index = 0; x_index < 1; x_index++)
  {
    for (int64_t y_index = 0; y_index < 2; y_index++)
    {
      for (int64_t z_index = 0; z_index < 2; z_index++)
      {
        occupancy_map.SetIndex(x_index, y_index, z_index, OccupancyCell(1.0f));
        occupancy_component_map.SetIndex(
            x_index, y_index, z_index, OccupancyComponentCell(1.0f));
        tagged_object_occupancy_map.SetIndex(
            x_index, y_index, z_index, TaggedObjectOccupancyCell(1.0f, 1u));
        tagged_object_occupancy_component_map.SetIndex(
            x_index, y_index, z_index,
            TaggedObjectOccupancyComponentCell(1.0f, 1u));
      }
    }
  }

  // Make SDFs
  const auto generated_sdfs = GenerateSignedDistanceFields<float>(
      occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
      tagged_object_occupancy_component_map, parallelism);

  const auto& occupancy_map_sdf = generated_sdfs.occupancy_map_sdf;
  const auto& occupancy_component_map_sdf =
      generated_sdfs.occupancy_component_map_sdf;
  const auto& tagged_object_occupancy_map_sdf =
      generated_sdfs.tagged_object_occupancy_map_sdf;
  const auto& tagged_object_occupancy_component_map_sdf =
      generated_sdfs.tagged_object_occupancy_component_map_sdf;

  const auto get_occupancy_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return occupancy_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_occupancy_component_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return occupancy_component_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_tagged_object_occupancy_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return tagged_object_occupancy_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_tagged_object_occupancy_component_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return tagged_object_occupancy_component_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 0), -2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 0), -2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 0), -2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 0), -2.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 1), -1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 1), -1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 1), -1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 1), -1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 2), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 2), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 2), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 2), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 3), 2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 3), 2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 3), 2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 3), 2.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 1, 0), -1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 1, 0), -1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 1, 0), -1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 1, 0), -1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 1, 1), -1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 1, 1), -1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 1, 1), -1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 1, 1), -1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 1, 2), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 1, 2), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 1, 2), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 1, 2), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 1, 3), 2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 1, 3), 2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 1, 3), 2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 1, 3), 2.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 2, 0), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 2, 0), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 2, 0), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 2, 0), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 2, 1), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 2, 1), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 2, 1), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 2, 1), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 2, 2), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(0, 2, 2), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(0, 2, 2), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 2, 2),
      std::sqrt(2.0f));

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 2, 3), std::sqrt(5.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(0, 2, 3), std::sqrt(5.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(0, 2, 3), std::sqrt(5.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 2, 3),
      std::sqrt(5.0f));

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 3, 0), 2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 3, 0), 2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 3, 0), 2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 3, 0), 2.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 3, 1), 2.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 3, 1), 2.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 3, 1), 2.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 3, 1), 2.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 3, 2), std::sqrt(5.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(0, 3, 2), std::sqrt(5.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(0, 3, 2), std::sqrt(5.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 3, 2),
      std::sqrt(5.0f));

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 3, 3), std::sqrt(8.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(0, 3, 3), std::sqrt(8.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(0, 3, 3), std::sqrt(8.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 3, 3),
      std::sqrt(8.0f));
}

TEST_P(SDFGenerationTestSuite, CubeExactTest)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const double resolution = 1.0;
  const double x_size = 2.0;
  const double y_size = 2.0;
  const double z_size = 2.0;
  const GridSizes grid_sizes(resolution, x_size, y_size, z_size);

  const Eigen::Translation3d origin_translation(0.0, 0.0, 0.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "test_frame";

  // Make occupancy map types
  OccupancyMap occupancy_map(
      origin_transform, frame, grid_sizes, OccupancyCell(0.0f));

  OccupancyComponentMap occupancy_component_map(
      origin_transform, frame, grid_sizes, OccupancyComponentCell(0.0f));

  TaggedObjectOccupancyMap tagged_object_occupancy_map(
      origin_transform, frame, grid_sizes, TaggedObjectOccupancyCell(0.0f, 0u));

  TaggedObjectOccupancyComponentMap tagged_object_occupancy_component_map(
      origin_transform, frame, grid_sizes,
      TaggedObjectOccupancyComponentCell(0.0f, 0u));

  // Fill an obstacle in a corner of the grid
  for (int64_t x_index = 0; x_index < 1; x_index++)
  {
    for (int64_t y_index = 0; y_index < 1; y_index++)
    {
      for (int64_t z_index = 0; z_index < 1; z_index++)
      {
        occupancy_map.SetIndex(x_index, y_index, z_index, OccupancyCell(1.0f));
        occupancy_component_map.SetIndex(
            x_index, y_index, z_index, OccupancyComponentCell(1.0f));
        tagged_object_occupancy_map.SetIndex(
            x_index, y_index, z_index, TaggedObjectOccupancyCell(1.0f, 1u));
        tagged_object_occupancy_component_map.SetIndex(
            x_index, y_index, z_index,
            TaggedObjectOccupancyComponentCell(1.0f, 1u));
      }
    }
  }

  // Make SDFs
  const auto generated_sdfs = GenerateSignedDistanceFields<float>(
      occupancy_map, occupancy_component_map, tagged_object_occupancy_map,
      tagged_object_occupancy_component_map, parallelism);

  const auto& occupancy_map_sdf = generated_sdfs.occupancy_map_sdf;
  const auto& occupancy_component_map_sdf =
      generated_sdfs.occupancy_component_map_sdf;
  const auto& tagged_object_occupancy_map_sdf =
      generated_sdfs.tagged_object_occupancy_map_sdf;
  const auto& tagged_object_occupancy_component_map_sdf =
      generated_sdfs.tagged_object_occupancy_component_map_sdf;

  const auto get_occupancy_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return occupancy_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_occupancy_component_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return occupancy_component_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_tagged_object_occupancy_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return tagged_object_occupancy_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  const auto get_tagged_object_occupancy_component_map_sdf_dist =
      [&](const int64_t x_index, const int64_t y_index, const int64_t z_index)
  {
    return tagged_object_occupancy_component_map_sdf.GetIndexImmutable(
        x_index, y_index, z_index).Value();
  };

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 0), -1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 0), -1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 0), -1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 0), -1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 0, 1), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 0, 1), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 0, 1), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 0, 1), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 1, 0), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(0, 1, 0), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(0, 1, 0), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 1, 0), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(0, 1, 1), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(0, 1, 1), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(0, 1, 1), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(0, 1, 1),
      std::sqrt(2.0f));

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(1, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(get_occupancy_component_map_sdf_dist(1, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(get_tagged_object_occupancy_map_sdf_dist(1, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(1, 0, 0), 1.0f);

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(1, 0, 1), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(1, 0, 1), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(1, 0, 1), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(1, 0, 1),
      std::sqrt(2.0f));

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(1, 1, 0), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(1, 1, 0), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(1, 1, 0), std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(1, 1, 0),
      std::sqrt(2.0f));

  EXPECT_FLOAT_EQ(get_occupancy_map_sdf_dist(1, 1, 1), std::sqrt(3.0f));
  EXPECT_FLOAT_EQ(
      get_occupancy_component_map_sdf_dist(1, 1, 1), std::sqrt(3.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_map_sdf_dist(1, 1, 1), std::sqrt(3.0f));
  EXPECT_FLOAT_EQ(
      get_tagged_object_occupancy_component_map_sdf_dist(1, 1, 1),
      std::sqrt(3.0f));
}

INSTANTIATE_TEST_SUITE_P(
    SerialSDFGenerationTest, SDFGenerationTestSuite,
    testing::Values(DegreeOfParallelism::None()));

// For fallback testing on platforms with no OpenMP support, specify 2 threads.
int32_t GetNumThreads()
{
  if (common_robotics_utilities::openmp_helpers::IsOmpEnabledInBuild())
  {
    return common_robotics_utilities::openmp_helpers::GetNumOmpThreads();
  }
  else
  {
    return 2;
  }
}

INSTANTIATE_TEST_SUITE_P(
    ParallelSDFGenerationTest, SDFGenerationTestSuite,
    testing::Values(DegreeOfParallelism(GetNumThreads())));
}  // namespace
}  // namespace voxelized_geometry_tools

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
