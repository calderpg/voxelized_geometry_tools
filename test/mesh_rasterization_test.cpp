#include <iostream>
#include <vector>

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <voxelized_geometry_tools/mesh_rasterizer.hpp>

using common_robotics_utilities::parallelism::DegreeOfParallelism;

namespace voxelized_geometry_tools
{
namespace
{
class MeshRasterizationTestSuite
    : public testing::TestWithParam<DegreeOfParallelism> {};

TEST_P(MeshRasterizationTestSuite, TestOccupancyMap)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const std::vector<Eigen::Vector3d> vertices = {
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0) };
  const std::vector<Eigen::Vector3i> triangles = { Eigen::Vector3i(0, 1, 2) };

  const double resolution = 0.125;

  const auto occupancy_map = mesh_rasterizer::RasterizeMeshIntoOccupancyMap(
      vertices, triangles, resolution, parallelism);

  const auto get_cell_occupancy =
      [&] (const int64_t x, const int64_t y, const int64_t z)
  {
    return occupancy_map.GetIndexImmutable(x, y, z).Value().Occupancy();
  };

  // Due to how the triangle discretizes, we expect the lower layer to be empty.
  for (int64_t xidx = 0; xidx < occupancy_map.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < occupancy_map.GetNumYCells(); yidx++)
    {
      EXPECT_EQ(get_cell_occupancy(xidx, yidx, 0), 0.0f);
    }
  }

  // Check the upper layer of voxels.
  for (int64_t xidx = 0; xidx < occupancy_map.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < occupancy_map.GetNumYCells(); yidx++)
    {
      if (xidx == 0 || yidx == 0)
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 0.0f);
      }
      else if (yidx >= (occupancy_map.GetNumYCells() - xidx))
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 0.0f);
      }
      else
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 1.0f);
      }
    }
  }
}

TEST_P(MeshRasterizationTestSuite, TestOccupancyComponentMap)
{
  const DegreeOfParallelism parallelism = GetParam();
  std::cout << "# of threads = " << parallelism.GetNumThreads() << std::endl;

  const std::vector<Eigen::Vector3d> vertices = {
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0) };
  const std::vector<Eigen::Vector3i> triangles = { Eigen::Vector3i(0, 1, 2) };

  const double resolution = 0.125;

  const auto occupancy_map =
      mesh_rasterizer::RasterizeMeshIntoOccupancyComponentMap(
          vertices, triangles, resolution, parallelism);

  const auto get_cell_occupancy =
      [&] (const int64_t x, const int64_t y, const int64_t z)
  {
    return occupancy_map.GetIndexImmutable(x, y, z).Value().Occupancy();
  };

  // Due to how the triangle discretizes, we expect the lower layer to be empty.
  for (int64_t xidx = 0; xidx < occupancy_map.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < occupancy_map.GetNumYCells(); yidx++)
    {
      EXPECT_EQ(get_cell_occupancy(xidx, yidx, 0), 0.0f);
    }
  }

  // Check the upper layer of voxels.
  for (int64_t xidx = 0; xidx < occupancy_map.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < occupancy_map.GetNumYCells(); yidx++)
    {
      if (xidx == 0 || yidx == 0)
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 0.0f);
      }
      else if (yidx >= (occupancy_map.GetNumYCells() - xidx))
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 0.0f);
      }
      else
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 1.0f);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerialMeshRasterizationTest, MeshRasterizationTestSuite,
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
    ParallelMeshRasterizationTest, MeshRasterizationTestSuite,
    testing::Values(DegreeOfParallelism(GetNumThreads())));
}  // namespace
}  // namespace voxelized_geometry_tools

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
