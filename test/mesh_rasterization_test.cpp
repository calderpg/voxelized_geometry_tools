#include <iostream>
#include <vector>

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <voxelized_geometry_tools/mesh_rasterizer.hpp>

namespace voxelized_geometry_tools
{
namespace
{
GTEST_TEST(MeshRasterizationTest, Test)
{
  const std::vector<Eigen::Vector3d> vertices = {
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0) };
  const std::vector<Eigen::Vector3i> triangles = { Eigen::Vector3i(0, 1, 2) };

  const double resolution = 0.125;
  const auto collision_map =
      RasterizeMeshIntoCollisionMap(vertices, triangles, resolution);

  const auto get_cell_occupancy =
      [&] (const int64_t x, const int64_t y, const int64_t z)
  {
    return collision_map.GetImmutable(x, y, z).Value().Occupancy();
  };

  // Due to how the triangle discretizes, we expect the lower layer to be empty.
  for (int64_t xidx = 0; xidx < collision_map.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < collision_map.GetNumYCells(); yidx++)
    {
      EXPECT_EQ(get_cell_occupancy(xidx, yidx, 0), 0.0f);
    }
  }

  // Check the upper layer of voxels.
  for (int64_t xidx = 0; xidx < collision_map.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < collision_map.GetNumYCells(); yidx++)
    {
      if (xidx == 0 || yidx == 0)
      {
        EXPECT_EQ(get_cell_occupancy(xidx, yidx, 1), 0.0f);
      }
      else if (yidx >= (collision_map.GetNumYCells() - xidx))
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
}  // namespace
}  // namespace voxelized_geometry_tools

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

