#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/color_builder.hpp>
#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <gtest/gtest.h>
#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>

using common_robotics_utilities::math::Interpolate;
using common_robotics_utilities::utility::UniformUnitRealFunction;
using common_robotics_utilities::voxel_grid::GridIndex;
using common_robotics_utilities::voxel_grid::Vector3i64;
using common_robotics_utilities::voxel_grid::VoxelGridSizes;
using voxelized_geometry_tools::pointcloud_voxelization
    ::CpuPointCloudVoxelizer;

namespace
{
void RunRaycastCycle(const UniformUnitRealFunction& uniform_unit_real_fn)
{
  // Define tracking grid for raycasting work.
  constexpr double resolution = 0.125;
  const auto voxel_grid_sizes =
      VoxelGridSizes::FromVoxelCounts(resolution, Vector3i64(40, 40, 40));
  const Eigen::Isometry3d origin_transform(Eigen::Translation3d(0.0, 0.0, 0.0));
  CpuPointCloudVoxelizer::CpuVoxelizationTrackingGrid raycast_grid(
      origin_transform, voxel_grid_sizes,
      CpuPointCloudVoxelizer::CpuVoxelizationTrackingCell());

  constexpr double min_axis_value = -2.0;
  constexpr double max_axis_value = 7.0;

  const auto sample_point = [](const UniformUnitRealFunction& draw_fn)
  {
    const double x = Interpolate(min_axis_value, max_axis_value, draw_fn());
    const double y = Interpolate(min_axis_value, max_axis_value, draw_fn());
    const double z = Interpolate(min_axis_value, max_axis_value, draw_fn());
    return Eigen::Vector4d(x, y, z, 1.0);
  };

  const Eigen::Vector4d origin = sample_point(uniform_unit_real_fn);
  const Eigen::Vector4d point = sample_point(uniform_unit_real_fn);

  constexpr double max_range = 10.0;

  const CpuPointCloudVoxelizer raycaster({});

  raycaster.RaycastSinglePoint(origin, point, max_range, raycast_grid);

  const int64_t num_x_voxels = raycast_grid.NumXVoxels();
  const int64_t num_y_voxels = raycast_grid.NumYVoxels();
  const int64_t num_z_voxels = raycast_grid.NumZVoxels();

  for (int64_t xidx = 0; xidx < num_x_voxels; xidx++)
  {
    for (int64_t yidx = 0; yidx < num_y_voxels; yidx++)
    {
      for (int64_t zidx = 0; zidx < num_z_voxels; zidx++)
      {
        const auto query = raycast_grid.GetIndexImmutable(xidx, yidx, zidx);
        const auto& cell = query.Value();
        const int32_t seen_free_count = cell.seen_free_count.load();
        const int32_t seen_filled_count = cell.seen_filled_count.load();

        EXPECT_GE(seen_free_count, 0);
        EXPECT_GE(seen_filled_count, 0);
        EXPECT_LE(seen_free_count, 1);
        EXPECT_LE(seen_filled_count, 1);

        const bool seen_free = seen_free_count > 0;
        const bool seen_filled = seen_filled_count > 0;
        EXPECT_FALSE(seen_free && seen_filled);
      }
    }
  }
}

GTEST_TEST(VoxelRaycastingTest, Test)
{

  std::mt19937_64 prng(42);

  const UniformUnitRealFunction uniform_unit_real_fn = [&]()
  {
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(
        prng);
  };

  const int32_t iterations = 1000;
  for (int32_t iter = 0; iter < iterations; iter++)
  {
    RunRaycastCycle(uniform_unit_real_fn);
  }
}
}  // namespace
