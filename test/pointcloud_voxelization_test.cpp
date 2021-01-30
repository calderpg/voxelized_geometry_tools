#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/math.hpp>
#include <gtest/gtest.h>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/device_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization.hpp>

namespace voxelized_geometry_tools
{
namespace
{
using pointcloud_voxelization::CpuPointCloudVoxelizer;
using pointcloud_voxelization::CudaPointCloudVoxelizer;
using pointcloud_voxelization::OpenCLPointCloudVoxelizer;
using pointcloud_voxelization::PointCloudVoxelizationFilterOptions;
using pointcloud_voxelization::PointCloudVoxelizationInterface;
using pointcloud_voxelization::PointCloudWrapper;
using pointcloud_voxelization::PointCloudWrapperPtr;

class VectorVector3dPointCloudWrapper : public PointCloudWrapper
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void PushBack(const Eigen::Vector3d& point)
  {
    points_.push_back(point);
  }

  double MaxRange() const override
  {
    return std::numeric_limits<double>::infinity();
  }

  int64_t Size() const override { return static_cast<int64_t>(points_.size()); }

  const Eigen::Isometry3d& GetPointCloudOriginTransform() const override
  {
    return origin_transform_;
  }

  void SetPointCloudOriginTransform(
      const Eigen::Isometry3d& origin_transform) override
  {
    origin_transform_ = origin_transform;
  }

private:
  void CopyPointLocationIntoDoublePtrImpl(
      const int64_t point_index, double* destination) const override
  {
    const Eigen::Vector3d& point = points_.at(static_cast<size_t>(point_index));
    std::memcpy(destination, point.data(), sizeof(double) * 3);
  }

  void CopyPointLocationIntoFloatPtrImpl(
      const int64_t point_index, float* destination) const override
  {
    const Eigen::Vector3f point =
        points_.at(static_cast<size_t>(point_index)).cast<float>();
    std::memcpy(destination, point.data(), sizeof(float) * 3);
  }

  common_robotics_utilities::math::VectorVector3d points_;
  Eigen::Isometry3d origin_transform_ = Eigen::Isometry3d::Identity();
};

void check_voxelization(const CollisionMap& occupancy)
{
  // Make sure the grid is properly filled
  for (int64_t xidx = 0; xidx < occupancy.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < occupancy.GetNumYCells(); yidx++)
    {
      for (int64_t zidx = 0; zidx < occupancy.GetNumZCells(); zidx++)
      {
        // Check grid querying
        const auto occupancy_query = occupancy.GetImmutable(xidx, yidx, zidx);
        // Check grid values
        const float cmap_occupancy = occupancy_query.Value().Occupancy();
        // Check the bottom cells
        if (zidx == 0)
        {
          ASSERT_EQ(cmap_occupancy, 1.0f);
        }
        // Check a few "seen empty" cells
        if ((xidx == 3) && (yidx >= 3) && (zidx >= 1))
        {
          ASSERT_EQ(cmap_occupancy, 0.0f);
        }
        if ((xidx >= 3) && (yidx == 3) && (zidx >= 1))
        {
          ASSERT_EQ(cmap_occupancy, 0.0f);
        }
        // Check a few "seen filled" cells
        if ((xidx == 4) && (yidx >= 4) && (zidx >= 1))
        {
          ASSERT_EQ(cmap_occupancy, 1.0f);
        }
        if ((xidx >= 4) && (yidx == 4) && (zidx >= 1))
        {
          ASSERT_EQ(cmap_occupancy, 1.0f);
        }
        // Check shadowed cells
        if ((xidx > 4) && (yidx > 4) && (zidx >= 1))
        {
          ASSERT_EQ(cmap_occupancy, 0.5f);
        }
      }
    }
  }
}

GTEST_TEST(PointCloudVoxelizationTest, Test)
{
  // Make the static environment
  const Eigen::Isometry3d X_WG(Eigen::Translation3d(-1.0, -1.0, -1.0));
  // Grid 2m in each axis
  const double x_size = 2.0;
  const double y_size = 2.0;
  const double z_size = 2.0;
  // 1/4 meter resolution, so 8 cells/axis
  const double grid_resolution = 0.25;
  const double step_size_multiplier = 0.5;
  const CollisionCell empty_cell(0.0f);
  const CollisionCell filled_cell(1.0f);

  const common_robotics_utilities::voxel_grid::GridSizes grid_size(
      grid_resolution, x_size, y_size, z_size);
  CollisionMap static_environment(X_WG, "world", grid_size, empty_cell);

  // Set the bottom cells filled
  for (int64_t xidx = 0; xidx < static_environment.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < static_environment.GetNumYCells(); yidx++)
    {
      static_environment.SetValue(xidx, yidx, 0, filled_cell);
    }
  }

  // Make some test pointclouds
  // Make the physical->optical frame transform
  const Eigen::Isometry3d X_CO = Eigen::Translation3d(0.0, 0.0, 0.0) *
      Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX()));

  // Camera 1 pose
  const Eigen::Isometry3d X_WC1(Eigen::Translation3d(-2.0, 0.0, 0.0));
  const Eigen::Isometry3d X_WC1O = X_WC1 * X_CO;
  PointCloudWrapperPtr cam1_cloud(new VectorVector3dPointCloudWrapper());
  static_cast<VectorVector3dPointCloudWrapper*>(cam1_cloud.get())
      ->SetPointCloudOriginTransform(X_WC1O);
  for (double x = -2.0; x <= 2.0; x += 0.03125)
  {
    for (double y = -2.0; y <= 2.0; y += 0.03125)
    {
      const double z = (x <= 0.0) ? 2.125 : 4.0;
      static_cast<VectorVector3dPointCloudWrapper*>(cam1_cloud.get())->PushBack(
          Eigen::Vector3d(x, y, z));
    }
  }

  // Camera 2 pose
  const Eigen::Isometry3d X_WC2 = Eigen::Translation3d(0.0, -2.0, 0.0) *
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()));
  const Eigen::Isometry3d X_WC2O = X_WC2 * X_CO;
  PointCloudWrapperPtr cam2_cloud(new VectorVector3dPointCloudWrapper());
  static_cast<VectorVector3dPointCloudWrapper*>(cam2_cloud.get())
      ->SetPointCloudOriginTransform(X_WC2O);
  for (double x = -2.0; x <= 2.0; x += 0.03125)
  {
    for (double y = -2.0; y <= 2.0; y += 0.03125)
    {
      const double z = (x >= 0.0) ? 2.125 : 4.0;
      static_cast<VectorVector3dPointCloudWrapper*>(cam2_cloud.get())->PushBack(
          Eigen::Vector3d(x, y, z));
    }
  }

  // Make control parameters
  // We require that 100% of points from the camera see through to see a voxel
  // as free.
  const double percent_seen_free = 1.0;
  // We don't worry about outliers.
  const int32_t outlier_points_threshold = 1;
  // We only need one camera to see a voxel as free.
  const int32_t num_cameras_seen_free = 1;
  const PointCloudVoxelizationFilterOptions filter_options(
      percent_seen_free, outlier_points_threshold, num_cameras_seen_free);
  // Voxelizer options (leave as default).
  const std::map<std::string, int32_t> options;

  // CUDA voxelizer
  try
  {
    std::cout << "Trying CUDA PointCloud Voxelizer..." << std::endl;
    std::unique_ptr<PointCloudVoxelizationInterface> voxelizer(
        new CudaPointCloudVoxelizer(options));
    const auto cuda_voxelized = voxelizer->VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options,
        {cam1_cloud, cam2_cloud});
    check_voxelization(cuda_voxelized);
  }
  catch (std::runtime_error& ex)
  {
    std::cerr << ex.what() << std::endl;
  }

  // OpenCL voxelizer
  try
  {
    std::cout << "Trying OpenCL PointCloud Voxelizer..." << std::endl;
    std::unique_ptr<PointCloudVoxelizationInterface> voxelizer(
        new OpenCLPointCloudVoxelizer(options));
    const auto opencl_voxelized = voxelizer->VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options,
        {cam1_cloud, cam2_cloud});
    check_voxelization(opencl_voxelized);
  }
  catch (std::runtime_error& ex)
  {
    std::cerr << ex.what() << std::endl;
  }

  // CPU voxelizer
  try
  {
    std::cout << "Trying CPU PointCloud Voxelizer..." << std::endl;
    std::unique_ptr<PointCloudVoxelizationInterface> voxelizer(
        new CpuPointCloudVoxelizer());
    const auto cpu_voxelized = voxelizer->VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options,
        {cam1_cloud, cam2_cloud});
    check_voxelization(cpu_voxelized);
  }
  catch (std::runtime_error&)
  {
    throw std::runtime_error("CPU PointCloud Voxelizer is not available");
  }
}
}  // namespace
}  // namespace voxelized_geometry_tools

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
