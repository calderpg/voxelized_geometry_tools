#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/color_builder.hpp>
#include <common_robotics_utilities/math.hpp>

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>
#else
#error "Undefined or unknown VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION"
#endif

#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/device_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/ros_interface.hpp>

using voxelized_geometry_tools::pointcloud_voxelization
    ::CpuPointCloudVoxelizer;
using voxelized_geometry_tools::pointcloud_voxelization
    ::CudaPointCloudVoxelizer;
using voxelized_geometry_tools::pointcloud_voxelization
    ::OpenCLPointCloudVoxelizer;
using voxelized_geometry_tools::pointcloud_voxelization
    ::PointCloudVoxelizationFilterOptions;
using voxelized_geometry_tools::pointcloud_voxelization
    ::PointCloudVoxelizationInterface;
using voxelized_geometry_tools::pointcloud_voxelization::PointCloudWrapper;
using voxelized_geometry_tools::pointcloud_voxelization
    ::PointCloudWrapperSharedPtr;

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
using ColorRGBA = std_msgs::msg::ColorRGBA;
using MarkerArray = visualization_msgs::msg::MarkerArray;
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
using ColorRGBA = std_msgs::ColorRGBA;
using MarkerArray = visualization_msgs::MarkerArray;
#endif

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

void check_equal(const float v1, const float v2)
{
  if (v1 != v2)
  {
    const std::string msg =
        std::to_string(v1) + " does not equal " + std::to_string(v2);
    throw std::runtime_error(msg);
  }
}

void check_voxelization(const voxelized_geometry_tools::CollisionMap& occupancy)
{
  // Make sure the grid is properly filled
  for (int64_t xidx = 0; xidx < occupancy.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < occupancy.GetNumYCells(); yidx++)
    {
      for (int64_t zidx = 0; zidx < occupancy.GetNumZCells(); zidx++)
      {
        // Check grid querying
        const auto occupancy_query =
            occupancy.GetIndexImmutable(xidx, yidx, zidx);
        // Check grid values
        const float cmap_occupancy = occupancy_query.Value().Occupancy();
        // Check the bottom cells
        if (zidx == 0)
        {
          check_equal(cmap_occupancy, 1.0f);
        }
        // Check a few "seen empty" cells
        if ((xidx == 3) && (yidx >= 3) && (zidx >= 1))
        {
          check_equal(cmap_occupancy, 0.0f);
        }
        if ((xidx >= 3) && (yidx == 3) && (zidx >= 1))
        {
          check_equal(cmap_occupancy, 0.0f);
        }
        // Check a few "seen filled" cells
        if ((xidx == 4) && (yidx >= 4) && (zidx >= 1))
        {
          check_equal(cmap_occupancy, 1.0f);
        }
        if ((xidx >= 4) && (yidx == 4) && (zidx >= 1))
        {
          check_equal(cmap_occupancy, 1.0f);
        }
        // Check shadowed cells
        if ((xidx > 4) && (yidx > 4) && (zidx >= 1))
        {
          check_equal(cmap_occupancy, 0.5f);
        }
      }
    }
  }
}

void test_pointcloud_voxelization(
    const std::function<void(const MarkerArray&)>& display_fn)
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
  const voxelized_geometry_tools::CollisionCell default_cell(0.0f);
  const common_robotics_utilities::voxel_grid::GridSizes grid_size(
      grid_resolution, x_size, y_size, z_size);
  voxelized_geometry_tools::CollisionMap static_environment(
      X_WG, "world", grid_size, default_cell);
  // Set the bottom cells filled
  for (int64_t xidx = 0; xidx < static_environment.GetNumXCells(); xidx++)
  {
    for (int64_t yidx = 0; yidx < static_environment.GetNumYCells(); yidx++)
    {
      static_environment.SetIndex(
          xidx, yidx, 0, voxelized_geometry_tools::CollisionCell(1.0f));
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
  PointCloudWrapperSharedPtr cam1_cloud(new VectorVector3dPointCloudWrapper());
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
  PointCloudWrapperSharedPtr cam2_cloud(new VectorVector3dPointCloudWrapper());
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
  // We don't worry about outliers
  const int32_t outlier_points_threshold = 1;
  // We only need one camera to see a voxel as free.
  const int32_t num_cameras_seen_free = 1;
  const PointCloudVoxelizationFilterOptions filter_options(
      percent_seen_free, outlier_points_threshold, num_cameras_seen_free);
  // Voxelize them
  MarkerArray display_markers;
  const ColorRGBA free_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(
              0.0, 0.25, 0.0, 0.5);
  const ColorRGBA filled_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(
              0.25, 0.0, 0.0, 0.5);
  const ColorRGBA unknown_color
      = common_robotics_utilities::color_builder
          ::MakeFromFloatColors<ColorRGBA>(
              0.0, 0.0, 0.25, 0.5);
  // Voxelizer options (leave as default)
  const std::map<std::string, int32_t> options;
  try
  {
    std::cout << "Trying CUDA PointCloud Voxelizer..." << std::endl;
    std::unique_ptr<PointCloudVoxelizationInterface> voxelizer(
        new CudaPointCloudVoxelizer(options));
    const auto cuda_voxelized = voxelizer->VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options,
        {cam1_cloud, cam2_cloud});
    check_voxelization(cuda_voxelized);
    auto environment_display =
        voxelized_geometry_tools::ros_interface
            ::ExportForSeparateDisplay(
                cuda_voxelized, filled_color, free_color, unknown_color);
    for (auto& marker : environment_display.markers)
    {
      marker.ns = "cuda_" + marker.ns;
    }
    display_markers.markers.insert(display_markers.markers.end(),
                                   environment_display.markers.begin(),
                                   environment_display.markers.end());
  }
  catch (const std::runtime_error& ex)
  {
    std::cerr << ex.what() << std::endl;
  }
  try
  {
    std::cout << "Trying OpenCL PointCloud Voxelizer..." << std::endl;
    std::unique_ptr<PointCloudVoxelizationInterface> voxelizer(
        new OpenCLPointCloudVoxelizer(options));
    const auto opencl_voxelized = voxelizer->VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options,
        {cam1_cloud, cam2_cloud});
    check_voxelization(opencl_voxelized);
    auto environment_display =
        voxelized_geometry_tools::ros_interface
            ::ExportForSeparateDisplay(
                opencl_voxelized, filled_color, free_color, unknown_color);
    for (auto& marker : environment_display.markers)
    {
      marker.ns = "opencl_" + marker.ns;
    }
    display_markers.markers.insert(display_markers.markers.end(),
                                   environment_display.markers.begin(),
                                   environment_display.markers.end());
  }
  catch (const std::runtime_error& ex)
  {
    std::cerr << ex.what() << std::endl;
  }
  try
  {
    std::cout << "Trying CPU PointCloud Voxelizer..." << std::endl;
    std::unique_ptr<PointCloudVoxelizationInterface> voxelizer(
        new CpuPointCloudVoxelizer(options));
    const auto cpu_voxelized = voxelizer->VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options,
        {cam1_cloud, cam2_cloud});
    check_voxelization(cpu_voxelized);
    auto environment_display =
        voxelized_geometry_tools::ros_interface
            ::ExportForSeparateDisplay(
                cpu_voxelized, filled_color, free_color, unknown_color);
    for (auto& marker : environment_display.markers)
    {
      marker.ns = "cpu_" + marker.ns;
    }
    display_markers.markers.insert(display_markers.markers.end(),
                                   environment_display.markers.begin(),
                                   environment_display.markers.end());
  }
  catch (const std::runtime_error&)
  {
    throw std::runtime_error("CPU PointCloud Voxelizer is not available");
  }
  // Draw the results
  display_fn(display_markers);
}

int main(int argc, char** argv)
{
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("pointcloud_voxelization_test");
  auto display_pub = node->create_publisher<MarkerArray>(
      "pointcloud_occupancy", rclcpp::QoS(1).transient_local());
  const std::function<void(const MarkerArray&)>& display_fn
      = [&] (const MarkerArray& markers)
  {
    display_pub->publish(markers);
  };
  const int32_t iterations =
      static_cast<int32_t>(node->declare_parameter(
          "iterations", rclcpp::ParameterValue(1)).get<int32_t>());
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ros::init(argc, argv, "pointcloud_voxelization_test");
  ros::NodeHandle nh;
  ros::NodeHandle nhp("~");
  ros::Publisher display_pub = nh.advertise<MarkerArray>(
      "pointcloud_occupancy", 1, true);
  const std::function<void(const MarkerArray&)>& display_fn
      = [&] (const MarkerArray& markers)
  {
    display_pub.publish(markers);
  };
  const int32_t iterations = nhp.param(std::string("iterations"), 1);
#endif

  for (int32_t iter = 0; iter < iterations; iter++)
  {
    test_pointcloud_voxelization(display_fn);
  }

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::spin(node);
  rclcpp::shutdown();
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ros::spin();
#endif
  return 0;
}
