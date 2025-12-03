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

#include <voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp>
#include <voxelized_geometry_tools/ros_interface.hpp>

using common_robotics_utilities::color_builder::MakeFromFloatColors;
using common_robotics_utilities::math::Interpolate;
using common_robotics_utilities::utility::UniformUnitRealFunction;
using common_robotics_utilities::voxel_grid::GridIndex;
using common_robotics_utilities::voxel_grid::Vector3i64;
using common_robotics_utilities::voxel_grid::VoxelGridSizes;
using voxelized_geometry_tools::pointcloud_voxelization
    ::CpuPointCloudVoxelizer;
using voxelized_geometry_tools::ros_interface::ExportVoxelGridToRViz;

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
using ColorRGBA = std_msgs::msg::ColorRGBA;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using Point = geometry_msgs::msg::Point;
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
using ColorRGBA = std_msgs::ColorRGBA;
using Marker = visualization_msgs::Marker;
using MarkerArray = visualization_msgs::MarkerArray;
using Point = geometry_msgs::Point;
#endif

namespace
{
using DisplayFunction = std::function<void(const MarkerArray&)>;

void RunRaycastCycle(
    const UniformUnitRealFunction& uniform_unit_real_fn,
    const DisplayFunction& display_fn)
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

  const auto color_fn = [](
      const CpuPointCloudVoxelizer::CpuVoxelizationTrackingCell& cell,
      const GridIndex& index) -> ColorRGBA
  {
    const int32_t seen_free_count = cell.seen_free_count.load();
    const int32_t seen_filled_count = cell.seen_filled_count.load();

    if (seen_free_count > 1)
    {
      std::cout << "WARNING: index " << index << " was counted free " << seen_free_count << " times" << std::endl;
    }
    if (seen_filled_count > 1)
    {
      std::cout << "WARNING: index " << index << " was counted filled " << seen_filled_count << " times" << std::endl;
    }
    if (seen_free_count > 0 && seen_filled_count > 0)
    {
      std::cout << "WARNING: index " << index << " was counted both free and filled" << std::endl;
    }

    if (seen_free_count > 0 && seen_filled_count > 0)
    {
      return MakeFromFloatColors<ColorRGBA>(1.0f, 0.0f, 1.0f, 0.5f);
    }
    else if (seen_free_count > 0)
    {
      return MakeFromFloatColors<ColorRGBA>(0.0f, 0.0f, 1.0f, 0.5f);
    }
    else if (seen_filled_count > 0)
    {
      return MakeFromFloatColors<ColorRGBA>(1.0f, 0.0f, 0.0f, 0.5f);
    }
    else
    {
      return MakeFromFloatColors<ColorRGBA>(0.0f, 0.0f, 0.0f, 0.0f);
    }
  };

  auto grid_marker = ExportVoxelGridToRViz
      <CpuPointCloudVoxelizer::CpuVoxelizationTrackingCell>(
          raycast_grid, "world", color_fn);
  grid_marker.ns = "raycast_grid";
  grid_marker.id = 1;

  Marker grid_box_marker;
  grid_box_marker.header.frame_id = "world";
  grid_box_marker.ns = "grid_box";
  grid_box_marker.id = 1;
  grid_box_marker.type = Marker::CUBE;
  grid_box_marker.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  grid_box_marker.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  grid_box_marker.lifetime = ros::Duration(0.0);
#endif
  grid_box_marker.frame_locked = true;
  grid_box_marker.scale.x = 5.0;
  grid_box_marker.scale.y = 5.0;
  grid_box_marker.scale.z = 5.0;
  grid_box_marker.pose.position.x = 2.5;
  grid_box_marker.pose.position.y = 2.5;
  grid_box_marker.pose.position.z = 2.5;
  grid_box_marker.pose.orientation.w = 1.0;
  grid_box_marker.pose.orientation.x = 0.0;
  grid_box_marker.pose.orientation.y = 0.0;
  grid_box_marker.pose.orientation.z = 0.0;
  grid_box_marker.color.r = 0.0;
  grid_box_marker.color.g = 1.0;
  grid_box_marker.color.b = 1.0;
  grid_box_marker.color.a = 0.125f;

  Marker origin_marker;
  origin_marker.header.frame_id = "world";
  origin_marker.ns = "origin";
  origin_marker.id = 1;
  origin_marker.type = Marker::SPHERE;
  origin_marker.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  origin_marker.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  origin_marker.lifetime = ros::Duration(0.0);
#endif
  origin_marker.frame_locked = true;
  origin_marker.scale.x = 0.05;
  origin_marker.scale.y = 0.05;
  origin_marker.scale.z = 0.05;
  origin_marker.pose.position.x = origin(0);
  origin_marker.pose.position.y = origin(1);
  origin_marker.pose.position.z = origin(2);
  origin_marker.pose.orientation.w = 1.0;
  origin_marker.pose.orientation.x = 0.0;
  origin_marker.pose.orientation.y = 0.0;
  origin_marker.pose.orientation.z = 0.0;
  origin_marker.color.r = 1.0;
  origin_marker.color.g = 0.0;
  origin_marker.color.b = 0.0;
  origin_marker.color.a = 1.0;

  Marker point_marker;
  point_marker.header.frame_id = "world";
  point_marker.ns = "point";
  point_marker.id = 1;
  point_marker.type = Marker::SPHERE;
  point_marker.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  point_marker.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  point_marker.lifetime = ros::Duration(0.0);
#endif
  point_marker.frame_locked = true;
  point_marker.scale.x = 0.05;
  point_marker.scale.y = 0.05;
  point_marker.scale.z = 0.05;
  point_marker.pose.position.x = point(0);
  point_marker.pose.position.y = point(1);
  point_marker.pose.position.z = point(2);
  point_marker.pose.orientation.w = 1.0;
  point_marker.pose.orientation.x = 0.0;
  point_marker.pose.orientation.y = 0.0;
  point_marker.pose.orientation.z = 0.0;
  point_marker.color.r = 0.0;
  point_marker.color.g = 1.0;
  point_marker.color.b = 0.0;
  point_marker.color.a = 1.0;

  const auto make_geom_point = [](const Eigen::Vector4d& p)
  {
    Point point_msg;
    point_msg.x = p(0);
    point_msg.y = p(1);
    point_msg.z = p(2);
    return point_msg;
  };

  Marker ray_marker;
  ray_marker.header.frame_id = "world";
  ray_marker.ns = "ray";
  ray_marker.id = 1;
  ray_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  ray_marker.action = visualization_msgs::msg::Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  ray_marker.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ray_marker.lifetime = ros::Duration(0.0);
#endif
  ray_marker.frame_locked = true;
  ray_marker.scale.x = 0.01;
  ray_marker.scale.y = 0.0;
  ray_marker.scale.z = 0.0;
  ray_marker.pose.position.x = 0.0;
  ray_marker.pose.position.y = 0.0;
  ray_marker.pose.position.z = 0.0;
  ray_marker.pose.orientation.w = 1.0;
  ray_marker.pose.orientation.x = 0.0;
  ray_marker.pose.orientation.y = 0.0;
  ray_marker.pose.orientation.z = 0.0;
  ray_marker.color.r = 0.0;
  ray_marker.color.g = 0.0;
  ray_marker.color.b = 0.0;
  ray_marker.color.a = 1.0;
  ray_marker.points.push_back(make_geom_point(origin));
  ray_marker.points.push_back(make_geom_point(point));

  MarkerArray markers;
  markers.markers =
      {grid_marker, grid_box_marker, origin_marker, point_marker, ray_marker};

  display_fn(markers);

  std::cout << "Press ENTER to continue..." << std::endl;
  std::cin.get();
}
}  // namespace

int main(int argc, char** argv)
{
  constexpr int32_t default_iterations = 100;

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("voxel_raycasting");
  auto display_pub = node->create_publisher<MarkerArray>(
      "voxel_raycasting", rclcpp::QoS(1).transient_local());
  const DisplayFunction display_fn = [&](const MarkerArray& markers)
  {
    display_pub->publish(markers);
  };
  const int32_t iterations =
      static_cast<int32_t>(node->declare_parameter(
          "iterations",
          rclcpp::ParameterValue(default_iterations)).get<int32_t>());
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ros::init(argc, argv, "voxel_raycasting");
  ros::NodeHandle nh;
  ros::NodeHandle nhp("~");
  ros::Publisher display_pub = nh.advertise<MarkerArray>(
      "voxel_raycasting", 1, true);
  const DisplayFunction display_fn = [&](const MarkerArray& markers)
  {
    display_pub.publish(markers);
  };
  const int32_t iterations =
      nhp.param(std::string("iterations"), default_iterations);
#endif

  std::mt19937_64 prng(42);

  const UniformUnitRealFunction uniform_unit_real_fn = [&]()
  {
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(
        prng);
  };

  for (int32_t iter = 0; iter < iterations; iter++)
  {
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
    if (rclcpp::ok())
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
    if (ros::ok())
#endif
    {
      RunRaycastCycle(uniform_unit_real_fn, display_fn);

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
      rclcpp::spin_some(node);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
      ros::spin();
#endif
    }
  }

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::shutdown();
#endif
  return 0;
}
