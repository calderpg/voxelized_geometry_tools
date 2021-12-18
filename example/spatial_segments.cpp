#include <common_robotics_utilities/print.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/ros_interface.hpp>

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

#include <functional>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/color_builder.hpp>

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
using ColorRGBA = std_msgs::msg::ColorRGBA;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
using ColorRGBA = std_msgs::ColorRGBA;
using Marker = visualization_msgs::Marker;
using MarkerArray = visualization_msgs::MarkerArray;
#endif

void test_spatial_segments(
    const std::function<void(const MarkerArray&)>& display_fn)
{
  const double res = 1.0;
  const int64_t x_size = 100;
  const int64_t y_size = 100;
  const int64_t z_size = 50;
  const Eigen::Isometry3d origin_transform
      = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond(
          Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()));
  const common_robotics_utilities::voxel_grid::GridSizes grid_sizes(res, x_size, y_size, z_size);
  voxelized_geometry_tools::TaggedObjectCollisionMap tocmap(origin_transform, "world", grid_sizes, voxelized_geometry_tools::TaggedObjectCollisionCell(0.0, 0u));
  for (int64_t x_idx = 0; x_idx < tocmap.GetNumXCells(); x_idx++)
  {
    for (int64_t y_idx = 0; y_idx < tocmap.GetNumYCells(); y_idx++)
    {
      for (int64_t z_idx = 0; z_idx < tocmap.GetNumZCells(); z_idx++)
      {
        if ((x_idx < 10) || (y_idx < 10) || (x_idx >= tocmap.GetNumXCells() - 10) || (y_idx >= tocmap.GetNumYCells() - 10))
        {
          tocmap.SetValue(x_idx, y_idx, z_idx, voxelized_geometry_tools::TaggedObjectCollisionCell(1.0, 1u));
        }
        else if ((x_idx >= 40) && (y_idx >= 40) && (x_idx < 60) && (y_idx < 60))
        {
          tocmap.SetValue(x_idx, y_idx, z_idx, voxelized_geometry_tools::TaggedObjectCollisionCell(1.0, 2u));
        }
        if (((x_idx >= 45) && (x_idx < 55)) || ((y_idx >= 45) && (y_idx < 55)))
        {
          tocmap.SetValue(x_idx, y_idx, z_idx, voxelized_geometry_tools::TaggedObjectCollisionCell(0.0, 0u));
        }
      }
    }
  }
  MarkerArray display_markers;
  Marker env_marker = voxelized_geometry_tools::ros_interface::ExportForDisplay(tocmap);
  env_marker.id = 1;
  env_marker.ns = "environment";
  display_markers.markers.push_back(env_marker);
  Marker components_marker = voxelized_geometry_tools::ros_interface::ExportConnectedComponentsForDisplay(tocmap, false);
  components_marker.id = 1;
  components_marker.ns = "environment_components";
  display_markers.markers.push_back(components_marker);
  const double connected_threshold = 1.75;
  const uint32_t number_of_spatial_segments_manual_border = tocmap.UpdateSpatialSegments(connected_threshold, false, false);
  std::cout << "Identified " << number_of_spatial_segments_manual_border
            << " spatial segments via SDF->maxima map->connected components (no border added)"
            << std::endl;
  for (uint32_t object_id = 0u; object_id <= 4u; object_id++)
  {
    for (uint32_t spatial_segment = 1u; spatial_segment <= number_of_spatial_segments_manual_border; spatial_segment++)
    {
      Marker segment_marker = voxelized_geometry_tools::ros_interface::ExportSpatialSegmentForDisplay(tocmap, object_id, spatial_segment);
      if (segment_marker.points.size() > 0)
      {
        segment_marker.ns += "_no_border";
        display_markers.markers.push_back(segment_marker);
      }
    }
  }
  const uint32_t number_of_spatial_segments_virtual_border = tocmap.UpdateSpatialSegments(connected_threshold, true, false);
  std::cout << "Identified " << number_of_spatial_segments_virtual_border
            << " spatial segments via SDF->maxima map->connected components (virtual border added)"
            << std::endl;
  for (uint32_t object_id = 0u; object_id <= 4u; object_id++)
  {
    for (uint32_t spatial_segment = 1u; spatial_segment <= number_of_spatial_segments_virtual_border; spatial_segment++)
    {
      Marker segment_marker = voxelized_geometry_tools::ros_interface::ExportSpatialSegmentForDisplay(tocmap, object_id, spatial_segment);
      if (segment_marker.points.size() > 0)
      {
        segment_marker.ns += "_virtual_border";
        display_markers.markers.push_back(segment_marker);
      }
    }
  }
  const auto sdf_result
      = tocmap.ExtractSignedDistanceFieldFloat(std::vector<uint32_t>(), std::numeric_limits<float>::infinity(), true, false, false);
  const auto& sdf = sdf_result.DistanceField();
  Marker sdf_marker = voxelized_geometry_tools::ros_interface::ExportSDFForDisplay(sdf, 1.0f);
  sdf_marker.id = 1;
  sdf_marker.ns = "environment_sdf_no_border";
  display_markers.markers.push_back(sdf_marker);
  const auto virtual_border_sdf_result
      = tocmap.ExtractSignedDistanceFieldFloat(std::vector<uint32_t>(), std::numeric_limits<float>::infinity(), true, false, true);
  const auto& virtual_border_sdf = virtual_border_sdf_result.DistanceField();
  Marker virtual_border_sdf_marker = voxelized_geometry_tools::ros_interface::ExportSDFForDisplay(virtual_border_sdf, 1.0f);
  virtual_border_sdf_marker.id = 1;
  virtual_border_sdf_marker.ns = "environment_sdf_virtual_border";
  display_markers.markers.push_back(virtual_border_sdf_marker);
  // Make extrema markers
  const auto maxima_map = virtual_border_sdf.ComputeLocalExtremaMap();
  for (int64_t x_idx = 0; x_idx < maxima_map.GetNumXCells(); x_idx++)
  {
    for (int64_t y_idx = 0; y_idx < maxima_map.GetNumYCells(); y_idx++)
    {
      for (int64_t z_idx = 0; z_idx < maxima_map.GetNumZCells(); z_idx++)
      {
        const Eigen::Vector4d location
            = maxima_map.GridIndexToLocation(x_idx, y_idx, z_idx);
        const Eigen::Vector3d extrema = maxima_map.GetImmutable(x_idx, y_idx, z_idx).Value();
        if (!std::isinf(extrema.x())
            && !std::isinf(extrema.y())
            && !std::isinf(extrema.z()))
        {
          const double distance = (extrema - location.block<3, 1>(0, 0)).norm();
          if (distance < sdf.GetResolution())
          {
            Marker maxima_rep;
            // Populate the header
            maxima_rep.header.frame_id = "world";
            // Populate the options
            maxima_rep.ns = "extrema";
            maxima_rep.id = static_cast<int32_t>(sdf.HashDataIndex(x_idx, y_idx, z_idx));
            maxima_rep.action = Marker::ADD;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
            maxima_rep.lifetime = rclcpp::Duration(0, 0);
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
            maxima_rep.lifetime = ros::Duration(0.0);
#endif
            maxima_rep.frame_locked = false;
            maxima_rep.pose.position = common_robotics_utilities::ros_conversions::EigenVector4dToGeometryPoint(location);
            maxima_rep.pose.orientation = common_robotics_utilities::ros_conversions::EigenQuaterniondToGeometryQuaternion(Eigen::Quaterniond::Identity());
            maxima_rep.type = Marker::SPHERE;
            maxima_rep.scale.x = sdf.GetResolution();
            maxima_rep.scale.y = sdf.GetResolution();
            maxima_rep.scale.z = sdf.GetResolution();
            maxima_rep.color = common_robotics_utilities::color_builder::MakeFromFloatColors<ColorRGBA>(1.0, 0.5, 0.0, 1.0);
            display_markers.markers.push_back(maxima_rep);
          }
        }
        else
        {
          std::cout << "Encountered inf extrema @ (" << x_idx << "," << y_idx << "," << z_idx << ")" << std::endl;
        }
      }
    }
  }
  std::cout << "(0,0,0) " << common_robotics_utilities::print::Print(virtual_border_sdf.GetCoarseGradient(static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0), true).Value()) << std::endl;
  std::cout << "(1,1,1) " << common_robotics_utilities::print::Print(virtual_border_sdf.GetCoarseGradient(static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1), true).Value()) << std::endl;
  std::cout << "(2,2,2) " << common_robotics_utilities::print::Print(virtual_border_sdf.GetCoarseGradient(static_cast<int64_t>(2), static_cast<int64_t>(2), static_cast<int64_t>(2), true).Value()) << std::endl;
  std::cout << "(0,0,0) " << common_robotics_utilities::print::Print(virtual_border_sdf.GetFineGradient(static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0), res).Value()) << std::endl;
  std::cout << "(1,1,1) " << common_robotics_utilities::print::Print(virtual_border_sdf.GetFineGradient(static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1), res).Value()) << std::endl;
  std::cout << "(2,2,2) " << common_robotics_utilities::print::Print(virtual_border_sdf.GetFineGradient(static_cast<int64_t>(2), static_cast<int64_t>(2), static_cast<int64_t>(2), res).Value()) << std::endl;
  std::cout << "(0,0,0) " << common_robotics_utilities::print::Print(maxima_map.GetImmutable(static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(0)).Value()) << std::endl;
  std::cout << "(1,1,1) " << common_robotics_utilities::print::Print(maxima_map.GetImmutable(static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1)).Value()) << std::endl;
  std::cout << "(2,2,2) " << common_robotics_utilities::print::Print(maxima_map.GetImmutable(static_cast<int64_t>(2), static_cast<int64_t>(2), static_cast<int64_t>(2)).Value()) << std::endl;
  display_fn(display_markers);
}

int main(int argc, char** argv)
{
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("compute_spatial_segments_test");
  auto display_pub = node->create_publisher<MarkerArray>(
      "display_test_voxel_grid", rclcpp::QoS(1).transient_local());
  const std::function<void(const MarkerArray&)>& display_fn
      = [&] (const MarkerArray& markers)
  {
    display_pub->publish(markers);
  };
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ros::init(argc, argv, "compute_spatial_segments_test");
  ros::NodeHandle nh;
  ros::Publisher display_pub = nh.advertise<MarkerArray>(
      "display_test_voxel_grid", 1, true);
  const std::function<void(const MarkerArray&)>& display_fn
      = [&] (const MarkerArray& markers)
  {
    display_pub.publish(markers);
  };
#endif

  test_spatial_segments(display_fn);

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::spin(node);
  rclcpp::shutdown();
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ros::spin();
#endif
  return 0;
}
