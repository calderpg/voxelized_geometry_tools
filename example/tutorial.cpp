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
#include <iostream>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/color_builder.hpp>

int main(int argc, char** argv)
{
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  using ColorRGBA = std_msgs::msg::ColorRGBA;
  using Marker = visualization_msgs::msg::Marker;

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("sdf_tools_tutorial");
  auto visualization_pub = node->create_publisher<Marker>(
      "sdf_tools_tutorial_visualization", rclcpp::QoS(1).transient_local());
  const std::function<void(const Marker&)> display_fn
    = [&] (const Marker& marker)
  {
    visualization_pub->publish(marker);
  };
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  using ColorRGBA = std_msgs::ColorRGBA;
  using Marker = visualization_msgs::Marker;

  // Make a ROS node, which we'll use to publish copies of the data in the
  // CollisionMap and SDF and Rviz markers that allow us to visualize them.
  ros::init(argc, argv, "sdf_tools_tutorial");
  // Get a handle to the current node
  ros::NodeHandle nh;
  // Make a publisher for visualization messages
  ros::Publisher visualization_pub = nh.advertise<Marker>(
      "sdf_tools_tutorial_visualization", 1, true);
  const std::function<void(const Marker&)> display_fn
    = [&] (const Marker& marker)
  {
    visualization_pub.publish(marker);
  };
#endif

  // In preparation, we want to set a couple common paramters
  const double resolution = 0.25; //2.5; //0.25;
  const double x_size = 10.0;
  const double y_size = 10.0;
  const double z_size = 10.0;
  const common_robotics_utilities::voxel_grid::GridSizes grid_sizes(
      resolution, x_size, y_size, z_size);
  // Let's center the grid around the origin
  const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
  const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
  const Eigen::Isometry3d origin_transform =
      origin_translation * origin_rotation;
  const std::string frame = "tutorial_frame";

  ///////////////////////////////////
  //// Let's make a CollisionMap ////
  ///////////////////////////////////

  // We pick a reasonable default and out-of-bounds value
  // By initializing like this, the component value is automatically set to 0.
  const voxelized_geometry_tools::CollisionCell default_cell(0.0);
  // First, let's make the container
  voxelized_geometry_tools::CollisionMap collision_map(
      origin_transform, frame, grid_sizes, default_cell);

  // Let's set some values
  // This is how you should iterate through the 3D grid's cells
  for (int64_t x_index = 0; x_index < collision_map.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < collision_map.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < collision_map.GetNumZCells();
           z_index++)
      {
        // Let's make the bottom corner (low x, low y, low z) an object
        if ((x_index < (collision_map.GetNumXCells() / 2)) &&
            (y_index < (collision_map.GetNumYCells() / 2)) &&
            (z_index < (collision_map.GetNumZCells() / 2)))
        {
          // Occupancy values > 0.5 are obstacles
          const voxelized_geometry_tools::CollisionCell obstacle_cell(1.0);
          collision_map.SetIndex(x_index, y_index, z_index, obstacle_cell);
        }
      }
    }
  }

  // We can also set by location - occupancy values > 0.5 are obstacles
  const voxelized_geometry_tools::CollisionCell obstacle_cell(1.0);
  collision_map.SetLocation(0.0, 0.0, 0.0, obstacle_cell);

  // Let's get some values
  // We can query by index
  int64_t x_index = 10;
  int64_t y_index = 10;
  int64_t z_index = 10;
  const auto index_query =
      collision_map.GetIndexImmutable(x_index, y_index, z_index);

  // Is it in the grid?
  if (index_query)
  {
    std::cout << "Index query result - stored value "
              << index_query.Value().Occupancy() << " (occupancy) "
              << index_query.Value().Component() << " (component)" << std::endl;
  }

  // Or we can query by location
  double x_location = 0.0;
  double y_location = 0.0;
  double z_location = 0.0;
  const auto location_query =
      collision_map.GetLocationImmutable(x_location, y_location, z_location);

  // Is it in the grid?
  if (location_query)
  {
    std::cout << "Location query result - stored value "
              << location_query.Value().Occupancy() << " (occupancy) "
              << location_query.Value().Component() << " (component)"
              << std::endl;
  }

  // Let's compute connected components
  const uint32_t num_connected_components =
      collision_map.UpdateConnectedComponents();
  std::cout << " There are " << num_connected_components
            << " connected components in the grid" << std::endl;

  // Let's display the results to Rviz
  // First, the CollisionMap itself
  // We need to provide colors to use
  ColorRGBA collision_color;
  collision_color.r = 1.0;
  collision_color.g = 0.0;
  collision_color.b = 0.0;
  collision_color.a = 0.5;
  ColorRGBA free_color;
  free_color.r = 0.0;
  free_color.g = 1.0;
  free_color.b = 0.0;
  free_color.a = 0.5;
  ColorRGBA unknown_color;
  unknown_color.r = 1.0;
  unknown_color.g = 1.0;
  unknown_color.b = 0.0;
  unknown_color.a = 0.5;
  Marker collision_map_marker =
      voxelized_geometry_tools::ros_interface::ExportForDisplay(
          collision_map, collision_color, free_color, unknown_color);
  // To be safe, you'll need to set these yourself. The namespace (ns) value
  // should distinguish between different things being displayed while the id
  // value lets you have multiple versions of the same message at once. Always
  // set this to 1 if you only want one copy.
  collision_map_marker.ns = "collision_map";
  collision_map_marker.id = 1;
  // Send it off for display
  display_fn(collision_map_marker);
  // Now, let's draw the connected components
  // Generally, you don't want a special color for unknown [P(occupancy) = 0.5]
  // components.
  Marker connected_components_marker =
      voxelized_geometry_tools::ros_interface
          ::ExportConnectedComponentsForDisplay(collision_map, false);
  connected_components_marker.ns = "connected_components";
  connected_components_marker.id = 1;
  display_fn(connected_components_marker);

  ///////////////////////////
  //// Let's make an SDF ////
  ///////////////////////////

  // We pick a reasonable out-of-bounds value
  const float oob_value = std::numeric_limits<float>::infinity();
  // Disable parallelism in SDF generation
  const auto parallelism =
      common_robotics_utilities::parallelism::DegreeOfParallelism(4); //::None();
  // Use the "bucket queue" strategy
  const auto strategy =
      voxelized_geometry_tools::SignedDistanceFieldGenerationParameters<float>
          ::GenerationStrategy::EDT;
  // Treat cells with unknown occupancy as if they were filled
  const bool unknown_is_filled = true;
  // Don't add a virtual border
  const bool add_virtual_border = false;
  const voxelized_geometry_tools::SignedDistanceFieldGenerationParameters<float>
      sdf_parameters(oob_value, parallelism, strategy, unknown_is_filled,
                     add_virtual_border);
  // We start by extracting the SDF from the CollisionMap
  const auto sdf =
      collision_map.ExtractSignedDistanceFieldFloat(sdf_parameters);
  const auto sdf_extrema = sdf.GetMinimumMaximum();
  std::cout << "Maximum distance in the SDF: " << sdf_extrema.Maximum()
            << ", minimum distance in the SDF: " << sdf_extrema.Minimum()
            << std::endl;

  // Let's get some values
  const auto index_sdf_query = sdf.GetIndexImmutable(x_index, y_index, z_index);
  // Is it in the grid?
  if (index_sdf_query)
  {
    std::cout << "Index query result - stored distance "
              << index_sdf_query.Value() << std::endl;
  }
  const auto location_sdf_query =
      sdf.GetLocationImmutable(x_location, y_location, z_location);
  if (location_sdf_query)
  {
    std::cout << "Location query result - stored distance "
              << location_sdf_query.Value() << std::endl;
  }

  // Let's get some gradients
  // Usually, you want to enable 'edge gradients' i.e. gradients for cells on
  // the edge of the grid that don't have 6 neighbors.
  const auto index_gradient_query =
      sdf.GetIndexCoarseGradient(x_index, y_index, z_index, true);
  if (index_gradient_query)
  {
    std::cout << "Index gradient query result - gradient "
              << common_robotics_utilities::print::Print(
                  index_gradient_query.Value())
              << std::endl;
  }
  const auto location_gradient_query =
      sdf.GetLocationCoarseGradient(x_location, y_location, z_location, true);
  if (location_gradient_query)
  {
    std::cout << "Location gradient query result - gradient "
              << common_robotics_utilities::print::Print(
                  location_gradient_query.Value())
              << std::endl;
  }

  // Let's display the results to Rviz
  Marker sdf_marker =
      voxelized_geometry_tools::ros_interface::ExportSDFForDisplay(sdf, 0.5);
  sdf_marker.ns = "sdf";
  sdf_marker.id = 1;
  display_fn(sdf_marker);
  std::cout << "...done" << std::endl;
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
  rclcpp::spin(node);
  rclcpp::shutdown();
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
  ros::spin();
#endif
  return 0;
}
