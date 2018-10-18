#include <common_robotics_utilities/print.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/ros_interface.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <functional>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/color_builder.hpp>

int main(int argc, char** argv)
{
    // Make a ROS node, which we'll use to publish copies of the data in the CollisionMap and SDF
    // and Rviz markers that allow us to visualize them.
    ros::init(argc, argv, "sdf_tools_tutorial");
    // Get a handle to the current node
    ros::NodeHandle nh;
    // Make a publisher for visualization messages
    ros::Publisher visualization_pub = nh.advertise<visualization_msgs::Marker>("sdf_tools_tutorial_visualization", 1, true);
    // In preparation, we want to set a couple common paramters
    const double resolution = 0.25;
    const double x_size = 10.0;
    const double y_size = 10.0;
    const double z_size = 10.0;
    const common_robotics_utilities::voxel_grid::GridSizes grid_sizes(resolution, x_size, y_size, z_size);
    // Let's center the grid around the origin
    const Eigen::Translation3d origin_translation(-5.0, -5.0, -5.0);
    const Eigen::Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
    const Eigen::Isometry3d origin_transform = origin_translation * origin_rotation;
    const std::string frame = "tutorial_frame";
    ///////////////////////////////////
    //// Let's make a CollisionMap ////
    ///////////////////////////////////
    // We pick a reasonable default and out-of-bounds value
    // WE initialize it like this - the component value is automatically set to 0
    const voxelized_geometry_tools::CollisionCell default_cell(0.0);
    // First, let's make the container
    voxelized_geometry_tools::CollisionMap collision_map(origin_transform, frame, grid_sizes, default_cell);
    // Let's set some values
    // This is how you should iterate through the 3D grid's cells
    for (int64_t x_index = 0; x_index < collision_map.GetNumXCells(); x_index++)
    {
        for (int64_t y_index = 0; y_index < collision_map.GetNumYCells(); y_index++)
        {
            for (int64_t z_index = 0; z_index < collision_map.GetNumZCells(); z_index++)
            {
                // Let's make the bottom corner (low x, low y, low z) an object
                if ((x_index < (collision_map.GetNumXCells() / 2)) && (y_index < (collision_map.GetNumYCells() / 2)) && (z_index < (collision_map.GetNumZCells() / 2)))
                {
                    const voxelized_geometry_tools::CollisionCell obstacle_cell(1.0); // Occupancy values > 0.5 are obstacles
                    collision_map.SetValue(x_index, y_index, z_index, obstacle_cell);
                }
            }
        }
    }
    // We can also set by location
    const voxelized_geometry_tools::CollisionCell obstacle_cell(1.0); // Occupancy values > 0.5 are obstacles
    collision_map.SetValue(0.0, 0.0, 0.0, obstacle_cell);
    // Let's get some values
    // We can query by index
    int64_t x_index = 10;
    int64_t y_index = 10;
    int64_t z_index = 10;
    const auto index_query = collision_map.GetImmutable(x_index, y_index, z_index);
    // Is it in the grid?
    if (index_query)
    {
      std::cout << "Index query result - stored value " << index_query.Value().Occupancy() << " (occupancy) " << index_query.Value().Component() << " (component)" << std::endl;
    }
    // Or we can query by location
    double x_location = 0.0;
    double y_location = 0.0;
    double z_location = 0.0;
    const auto location_query = collision_map.GetImmutable(x_location, y_location, z_location);
    // Is it in the grid?
    if (location_query)
    {
      std::cout << "Location query result - stored value " << location_query.Value().Occupancy() << " (occupancy) " << location_query.Value().Component() << " (component)" << std::endl;
    }
    // Let's compute connected components
    const uint32_t num_connected_components = collision_map.UpdateConnectedComponents();
    std::cout << " There are " << num_connected_components << " connected components in the grid" << std::endl;
    // Let's display the results to Rviz
    // First, the CollisionMap itself
    // We need to provide colors to use
    std_msgs::ColorRGBA collision_color;
    collision_color.r = 1.0;
    collision_color.g = 0.0;
    collision_color.b = 0.0;
    collision_color.a = 0.5;
    std_msgs::ColorRGBA free_color;
    free_color.r = 0.0;
    free_color.g = 1.0;
    free_color.b = 0.0;
    free_color.a = 0.5;
    std_msgs::ColorRGBA unknown_color;
    unknown_color.r = 1.0;
    unknown_color.g = 1.0;
    unknown_color.b = 0.0;
    unknown_color.a = 0.5;
    visualization_msgs::Marker collision_map_marker = voxelized_geometry_tools::ros_interface::ExportForDisplay(collision_map, collision_color, free_color, unknown_color);
    // To be safe, you'll need to set these yourself. The namespace (ns) value should distinguish between different things being displayed
    // while the id value lets you have multiple versions of the same message at once. Always set this to 1 if you only want one copy.
    collision_map_marker.ns = "collision_map";
    collision_map_marker.id = 1;
    // Send it off for display
    visualization_pub.publish(collision_map_marker);
    // Now, let's draw the connected components
    visualization_msgs::Marker connected_components_marker = voxelized_geometry_tools::ros_interface::ExportConnectedComponentsForDisplay(collision_map, false); // Generally, you don't want a special color for unknown [P(occupancy) = 0.5] components
    connected_components_marker.ns = "connected_components";
    connected_components_marker.id = 1;
    visualization_pub.publish(connected_components_marker);
    ///////////////////////////
    //// Let's make an SDF ////
    ///////////////////////////
    // We pick a reasonable out-of-bounds value
    float oob_value = INFINITY;
    // We start by extracting the SDF from the CollisionMap
    const std::pair<voxelized_geometry_tools::SignedDistanceField<std::vector<float>>, std::pair<double, double>> sdf_with_extrema = collision_map.ExtractSignedDistanceField(oob_value, true, false, false);
    const voxelized_geometry_tools::SignedDistanceField<std::vector<float>>& sdf = sdf_with_extrema.first;
    const std::pair<double, double>& sdf_extrema = sdf_with_extrema.second;
    std::cout << "Maximum distance in the SDF: " << sdf_extrema.first << ", minimum distance in the SDF: " << sdf_extrema.second << std::endl;
    // Let's get some values
    const auto index_sdf_query = sdf.GetImmutable(x_index, y_index, z_index);
    // Is it in the grid?
    if (index_sdf_query)
    {
      std::cout << "Index query result - stored distance " << index_sdf_query.Value() << std::endl;
    }
    const auto location_sdf_query = sdf.GetImmutable(x_location, y_location, z_location);
    if (location_sdf_query)
    {
      std::cout << "Location query result - stored distance " << location_sdf_query.Value() << std::endl;
    }
    // Let's get some gradients
    const auto index_gradient_query = sdf.GetCoarseGradient(x_index, y_index, z_index, true); // Usually, you want to enable 'edge gradients' i.e. gradients for cells on the edge of the grid that don't have 6 neighbors
    if (index_gradient_query)
    {
        std::cout << "Index gradient query result - gradient " << common_robotics_utilities::print::Print(index_gradient_query.Value()) << std::endl;
    }
    const auto location_gradient_query = sdf.GetCoarseGradient(x_location, y_location, z_location, true); // Usually, you want to enable 'edge gradients' i.e. gradients for cells on the edge of the grid that don't have 6 neighbors
    if (location_gradient_query)
    {
        std::cout << "Location gradient query result - gradient " << common_robotics_utilities::print::Print(location_gradient_query.Value()) << std::endl;
    }
    // Let's display the results to Rviz
    visualization_msgs::Marker sdf_marker = voxelized_geometry_tools::ros_interface::ExportSDFForDisplay<std::vector<float>>(sdf, 0.5); // Set the alpha for display
    sdf_marker.ns = "sdf";
    sdf_marker.id = 1;
    visualization_pub.publish(sdf_marker);
    std::cout << "...done" << std::endl;
    return 0;
}
