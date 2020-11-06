#include <voxelized_geometry_tools/pointcloud_voxelization_ros_interface.hpp>

#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
#include <sensor_msgs/msg/point_field.hpp>
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <sensor_msgs/PointField.h>
#endif

#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
using PointField = sensor_msgs::msg::PointField;
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
using PointField = sensor_msgs::PointField;
#endif

PointCloud2Wrapper::PointCloud2Wrapper(
    const PointCloud2* const cloud_ptr,
    const Eigen::Isometry3d& origin_transform, const double max_range)
    : cloud_ptr_(cloud_ptr), origin_transform_(origin_transform),
      max_range_(max_range)
{
  if (cloud_ptr_ == nullptr)
  {
    throw std::invalid_argument("cloud_ptr_ == nullptr");
  }
  if (max_range_ <= 0.0)
  {
    throw std::runtime_error("max_range_ <= 0.0");
  }
  // Figure out what the size and offset for XYZ fields in the pointcloud are.
  std::map<std::string, uint8_t> field_type_map;
  std::map<std::string, size_t> field_offset_map;
  for (const auto& field : cloud_ptr_->fields)
  {
    field_type_map[field.name] = field.datatype;
    field_offset_map[field.name] = static_cast<size_t>(field.offset);
  }
  // Check field types
  if (field_type_map.at("x") != PointField::FLOAT32)
  {
    throw std::invalid_argument("PointCloud x field is not FLOAT32");
  }
  if (field_type_map.at("y") != PointField::FLOAT32)
  {
    throw std::invalid_argument("PointCloud y field is not FLOAT32");
  }
  if (field_type_map.at("z") != PointField::FLOAT32)
  {
    throw std::invalid_argument("PointCloud z field is not FLOAT32");
  }
  // Check that x, y, and z are sequential.
  const size_t x_offset = field_offset_map.at("x");
  const size_t y_offset = field_offset_map.at("y");
  const size_t z_offset = field_offset_map.at("z");
  if ((z_offset - y_offset) == sizeof(float) &&
      (y_offset - x_offset) == sizeof(float))
  {
    xyz_offset_from_point_start_ = x_offset;
  }
  else
  {
    throw std::invalid_argument(
        "PointCloud does not have sequential xyz fields");
  }
}
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
