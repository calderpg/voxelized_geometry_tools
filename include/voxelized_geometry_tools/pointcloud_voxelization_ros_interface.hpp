#pragma once

#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
#include <sensor_msgs/msg/point_cloud2.hpp>
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
#include <sensor_msgs/PointCloud2.h>
#else
#error "Undefined or unknown VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION"
#endif
#include <voxelized_geometry_tools/pointcloud_voxelization_interface.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{

#if VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 2
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointCloud2ConstSharedPtr = std::shared_ptr<const PointCloud2>;
#elif VOXELIZED_GEOMETRY_TOOLS__SUPPORTED_ROS_VERSION == 1
using PointCloud2 = sensor_msgs::PointCloud2;
using PointCloud2ConstSharedPtr = sensor_msgs::PointCloud2ConstPtr;
#endif

class PointCloud2Wrapper : public PointCloudWrapper
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double MaxRange() const override { return max_range_; }

  int64_t Size() const override
  {
    return static_cast<int64_t>(cloud_ptr_->width * cloud_ptr_->height);
  }

  const Eigen::Isometry3d& GetPointCloudOriginTransform() const override
  {
    return origin_transform_;
  }

  void SetPointCloudOriginTransform(
      const Eigen::Isometry3d& origin_transform) override
  {
    origin_transform_ = origin_transform;
  }

protected:
  PointCloud2Wrapper(
      const PointCloud2* const cloud_ptr,
      const Eigen::Isometry3d& origin_transform, const double max_range);

private:
  void CopyPointLocationIntoDoublePtrImpl(
      const int64_t point_index, double* destination) const override
  {
    const Eigen::Vector4d point =
        GetPointLocationVector4f(point_index).cast<double>();
    std::memcpy(destination, point.data(), sizeof(double) * 3);
  }

  void CopyPointLocationIntoFloatPtrImpl(
      const int64_t point_index, float* destination) const override
  {
    const size_t starting_offset = GetStartingOffsetForPointXYZ(point_index);
    std::memcpy(destination, &(cloud_ptr_->data.at(starting_offset)),
                sizeof(float) * 3);
  }

  size_t GetStartingOffsetForPointXYZ(const int64_t point_index) const
  {
    const size_t starting_offset =
        (static_cast<size_t>(point_index)
         * static_cast<size_t>(cloud_ptr_->point_step))
        + xyz_offset_from_point_start_;
    return starting_offset;
  }

  const PointCloud2* const cloud_ptr_ = nullptr;
  size_t xyz_offset_from_point_start_ = 0;
  Eigen::Isometry3d origin_transform_ = Eigen::Isometry3d::Identity();
  double max_range_ = std::numeric_limits<double>::infinity();
};

class NonOwningPointCloud2Wrapper : public PointCloud2Wrapper
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  NonOwningPointCloud2Wrapper(
      const PointCloud2* const cloud_ptr,
      const Eigen::Isometry3d& origin_transform,
      const double max_range = std::numeric_limits<double>::infinity())
      : PointCloud2Wrapper(cloud_ptr, origin_transform, max_range) {}
};

class OwningPointCloud2Wrapper : public PointCloud2Wrapper
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OwningPointCloud2Wrapper(
      const PointCloud2ConstSharedPtr& cloud_ptr,
      const Eigen::Isometry3d& origin_transform,
      const double max_range = std::numeric_limits<double>::infinity())
      : PointCloud2Wrapper(cloud_ptr.get(), origin_transform, max_range),
        owned_cloud_ptr_(cloud_ptr) {}

private:
  PointCloud2ConstSharedPtr owned_cloud_ptr_;
};
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
