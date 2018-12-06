#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <voxelized_geometry_tools/collision_map.hpp>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
enum class SeenAs : uint8_t {UNKNOWN = 0x00, FILLED = 0x01, FREE = 0x02};

class PointCloudVoxelizationFilterOptions
{
public:
  PointCloudVoxelizationFilterOptions(
      const double percent_seen_free, const int32_t outlier_points_threshold,
      const int32_t num_cameras_seen_free)
      : percent_seen_free_(percent_seen_free),
        outlier_points_threshold_(outlier_points_threshold),
        num_cameras_seen_free_(num_cameras_seen_free)
  {
    if (percent_seen_free_ <= 0.0 || percent_seen_free_ > 1.0)
    {
      throw std::invalid_argument("0 < percent_seen_free_ <= 1 must be true");
    }
    if (outlier_points_threshold_ <= 0)
    {
      throw std::invalid_argument("outlier_points_threshold_ <= 0");
    }
    if (num_cameras_seen_free_ <= 0)
    {
      throw std::invalid_argument("num_cameras_seen_free_ <= 0");
    }
  }

  PointCloudVoxelizationFilterOptions()
      : percent_seen_free_(1.0),
        outlier_points_threshold_(1),
        num_cameras_seen_free_(1) {}

  double PercentSeenFree() const { return percent_seen_free_; }

  int32_t OutlierPointsThreshold() const { return outlier_points_threshold_; }

  int32_t NumCamerasSeenFree() const { return num_cameras_seen_free_; }

  SeenAs CountsSeenAs(
      const int32_t seen_free_count, const int32_t seen_filled_count) const
  {
    const int32_t filtered_seen_filled_count =
        (seen_filled_count >= OutlierPointsThreshold()) ? seen_filled_count : 0;
    if (seen_free_count > 0 && filtered_seen_filled_count > 0)
    {
      const double percent_seen_free =
          static_cast<double>(seen_free_count)
          / static_cast<double>(seen_free_count + filtered_seen_filled_count);
      if (percent_seen_free >= PercentSeenFree())
      {
        return SeenAs::FREE;
      }
      else
      {
        return SeenAs::FILLED;
      }
    }
    else if (seen_free_count > 0)
    {
      return SeenAs::FREE;
    }
    else if (filtered_seen_filled_count > 0)
    {
      return SeenAs::FILLED;
    }
    else
    {
      return SeenAs::UNKNOWN;
    }
  }

private:
  double percent_seen_free_ = 1.0;
  int32_t outlier_points_threshold_ = 1;
  int32_t num_cameras_seen_free_ = 1;
};

class PointCloudWrapper
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual ~PointCloudWrapper() {}

  virtual int64_t Size() const = 0;

  virtual const Eigen::Isometry3d& GetPointCloudOriginTransform() const = 0;

  virtual Eigen::Vector4d GetPointLocationDouble(
      const int64_t point_index) const = 0;

  virtual Eigen::Vector4f GetPointLocationFloat(
      const int64_t point_index) const = 0;

  virtual void CopyPointLocationIntoVectorDouble(
      const int64_t point_index, std::vector<double>& vector,
      const int64_t vector_index) const = 0;

  virtual void CopyPointLocationIntoVectorFloat(
      const int64_t point_index, std::vector<float>& vector,
      const int64_t vector_index) const = 0;
};
using PointCloudWrapperPtr = std::shared_ptr<PointCloudWrapper>;

class PointCloudVoxelizationInterface
{
public:
  virtual ~PointCloudVoxelizationInterface() {}

  virtual CollisionMap VoxelizePointClouds(
      const CollisionMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperPtr>& pointclouds) const = 0;
};
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
