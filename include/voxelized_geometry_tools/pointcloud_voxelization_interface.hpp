#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <voxelized_geometry_tools/occupancy_map.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
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
  explicit PointCloudWrapper(const PointCloudWrapper&) = delete;
  explicit PointCloudWrapper(PointCloudWrapper&&) = delete;
  PointCloudWrapper& operator=(const PointCloudWrapper&) = delete;
  PointCloudWrapper& operator=(PointCloudWrapper&&) = delete;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual ~PointCloudWrapper() {}

  virtual double MaxRange() const = 0;

  virtual int64_t Size() const = 0;

  virtual const Eigen::Isometry3d& PointCloudOriginTransform() const = 0;

  virtual void SetPointCloudOriginTransform(
      const Eigen::Isometry3d& origin_transform) = 0;

  Eigen::Vector4d GetPointLocationVector4d(const int64_t point_index) const
  {
    Eigen::Vector4d point(0.0, 0.0, 0.0, 1.0);
    CopyPointLocationIntoVector4d(point_index, point);
    return point;
  }

  Eigen::Vector4f GetPointLocationVector4f(const int64_t point_index) const
  {
    Eigen::Vector4f point(0.0f, 0.0f, 0.0f, 1.0f);
    CopyPointLocationIntoVector4f(point_index, point);
    return point;
  }

  void CopyPointLocationIntoVector4d(
      const int64_t point_index, Eigen::Vector4d& point_location) const
  {
    CopyPointLocationIntoDoublePtr(point_index, point_location.data());
    point_location(3) = 1.0;
  }

  void CopyPointLocationIntoVector4f(
      const int64_t point_index, Eigen::Vector4f& point_location) const
  {
    CopyPointLocationIntoFloatPtr(point_index, point_location.data());
    point_location(3) = 1.0f;
  }

  void CopyPointLocationIntoVectorDouble(
      const int64_t point_index, std::vector<double>& vector,
      const int64_t vector_index) const
  {
    EnforceVectorIndexInRange(vector_index, vector);
    CopyPointLocationIntoDoublePtr(point_index, vector.data() + vector_index);
  }

  void CopyPointLocationIntoVectorFloat(
      const int64_t point_index, std::vector<float>& vector,
      const int64_t vector_index) const
  {
    EnforceVectorIndexInRange(vector_index, vector);
    CopyPointLocationIntoFloatPtr(point_index, vector.data() + vector_index);
  }

  void CopyPointLocationIntoDoublePtr(
      const int64_t point_index, double* destination) const
  {
    EnforcePointIndexInRange(point_index);
    CopyPointLocationIntoDoublePtrImpl(point_index, destination);
  }

  void CopyPointLocationIntoFloatPtr(
      const int64_t point_index, float* destination) const
  {
    EnforcePointIndexInRange(point_index);
    CopyPointLocationIntoFloatPtrImpl(point_index, destination);
  }

  void EnforcePointIndexInRange(const int64_t point_index) const
  {
    if (point_index < 0 || point_index >= Size())
    {
      throw std::out_of_range("point_index out of range");
    }
  }

  template<typename T>
  static void EnforceVectorIndexInRange(
      const int64_t vector_index, const std::vector<T>& destination)
  {
    if (vector_index < 0 ||
        (vector_index + 3) > static_cast<int64_t>(destination.size()))
    {
      throw std::out_of_range("vector_index out of range");
    }
  }

protected:
  virtual void CopyPointLocationIntoDoublePtrImpl(
      const int64_t point_index, double* destination) const = 0;

  virtual void CopyPointLocationIntoFloatPtrImpl(
      const int64_t point_index, float* destination) const = 0;

  PointCloudWrapper() = default;
};

using PointCloudWrapperSharedPtr = std::shared_ptr<PointCloudWrapper>;

class VoxelizerRuntime
{
public:
  VoxelizerRuntime(const double raycasting_time, const double filtering_time)
      : raycasting_time_(raycasting_time), filtering_time_(filtering_time)
  {
    if (raycasting_time_ < 0.0)
    {
      throw std::invalid_argument("raycasting_time < 0.0");
    }
    if (filtering_time_ < 0.0)
    {
      throw std::invalid_argument("filtering_time < 0.0");
    }
  }

  double RaycastingTime() const { return raycasting_time_; }

  double FilteringTime() const { return filtering_time_; }

private:
  double raycasting_time_ = 0.0;
  double filtering_time_ = 0.0;
};

class PointCloudVoxelizationInterface
{
public:
  // Delete copy and move constructors and assignment operators.
  explicit PointCloudVoxelizationInterface(
      const PointCloudVoxelizationInterface&) = delete;
  explicit PointCloudVoxelizationInterface(
      PointCloudVoxelizationInterface&&) = delete;
  PointCloudVoxelizationInterface& operator=(
      const PointCloudVoxelizationInterface&) = delete;
  PointCloudVoxelizationInterface& operator=(
      PointCloudVoxelizationInterface&&) = delete;

  virtual ~PointCloudVoxelizationInterface() {}

  OccupancyMap VoxelizePointClouds(
      const OccupancyMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
      const std::function<void(const VoxelizerRuntime&)>&
          runtime_log_fn = {}) const {
    OccupancyMap output_environment = static_environment;
    const auto voxelizer_runtime = VoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options, pointclouds,
        output_environment);
    if (runtime_log_fn)
    {
      runtime_log_fn(voxelizer_runtime);
    }
    return output_environment;
  }

  VoxelizerRuntime VoxelizePointClouds(
      const OccupancyMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
      OccupancyMap& output_environment) const {
    if (!static_environment.IsInitialized())
    {
      throw std::invalid_argument("!static_environment.IsInitialized()");
    }
    if (step_size_multiplier > 1.0 || step_size_multiplier <= 0.0)
    {
      throw std::invalid_argument("step_size_multiplier is not in (0, 1]");
    }
    if (!output_environment.IsInitialized())
    {
      throw std::invalid_argument("!output_environment.IsInitialized()");
    }
    if (static_environment.ControlSizes() != output_environment.ControlSizes())
    {
      throw std::invalid_argument(
          "static_environment.ControlSizes() != "
          "output_environment.ControlSizes()");
    }
    for (size_t idx = 0; idx < pointclouds.size(); idx++)
    {
      const PointCloudWrapperSharedPtr& cloud_ptr = pointclouds.at(idx);
      if (!cloud_ptr)
      {
        throw std::invalid_argument(
            "pointclouds[" + std::to_string(idx) + "] is null");
      }
    }
    return DoVoxelizePointClouds(
        static_environment, step_size_multiplier, filter_options, pointclouds,
        output_environment);
  }

protected:
  virtual VoxelizerRuntime DoVoxelizePointClouds(
      const OccupancyMap& static_environment, const double step_size_multiplier,
      const PointCloudVoxelizationFilterOptions& filter_options,
      const std::vector<PointCloudWrapperSharedPtr>& pointclouds,
      OccupancyMap& output_environment) const = 0;

  PointCloudVoxelizationInterface() = default;
};
}  // namespace pointcloud_voxelization
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
