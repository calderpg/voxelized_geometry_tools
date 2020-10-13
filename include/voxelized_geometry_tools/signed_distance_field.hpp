#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/maybe.hpp>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <unsupported/Eigen/AutoDiff>

namespace voxelized_geometry_tools
{
/// Using declaration to make naming clearer and easier to read.
using EstimateDistanceQuery = common_robotics_utilities::OwningMaybe<double>;

/// This is similar to std::optional<Eigen::Vector4d>, but lets us enforce
/// specific behavior in the contained Vector4d.
class GradientQuery
{
private:
  Eigen::Vector4d gradient_ = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
  bool has_value_ = false;

public:
  explicit GradientQuery(const Eigen::Vector4d& gradient)
      : gradient_(gradient), has_value_(true)
  {
    if (gradient_(3) != 0.0)
    {
      throw std::invalid_argument("gradient(3) != 0.0");
    }
  }

  GradientQuery(const double x, const double y, const double z)
      : gradient_(Eigen::Vector4d(x, y, z, 0.0)), has_value_(true) {}

  GradientQuery() : has_value_(false) {}

  const Eigen::Vector4d& Value() const
  {
    if (HasValue())
    {
      return gradient_;
    }
    else
    {
      throw std::runtime_error("GradientQuery does not have value");
    }
  }

  bool HasValue() const { return has_value_; }

  explicit operator bool() const { return HasValue(); }
};

/// This is equivalent to std::optional<Eigen::Vector4d>, but kept separate here
/// so as to not require C++17 support with a working std::optional<T>. This
/// also allows us to enforce specific behavior in the contained Vector4d.
class ProjectedPosition
{
private:
  const Eigen::Vector4d gradient_ = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  const bool has_value_ = false;

public:
  explicit ProjectedPosition(const Eigen::Vector4d& gradient)
      : gradient_(gradient), has_value_(true)
  {
    if (gradient_(3) != 1.0)
    {
      throw std::invalid_argument("gradient(3) != 1.0");
    }
  }

  ProjectedPosition(const double x, const double y, const double z)
      : gradient_(Eigen::Vector4d(x, y, z, 1.0)), has_value_(true) {}

  ProjectedPosition() : has_value_(false) {}

  const Eigen::Vector4d& Value() const
  {
    if (HasValue())
    {
      return gradient_;
    }
    else
    {
      throw std::runtime_error("ProjectedPosition does not have value");
    }
  }

  bool HasValue() const { return has_value_; }

  explicit operator bool() const { return HasValue(); }
};

template<typename BackingStore=std::vector<float>>
class SignedDistanceField final
    : public common_robotics_utilities::voxel_grid
        ::VoxelGridBase<float, BackingStore>
{
private:
  using FloatSerializer
      = common_robotics_utilities::serialization::Serializer<float>;
  using FloatDeserializer
      = common_robotics_utilities::serialization::Deserializer<float>;
  using DeserializedSignedDistanceField
      = common_robotics_utilities::serialization
          ::Deserialized<SignedDistanceField<BackingStore>>;

  std::string frame_;
  bool locked_ = false;

  /// Internal helper used in "Fine" gradient computation.
  inline double ComputeAxisFineGradient(
      const EstimateDistanceQuery& query_point_distance_estimate,
      const EstimateDistanceQuery& minus_axis_distance_estimate,
      const EstimateDistanceQuery& plus_axis_distance_estimate,
      const double query_point_axis_value,
      const double minus_point_axis_value,
      const double plus_point_axis_value) const
  {
    if (query_point_distance_estimate.HasValue()
        && minus_axis_distance_estimate.HasValue()
        && plus_axis_distance_estimate.HasValue())
    {
      const double window_size = plus_point_axis_value
                                 - minus_point_axis_value;
      const double distance_delta = plus_axis_distance_estimate.Value()
                                    - minus_axis_distance_estimate.Value();
      return distance_delta / window_size;
    }
    else if (query_point_distance_estimate.HasValue()
             && minus_axis_distance_estimate.HasValue())
    {
      const double window_size = query_point_axis_value
                                 - minus_point_axis_value;
      const double distance_delta = query_point_distance_estimate.Value()
                                    - minus_axis_distance_estimate.Value();
      return distance_delta / window_size;
    }
    else if (query_point_distance_estimate.HasValue()
             && plus_axis_distance_estimate.HasValue())
    {
      const double window_size = plus_point_axis_value
                                 - query_point_axis_value;
      const double distance_delta = plus_axis_distance_estimate.Value()
                                    - query_point_distance_estimate.Value();
      return distance_delta / window_size;
    }
    else
    {
      throw std::runtime_error(
            "Window size for GetSmoothGradient is too large for SDF");
    }
  }

  /// Bilinear interpolation used in trilinear interpolation.
  template<typename T>
  inline T BilinearInterpolate(const double low_d1,
                               const double high_d1,
                               const double low_d2,
                               const double high_d2,
                               const T query_d1,
                               const T query_d2,
                               const double l1l2_val,
                               const double l1h2_val,
                               const double h1l2_val,
                               const double h1h2_val) const
  {
    Eigen::Matrix<T, 1, 2> d1_offsets;
    d1_offsets(0, 0) = high_d1 - query_d1;
    d1_offsets(0, 1) = query_d1 - low_d1;
    Eigen::Matrix<T, 2, 2> values;
    values(0, 0) = l1l2_val;
    values(0, 1) = l1h2_val;
    values(1, 0) = h1l2_val;
    values(1, 1) = h1h2_val;
    Eigen::Matrix<T, 2, 1> d2_offsets;
    d2_offsets(0, 0) = high_d2 - query_d2;
    d2_offsets(1, 0) = query_d2 - low_d2;
    const T multiplier = 1.0 / ((high_d1 - low_d1) * (high_d2 - low_d2));
    const T bilinear_interpolated
        = multiplier * d1_offsets * values * d2_offsets;
    return bilinear_interpolated;
  }

  /// Wrapper around bilinear interpolation to set arguments more easily.
  template<typename T>
  inline T BilinearInterpolateDistanceXY(
      const Eigen::Vector4d& corner_location,
      const Eigen::Matrix<T, 4, 1>& query_location,
      const double mxmy_dist, const double mxpy_dist,
      const double pxmy_dist, const double pxpy_dist) const
  {
    return BilinearInterpolate(corner_location(0),
                               corner_location(0) + GetResolution(),
                               corner_location(1),
                               corner_location(1) + GetResolution(),
                               query_location(0),
                               query_location(1),
                               mxmy_dist, mxpy_dist,
                               pxmy_dist, pxpy_dist);
  }

  /// Trilinear distance interpolation.
  template<typename T>
  inline T TrilinearInterpolateDistance(
      const Eigen::Vector4d& corner_location,
      const Eigen::Matrix<T, 4, 1>& query_location,
      const double mxmymz_dist, const double mxmypz_dist,
      const double mxpymz_dist, const double mxpypz_dist,
      const double pxmymz_dist, const double pxmypz_dist,
      const double pxpymz_dist, const double pxpypz_dist) const
  {
    // Do bilinear interpolation in the lower XY plane
    const T mz_bilinear_interpolated
        = BilinearInterpolateDistanceXY(corner_location, query_location,
                                        mxmymz_dist, mxpymz_dist,
                                        pxmymz_dist, pxpymz_dist);
    // Do bilinear interpolation in the upper XY plane
    const T pz_bilinear_interpolated
        = BilinearInterpolateDistanceXY(corner_location, query_location,
                                        mxmypz_dist, mxpypz_dist,
                                        pxmypz_dist, pxpypz_dist);
    // Perform linear interpolation/extrapolation between lower and upper planes
    const double inv_resolution = 1.0 / GetResolution();
    const T distance_delta
        = pz_bilinear_interpolated - mz_bilinear_interpolated;
    const T distance_slope = distance_delta * inv_resolution;
    const T query_z_delta = query_location(2) - T(corner_location(2));
    return mz_bilinear_interpolated + (query_z_delta * distance_slope);
  }

  /// Computes "corrected" center distance accounting for the fact that SDF
  /// distance is the distance to the next filled cell center, *not* the
  /// distance to the boundary.
  inline double GetCorrectedCenterDistance(const int64_t x_idx,
                                           const int64_t y_idx,
                                           const int64_t z_idx) const
  {
    const auto query = this->GetImmutable(x_idx, y_idx, z_idx);
    if (query)
    {
      const double nominal_sdf_distance = static_cast<double>(query.Value());
      const double cell_center_distance_offset = GetResolution() * 0.5;
      if (nominal_sdf_distance >= 0.0)
      {
        return nominal_sdf_distance - cell_center_distance_offset;
      }
      else
      {
        return nominal_sdf_distance + cell_center_distance_offset;
      }
    }
    else
    {
      throw std::invalid_argument("Index out of bounds");
    }
  }

  /// Computes the axis lookup indices to use for interpolation.
  template<typename T>
  std::pair<int64_t, int64_t> GetAxisInterpolationIndices(
      const int64_t initial_index,
      const int64_t axis_size,
      const T axis_offset) const
  {
    int64_t lower = initial_index;
    int64_t upper = initial_index;
    if (axis_offset >= 0.0)
    {
      upper = initial_index + 1;
      if (upper >= axis_size)
      {
        upper = initial_index;
        lower = initial_index -1;
        if (lower < 0)
        {
          lower = initial_index;
        }
      }
    }
    else
    {
      lower = initial_index - 1;
      if (lower < 0)
      {
        upper = initial_index + 1;
        lower = initial_index;
        if (upper >= axis_size)
        {
          upper = initial_index;
        }
      }
    }
    return std::make_pair(lower, upper);
  }

  /// Estimates distance via trilinear interpolation of the surrounding 8 cells.
  template<typename T>
  inline T EstimateDistanceInterpolateFromNeighbors(
      const Eigen::Matrix<T, 4, 1>& query_location,
      const int64_t x_idx, const int64_t y_idx, const int64_t z_idx) const
  {
    // Get the query location in grid frame
    const Eigen::Matrix<T, 4, 1> grid_frame_query_location
        = this->GetInverseOriginTransform() * query_location;
    // Switch between all the possible options of where we are
    const Eigen::Vector4d cell_center_location
        = this->GridIndexToLocationInGridFrame(x_idx, y_idx, z_idx);
    const Eigen::Matrix<T, 4, 1> query_offset
        = grid_frame_query_location - cell_center_location.cast<T>();
    // We can't catch the easiest case of being at a cell center, since doing so
    // breaks the ability of Eigen's Autodiff to autodiff this function.
    // Find the best-matching 8 surrounding cell centers
    const std::pair<int64_t, int64_t> x_axis_indices
        = GetAxisInterpolationIndices(
            x_idx, this->GetNumXCells(), query_offset(0));
    const std::pair<int64_t, int64_t> y_axis_indices
        = GetAxisInterpolationIndices(
            y_idx, this->GetNumYCells(), query_offset(1));
    const std::pair<int64_t, int64_t> z_axis_indices
        = GetAxisInterpolationIndices(
            z_idx, this->GetNumZCells(), query_offset(2));
    const Eigen::Vector4d lower_corner_location
        = this->GridIndexToLocationInGridFrame(x_axis_indices.first,
                                               y_axis_indices.first,
                                               z_axis_indices.first);
    const double mxmymz_distance
        = GetCorrectedCenterDistance(x_axis_indices.first,
                                     y_axis_indices.first,
                                     z_axis_indices.first);
    const double mxmypz_distance
        = GetCorrectedCenterDistance(x_axis_indices.first,
                                     y_axis_indices.first,
                                     z_axis_indices.second);
    const double mxpymz_distance
        = GetCorrectedCenterDistance(x_axis_indices.first,
                                     y_axis_indices.second,
                                     z_axis_indices.first);
    const double mxpypz_distance
        = GetCorrectedCenterDistance(x_axis_indices.first,
                                     y_axis_indices.second,
                                     z_axis_indices.second);
    const double pxmymz_distance
        = GetCorrectedCenterDistance(x_axis_indices.second,
                                     y_axis_indices.first,
                                     z_axis_indices.first);
    const double pxmypz_distance
        = GetCorrectedCenterDistance(x_axis_indices.second,
                                     y_axis_indices.first,
                                     z_axis_indices.second);
    const double pxpymz_distance
        = GetCorrectedCenterDistance(x_axis_indices.second,
                                     y_axis_indices.second,
                                     z_axis_indices.first);
    const double pxpypz_distance
        = GetCorrectedCenterDistance(x_axis_indices.second,
                                     y_axis_indices.second,
                                     z_axis_indices.second);
    return TrilinearInterpolateDistance(lower_corner_location,
                                        grid_frame_query_location,
                                        mxmymz_distance, mxmypz_distance,
                                        mxpymz_distance, mxpypz_distance,
                                        pxmymz_distance, pxmypz_distance,
                                        pxpymz_distance, pxpypz_distance);
  }

  /// You *must* provide valid indices to this function, hence why it's private.
  void FollowGradientsToLocalExtremaUnsafe(
      const int64_t x_index, const int64_t y_index, const int64_t z_index,
      common_robotics_utilities::voxel_grid::VoxelGrid<Eigen::Vector3d>&
          watershed_map) const
  {
    // First, check if we've already found the local extrema for the current
    // cell
    const Eigen::Vector3d& stored
        = watershed_map.GetImmutable(x_index, y_index, z_index).Value();
    if (stored.x() != -std::numeric_limits<double>::infinity()
        && stored.y() != -std::numeric_limits<double>::infinity()
        && stored.z() != -std::numeric_limits<double>::infinity())
    {
      // We've already found it for this cell, so we can skip it
      return;
    }
    // Find the local extrema
    GradientQuery raw_gradient
        = GetCoarseGradient(x_index, y_index, z_index, true);
    Eigen::Vector4d gradient_vector = raw_gradient.Value();
    if (GradientIsEffectiveFlat(gradient_vector))
    {
      const Eigen::Vector4d location
          = this->GridIndexToLocationInGridFrame(x_index, y_index, z_index);
      const Eigen::Vector3d local_extrema(
          location(0), location(1), location(2));
      watershed_map.SetValue(x_index, y_index, z_index, local_extrema);
    }
    else
    {
      // Follow the gradient, one cell at a time, until we reach a local maxima
      std::unordered_map<common_robotics_utilities::voxel_grid::GridIndex,
                         int8_t> path;
      common_robotics_utilities::voxel_grid::GridIndex current_index(
            x_index, y_index, z_index);
      path[current_index] = 1;
      Eigen::Vector3d local_extrema(-std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity());
      while (true)
      {
        if (path.size() == 10000)
        {
          std::cerr << "Warning, gradient path is long (i.e >= 10000 steps)"
                    << std::endl;
        }
        current_index = GetNextFromGradient(current_index, gradient_vector);
        if (path[current_index] != 0)
        {
          // If we've already been here, then we are done
          const Eigen::Vector4d location
              = this->GridIndexToLocationInGridFrame(current_index);
          local_extrema
              = Eigen::Vector3d(location(0), location(1), location(2));
          break;
        }
        // Check if we've been pushed past the edge
        if (!this->IndexInBounds(current_index))
        {
          // We have the "off the grid" local maxima
          local_extrema
              = Eigen::Vector3d(std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity());
          break;
        }
        path[current_index] = 1;
        // Check if the new index has already been checked
        const Eigen::Vector3d& new_stored
            = watershed_map.GetImmutable(current_index).Value();
        if (new_stored.x() != -std::numeric_limits<double>::infinity()
            && new_stored.y() != -std::numeric_limits<double>::infinity()
            && new_stored.z() != -std::numeric_limits<double>::infinity())
        {
          // We have the local maxima
          local_extrema = new_stored;
          break;
        }
        else
        {
          raw_gradient = GetCoarseGradient(current_index, true);
          gradient_vector = raw_gradient.Value();
          if (GradientIsEffectiveFlat(gradient_vector))
          {
            // We have the local maxima
            const Eigen::Vector4d location
                = this->GridIndexToLocationInGridFrame(current_index);
            local_extrema
                = Eigen::Vector3d(location(0), location(1), location(2));
            break;
          }
        }
      }
      // Now, go back and mark the entire explored path with the local maxima
      for (auto path_itr = path.begin(); path_itr != path.end(); ++path_itr)
      {
        const common_robotics_utilities::voxel_grid::GridIndex& index
            = path_itr->first;
        watershed_map.SetValue(index, local_extrema);
      }
    }
  }

  bool GradientIsEffectiveFlat(const Eigen::Vector4d& gradient) const
  {
    // A gradient is at a local maxima if the absolute value of all components
    // (x,y,z) are less than 1/2 SDF resolution
    const double step_resolution = GetResolution() * 0.06125;
    if (std::abs(gradient(0)) <= step_resolution
        && std::abs(gradient(1)) <= step_resolution
        && std::abs(gradient(2)) <= step_resolution)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  common_robotics_utilities::voxel_grid::GridIndex GetNextFromGradient(
      const common_robotics_utilities::voxel_grid::GridIndex& index,
      const Eigen::Vector4d& gradient) const
  {
    // Check if it's inside an obstacle
    const float stored_distance = this->GetImmutable(index).Value();
    Eigen::Vector4d working_gradient = gradient;
    if (stored_distance < 0.0)
    {
      working_gradient = gradient * -1.0;
    }
    // Given the gradient, pick the "best fit" of the 26 neighboring points
    common_robotics_utilities::voxel_grid::GridIndex next_index = index;
    const double step_resolution = GetResolution() * 0.06125;
    if (working_gradient(0) > step_resolution)
    {
      next_index.X() += 1;
    }
    else if (working_gradient(0) < -step_resolution)
    {
      next_index.X() -= 1;
    }
    if (working_gradient(1) > step_resolution)
    {
      next_index.Y() += 1;
    }
    else if (working_gradient(1) < -step_resolution)
    {
      next_index.Y() -= 1;
    }
    if (working_gradient(2) > step_resolution)
    {
      next_index.Z() += 1;
    }
    else if (working_gradient(2) < -step_resolution)
    {
      next_index.Z() -= 1;
    }
    return next_index;
  }

  /// Implement the VoxelGridBase interface.

  /// We need to implement cloning.
  std::unique_ptr<common_robotics_utilities::voxel_grid
      ::VoxelGridBase<float, BackingStore>>
  DoClone() const override
  {
    return std::unique_ptr<SignedDistanceField<BackingStore>>(
        new SignedDistanceField<BackingStore>(*this));
  }

  /// We need to serialize the frame and locked flag.
  uint64_t DerivedSerializeSelf(
      std::vector<uint8_t>& buffer,
      const FloatSerializer& value_serializer) const override
  {
    UNUSED(value_serializer);
    const uint64_t start_size = buffer.size();
    common_robotics_utilities::serialization::SerializeString(frame_, buffer);
    common_robotics_utilities::serialization::SerializeMemcpyable<uint8_t>(
        static_cast<uint8_t>(locked_), buffer);
    const uint64_t bytes_written = buffer.size() - start_size;
    return bytes_written;
  }

  /// We need to deserialize the frame and locked flag.
  uint64_t DerivedDeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
      const FloatDeserializer& value_deserializer) override
  {
    UNUSED(value_deserializer);
    uint64_t current_position = starting_offset;
    // Deserialize SDF stuff
    const auto frame_deserialized
        = common_robotics_utilities::serialization::DeserializeString<char>(
            buffer, current_position);
    frame_ = frame_deserialized.Value();
    current_position += frame_deserialized.BytesRead();
    const auto locked_deserialized
        = common_robotics_utilities::serialization
            ::DeserializeMemcpyable<uint8_t>(buffer, current_position);
    locked_ = static_cast<bool>(locked_deserialized.Value());
    current_position += locked_deserialized.BytesRead();
    // Figure out how many bytes were read
    const uint64_t bytes_read = current_position - starting_offset;
    return bytes_read;
  }

  /// We do not allow mutable access if the SDF is locked.
  bool OnMutableAccess(const int64_t x_index,
                       const int64_t y_index,
                       const int64_t z_index) override
  {
    UNUSED(x_index);
    UNUSED(y_index);
    UNUSED(z_index);
    return !IsLocked();
  }

public:
  static uint64_t Serialize(
      const SignedDistanceField<BackingStore>& sdf,
      std::vector<uint8_t>& buffer)
  {
    return sdf.SerializeSelf(buffer, common_robotics_utilities::serialization
                                         ::SerializeMemcpyable<float>);
  }

  static DeserializedSignedDistanceField Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
  {
    SignedDistanceField<BackingStore> temp_sdf;
    const uint64_t bytes_read
        = temp_sdf.DeserializeSelf(
            buffer, starting_offset,
            common_robotics_utilities::serialization
                ::DeserializeMemcpyable<float>);
    return common_robotics_utilities::serialization::MakeDeserialized(
        temp_sdf, bytes_read);
  }

  static void SaveToFile(
      const SignedDistanceField<BackingStore>& sdf, const std::string& filepath,
      const bool compress)
  {
    std::vector<uint8_t> buffer;
    SignedDistanceField<BackingStore>::Serialize(sdf, buffer);
    std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
    if (compress)
    {
      output_file.write("SDFZ", 4);
      const std::vector<uint8_t> compressed
          = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
      const size_t serialized_size = compressed.size();
      output_file.write(
          reinterpret_cast<const char*>(compressed.data()),
          static_cast<std::streamsize>(serialized_size));
    }
    else
    {
      output_file.write("SDFR", 4);
      const size_t serialized_size = buffer.size();
      output_file.write(
          reinterpret_cast<const char*>(buffer.data()),
          static_cast<std::streamsize>(serialized_size));
    }
    output_file.close();
  }

  static SignedDistanceField<BackingStore> LoadFromFile(
      const std::string& filepath)
  {
    std::ifstream input_file(
        filepath, std::ios::in | std::ios::binary | std::ios::ate);
    if (input_file.good() == false)
    {
      throw std::invalid_argument("File does not exist");
    }
    const std::streampos end = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    const std::streampos begin = input_file.tellg();
    const std::streamsize serialized_size = end - begin;
    const std::streamsize header_size = 4;
    if (serialized_size >= header_size)
    {
      // Load the header
      std::vector<uint8_t> file_header(header_size + 1, 0x00);
      input_file.read(reinterpret_cast<char*>(file_header.data()),
                      header_size);
      const std::string header_string(
            reinterpret_cast<const char*>(file_header.data()));
      // Load the rest of the file
      std::vector<uint8_t> file_buffer(
            static_cast<size_t>(serialized_size - header_size), 0x00);
      input_file.read(reinterpret_cast<char*>(file_buffer.data()),
                      serialized_size - header_size);
      // Deserialize
      if (header_string == "SDFZ")
      {
        const std::vector<uint8_t> decompressed
            = common_robotics_utilities::zlib_helpers
                ::DecompressBytes(file_buffer);
        return SignedDistanceField<BackingStore>::Deserialize(
            decompressed, 0).Value();
      }
      else if (header_string == "SDFR")
      {
        return SignedDistanceField<BackingStore>::Deserialize(
            file_buffer, 0).Value();
      }
      else
      {
        throw std::invalid_argument(
              "File has invalid header [" + header_string + "]");
      }
    }
    else
    {
      throw std::invalid_argument("File is too small");
    }
  }

  SignedDistanceField(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const float default_value)
      : SignedDistanceField(
          origin_transform, frame, sizes, default_value, default_value) {}

  SignedDistanceField(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const float default_value)
      : SignedDistanceField(frame, sizes, default_value, default_value) {}

  SignedDistanceField(
      const Eigen::Isometry3d& origin_transform, const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const float default_value, const float oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<float, BackingStore>(
              origin_transform, sizes, default_value, oob_value), frame_(frame),
        locked_(false)
  {
    if (!this->HasUniformCellSize())
    {
      throw std::invalid_argument("SDF cannot have non-uniform cell sizes");
    }
  }

  SignedDistanceField(
      const std::string& frame,
      const common_robotics_utilities::voxel_grid::GridSizes& sizes,
      const float default_value, const float oob_value)
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<float, BackingStore>(sizes, default_value, oob_value),
        frame_(frame), locked_(false)
  {
    if (!this->HasUniformCellSize())
    {
      throw std::invalid_argument("SDF cannot have non-uniform cell sizes");
    }
  }

  SignedDistanceField()
      : common_robotics_utilities::voxel_grid
          ::VoxelGridBase<float, BackingStore>() {}

  bool IsLocked() const { return locked_; }

  void Lock() { locked_ = true; }

  void Unlock() { locked_ = false; }

  double GetResolution() const { return this->GetCellSizes().x(); }

  const std::string& GetFrame() const { return frame_; }

  void SetFrame(const std::string& frame) { frame_ = frame; }

  /// For classical SDF distance queries, see the API of VoxelGridBase for how
  /// to retrieve and set cell values.

  /// "Estimated" distance is computed by performing trilinear interpolation
  /// against the distances in eight cells that surround the query point. This
  /// produces a continuous, but not necessarily smooth distance function that
  /// is more accurate and smoother than the raw distance values inside cells.
  /// Note that it is both more expensive than a simple grid lookup, but also
  /// this rounds inside and outside corners of cells. This can be a problem at
  /// coarse grid resolutions.

  EstimateDistanceQuery EstimateDistance(
      const double x, const double y, const double z) const
  {
      return EstimateDistance4d(Eigen::Vector4d(x, y, z, 1.0));
  }

  EstimateDistanceQuery EstimateDistance3d(
      const Eigen::Vector3d& location) const
  {
    const common_robotics_utilities::voxel_grid::GridIndex index
        = this->LocationToGridIndex3d(location);
    if (this->IndexInBounds(index))
    {
      return EstimateDistanceQuery(
          EstimateDistanceInterpolateFromNeighbors<double>(
              Eigen::Vector4d(location.x(), location.y(), location.z(), 1.0),
              index.X(), index.Y(), index.Z()));
    }
    else
    {
      return EstimateDistanceQuery();
    }
  }

  EstimateDistanceQuery EstimateDistance4d(
      const Eigen::Vector4d& location) const
  {
    const common_robotics_utilities::voxel_grid::GridIndex index
        = this->LocationToGridIndex4d(location);
    if (this->IndexInBounds(index))
    {
      return EstimateDistanceQuery(
          EstimateDistanceInterpolateFromNeighbors<double>(
              location, index.X(), index.Y(), index.Z()));
    }
    else
    {
      return EstimateDistanceQuery();
    }
  }

  /// "Coarse" gradient is computed by retrieving the distance from the
  /// surrounding size cells (+/-x, +/-y, +/-z) and differencing. This is the
  /// fastest method to compute a gradient, and also has the most potential
  /// error and least-smooth behavior.

  GradientQuery GetCoarseGradient(
      const double x, const double y, const double z,
      const bool enable_edge_gradients=false) const
  {
    return GetCoarseGradient4d(
        Eigen::Vector4d(x, y, z, 1.0), enable_edge_gradients);
  }

  GradientQuery GetCoarseGradient3d(
      const Eigen::Vector3d& location,
      const bool enable_edge_gradients=false) const
  {
    const common_robotics_utilities::voxel_grid::GridIndex index
        = this->LocationToGridIndex3d(location);
    if (this->IndexInBounds(index))
    {
      return GetCoarseGradient(index, enable_edge_gradients);
    }
    else
    {
      return GradientQuery();
    }
  }

  GradientQuery GetCoarseGradient4d(
      const Eigen::Vector4d& location,
      const bool enable_edge_gradients=false) const
  {
    const common_robotics_utilities::voxel_grid::GridIndex index
        = this->LocationToGridIndex4d(location);
    if (this->IndexInBounds(index))
    {
      return GetCoarseGradient(index, enable_edge_gradients);
    }
    else
    {
      return GradientQuery();
    }
  }

  GradientQuery GetCoarseGradient(
      const common_robotics_utilities::voxel_grid::GridIndex& index,
      const bool enable_edge_gradients=false) const
  {
    return GetCoarseGradient(
        index.X(), index.Y(), index.Z(), enable_edge_gradients);
  }

  GradientQuery GetCoarseGradient(
      const int64_t x_index, const int64_t y_index, const int64_t z_index,
      const bool enable_edge_gradients=false) const
  {
    const GradientQuery grid_aligned_gradient
        = GetGridAlignedCoarseGradient(x_index, y_index, z_index,
                                       enable_edge_gradients);
    if (grid_aligned_gradient.HasValue())
    {
      const Eigen::Vector4d& raw_grid_aligned_gradient
          = grid_aligned_gradient.Value();
      const Eigen::Quaterniond grid_rotation(
          this->GetOriginTransform().rotation());
      const Eigen::Quaterniond temp(0.0,
                                    raw_grid_aligned_gradient(0),
                                    raw_grid_aligned_gradient(1),
                                    raw_grid_aligned_gradient(2));
      const Eigen::Quaterniond result
          = grid_rotation * (temp * grid_rotation.inverse());
      return GradientQuery(result.x(), result.y(), result.z());
    }
    else
    {
      return grid_aligned_gradient;
    }
  }

  GradientQuery GetGridAlignedCoarseGradient(
      const int64_t x_index, const int64_t y_index, const int64_t z_index,
      const bool enable_edge_gradients=false) const
  {
    // Make sure the index is inside bounds
    if (this->IndexInBounds(x_index, y_index, z_index))
    {
      // See if the index we're trying to query is one cell in from the edge
      if ((x_index > 0) && (y_index > 0) && (z_index > 0)
          && (x_index < (this->GetNumXCells() - 1))
          && (y_index < (this->GetNumYCells() - 1))
          && (z_index < (this->GetNumZCells() - 1)))
      {
        const double inv_twice_resolution = 1.0 / (2.0 * GetResolution());
        const double gx
            = (this->GetImmutable(x_index + 1, y_index, z_index).Value()
               - this->GetImmutable(x_index - 1, y_index, z_index).Value())
              * inv_twice_resolution;
        const double gy
            = (this->GetImmutable(x_index, y_index + 1, z_index).Value()
               - this->GetImmutable(x_index, y_index - 1, z_index).Value())
              * inv_twice_resolution;
        const double gz
            = (this->GetImmutable(x_index, y_index, z_index + 1).Value()
               - this->GetImmutable(x_index, y_index, z_index - 1).Value())
              * inv_twice_resolution;
        return GradientQuery(gx, gy, gz);
      }
      // If we're on the edge, handle it specially
      // TODO: we actually need to handle corners even more carefully,
      // since if the SDF is build with a virtual border, these cells will
      // get zero gradient from this approach!
      else if (enable_edge_gradients)
      {
        // Get the "best" indices we can use
        const int64_t low_x_index
            = std::max(static_cast<int64_t>(0), x_index - 1);
        const int64_t high_x_index
            = std::min(this->GetNumXCells() - 1, x_index + 1);
        const int64_t low_y_index
            = std::max(static_cast<int64_t>(0), y_index - 1);
        const int64_t high_y_index
            = std::min(this->GetNumYCells() - 1, y_index + 1);
        const int64_t low_z_index
            = std::max(static_cast<int64_t>(0), z_index - 1);
        const int64_t high_z_index
            = std::min(this->GetNumZCells() - 1, z_index + 1);
        // Compute the axis increments
        const double x_increment
            = static_cast<double>(high_x_index - low_x_index) * GetResolution();
        const double y_increment
            = static_cast<double>(high_y_index - low_y_index) * GetResolution();
        const double z_increment
            = static_cast<double>(high_z_index - low_z_index) * GetResolution();
        // Compute the gradients for each axis - by default these are zero
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
        // Only if the increments are non-zero do we compute the axis gradient
        if (x_increment > 0.0)
        {
          const double inv_x_increment = 1.0 / x_increment;
          const double high_x_value
              = this->GetImmutable(high_x_index, y_index, z_index).Value();
          const double low_x_value
              = this->GetImmutable(low_x_index, y_index, z_index).Value();
          // Compute the gradient
          gx = (high_x_value - low_x_value) * inv_x_increment;
        }
        if (y_increment > 0.0)
        {
          const double inv_y_increment = 1.0 / y_increment;
          const double high_y_value
              = this->GetImmutable(x_index, high_y_index, z_index).Value();
          const double low_y_value
              = this->GetImmutable(x_index, low_y_index, z_index).Value();
          // Compute the gradient
          gy = (high_y_value - low_y_value) * inv_y_increment;
        }
        if (z_increment > 0.0)
        {
          const double inv_z_increment = 1.0 / z_increment;
          const double high_z_value
              = this->GetImmutable(x_index, y_index, high_z_index).Value();
          const double low_z_value
              = this->GetImmutable(x_index, y_index, low_z_index).Value();
          // Compute the gradient
          gz = (high_z_value - low_z_value) * inv_z_increment;
        }
        // Assemble and return the computed gradient
        return GradientQuery(gx, gy, gz);
      }
      // Edge gradients disabled, return no gradient
      else
      {
        return GradientQuery();
      }
    }
    // If we're out of bounds, return no gradient
    else
    {
      return GradientQuery();
    }
  }

  /// "Fine" gradient is also computed by differencing, but instead of using the
  /// surrounding cells, this uses EstimateDistance() at +/- half of
  /// "window size" to compute the gradient. This is slower than the "coarse"
  /// gradient, since this needs to call EstimateDistance() six times, but
  /// produces a more accurate and smoother gradient. In the limit as
  /// "window size" -> 0 this produces the true gradient.

  GradientQuery GetFineGradient3d(
      const Eigen::Vector3d& location,
      const double nominal_window_size) const
  {
    return GetFineGradient(location.x(), location.y(), location.z(),
                           nominal_window_size);
  }

  GradientQuery GetFineGradient4d(
      const Eigen::Vector4d& location,
      const double nominal_window_size) const
  {
    return GetFineGradient(location(0), location(1), location(2),
                           nominal_window_size);
  }

  GradientQuery GetFineGradient(
      const double x, const double y, const double z,
      const double nominal_window_size) const
  {
    const double ideal_window_size = std::abs(nominal_window_size);
    if (this->LocationInBounds(x, y, z))
    {
      const double min_x = x - ideal_window_size;
      const double max_x = x + ideal_window_size;
      const double min_y = y - ideal_window_size;
      const double max_y = y + ideal_window_size;
      const double min_z = z - ideal_window_size;
      const double max_z = z + ideal_window_size;
      // Retrieve distance estimates
      const EstimateDistanceQuery point_distance = EstimateDistance(x, y, z);
      const EstimateDistanceQuery mx_distance = EstimateDistance(min_x, y, z);
      const EstimateDistanceQuery px_distance = EstimateDistance(max_x, y, z);
      const EstimateDistanceQuery my_distance = EstimateDistance(x, min_y, z);
      const EstimateDistanceQuery py_distance = EstimateDistance(x, max_y, z);
      const EstimateDistanceQuery mz_distance = EstimateDistance(x, y, min_z);
      const EstimateDistanceQuery pz_distance = EstimateDistance(x, y, max_z);
      // Compute gradient for each axis
      const double gx = ComputeAxisFineGradient(point_distance,
                                                  mx_distance,
                                                  px_distance,
                                                  x, min_x, max_x);
      const double gy = ComputeAxisFineGradient(point_distance,
                                                  my_distance,
                                                  py_distance,
                                                  y, min_y, max_y);
      const double gz = ComputeAxisFineGradient(point_distance,
                                                  mz_distance,
                                                  pz_distance,
                                                  z, min_z, max_z);
      return GradientQuery(gx, gy, gz);
    }
    else
    {
      return GradientQuery();
    }
  }

  GradientQuery GetFineGradient(
      const common_robotics_utilities::voxel_grid::GridIndex& index,
      const double nominal_window_size) const
  {
    return GetFineGradient4d(
        this->GridIndexToLocation(index), nominal_window_size);
  }

  GradientQuery GetFineGradient(
      const int64_t x_index, const int64_t y_index, const int64_t z_index,
      const double nominal_window_size) const
  {
    return GetFineGradient4d(
        this->GridIndexToLocation(x_index, y_index, z_index),
        nominal_window_size);
  }

  /// "Autodiff" gradient is computed by using Eigen's autodiff system to
  /// autodiff EstimateDistance instead. This is potentially the slowest of the
  /// gradient computation options, but should be the smoothest.

  GradientQuery GetAutoDiffGradient3d(
      const Eigen::Vector3d& location) const
  {
    return GetAutoDiffGradient(location.x(), location.y(), location.z());
  }

  GradientQuery GetAutoDiffGradient4d(
      const Eigen::Vector4d& location) const
  {
    return GetAutoDiffGradient(location(0), location(1), location(2));
  }

  GradientQuery GetAutoDiffGradient(
      const double x, const double y, const double z) const
  {
    const common_robotics_utilities::voxel_grid::GridIndex index
        = this->LocationToGridIndex(x, y, z);
    // Query
    if (this->IndexInBounds(index))
    {
      // Check if the query location is the cell center
      const Eigen::Vector4d cell_center_location
          = this->GridIndexToLocation(index);
      // TODO - check how this affects results
      // TODO - this definitely doesn't fix it
      // Bump the query point away from the cell center
      double adjusted_x = x;
      double adjusted_y = y;
      double adjusted_z = z;
      if (adjusted_x == cell_center_location(0))
      {
        adjusted_x += (GetResolution() * 0.125);
      }
      if (adjusted_y == cell_center_location(1))
      {
        adjusted_y += (GetResolution() * 0.125);
      }
      if (adjusted_z == cell_center_location(2))
      {
        adjusted_z += (GetResolution() * 0.125);
      }
      // Use with AutoDiffScalar
      typedef Eigen::AutoDiffScalar<Eigen::Vector4d> AScalar;
      typedef Eigen::Matrix<AScalar, 4, 1> APosition;
      APosition Alocation;
      Alocation(0) = adjusted_x;
      Alocation(1) = adjusted_y;
      Alocation(2) = adjusted_z;
      Alocation(3) = 1.0;
      Alocation(0).derivatives() = Eigen::Vector4d::Unit(0);
      Alocation(1).derivatives() = Eigen::Vector4d::Unit(1);
      Alocation(2).derivatives() = Eigen::Vector4d::Unit(2);
      Alocation(3).derivatives() = Eigen::Vector4d::Unit(3);
      AScalar Adist = EstimateDistanceInterpolateFromNeighbors<AScalar>(
                        Alocation, index.X(), index.Y(), index.Z());
      return GradientQuery(Adist.derivatives()(0),
                           Adist.derivatives()(1),
                           Adist.derivatives()(2));
    }
    else
    {
      return GradientQuery();
    }
  }

  GradientQuery GetAutoDiffGradient(
      const common_robotics_utilities::voxel_grid::GridIndex& index) const
  {
    return GetAutoDiffGradient4d(this->GridIndexToLocation(index));
  }

  GradientQuery GetAutoDiffGradient(
      const int64_t x_index, const int64_t y_index, const int64_t z_index) const
  {
    return GetAutoDiffGradient4d(
          this->GridIndexToLocation(x_index, y_index, z_index));
  }

  /// Project the provided point out of collision.

  ProjectedPosition ProjectOutOfCollision(
      const double x, const double y, const double z,
      const double stepsize_multiplier = 1.0 / 10.0) const
  {
    return ProjectOutOfCollision4d(
        Eigen::Vector4d(x, y, z, 1.0), stepsize_multiplier);
  }

  ProjectedPosition ProjectOutOfCollision3d(
      const Eigen::Vector3d& location,
      const double stepsize_multiplier = 1.0 / 10.0) const
  {
    return ProjectOutOfCollision(
        location.x(), location.y(), location.z(), stepsize_multiplier);
  }

  ProjectedPosition ProjectOutOfCollision4d(
      const Eigen::Vector4d& location,
      const double stepsize_multiplier = 1.0 / 10.0) const
  {
    return ProjectOutOfCollisionToMinimumDistance4d(
        location, 0.0, stepsize_multiplier);
  }

  ProjectedPosition ProjectOutOfCollisionToMinimumDistance(
      const double x, const double y, const double z,
      const double minimum_distance,
      const double stepsize_multiplier = 1.0 / 10.0) const
  {
    return ProjectOutOfCollisionToMinimumDistance4d(
        Eigen::Vector4d(x, y, z, 1.0), minimum_distance, stepsize_multiplier);
  }

  ProjectedPosition ProjectOutOfCollisionToMinimumDistance3d(
      const Eigen::Vector3d& location, const double minimum_distance,
      const double stepsize_multiplier = 1.0 / 10.0) const
  {
    return ProjectOutOfCollisionToMinimumDistance(
        location.x(), location.y(), location.z(),
        minimum_distance, stepsize_multiplier);
  }

  ProjectedPosition ProjectOutOfCollisionToMinimumDistance4d(
      const Eigen::Vector4d& location,
      const double minimum_distance,
      const double stepsize_multiplier = 1.0 / 10.0) const
  {
    // To avoid potential problems with alignment, we need to pass location
    // by reference, so we make a local copy here that we can change.
    // See https://eigen.tuxfamily.org/dox/group__TopicPassingByValue.html
    Eigen::Vector4d mutable_location = location;
    // If we are in bounds, start the projection process,
    // otherwise return the location unchanged
    if (this->LocationInBounds4d(mutable_location))
    {
      // Add a small collision margin to account for rounding and similar
      const double minimum_distance_with_margin
          = minimum_distance + GetResolution() * stepsize_multiplier * 1e-3;
      const double max_stepsize = GetResolution() * stepsize_multiplier;
      const bool enable_edge_gradients = true;
      double sdf_dist = EstimateDistance4d(mutable_location).Value();
      while (sdf_dist <= minimum_distance)
      {
        const GradientQuery gradient
            = GetCoarseGradient4d(mutable_location, enable_edge_gradients);
        if (gradient.HasValue())
        {
          const Eigen::Vector4d& grad_vector = gradient.Value();
          if (grad_vector.norm() > GetResolution() * 0.25) // Sanity check
          {
            // Don't step any farther than is needed
            const double step_distance
                = std::min(max_stepsize,
                           minimum_distance_with_margin - sdf_dist);
            mutable_location += grad_vector.normalized() * step_distance;
            sdf_dist = EstimateDistance4d(mutable_location).Valeu();
          }
          else
          {
            std::cerr << "Encountered flat gradient - stuck" << std::endl;
            return ProjectedPosition();
          }
        }
        else
        {
          std::cerr << "Failed to compute gradient - out of SDF?" << std::endl;
          return ProjectedPosition();
        }
      }
    }
    return ProjectedPosition(mutable_location);
  }

  /// The following function can be *very expensive* to compute, since it
  /// performs gradient ascent/descent across the SDF.
  common_robotics_utilities::voxel_grid::VoxelGrid<Eigen::Vector3d>
  ComputeLocalExtremaMap() const
  {
    const Eigen::Vector3d default_value(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    common_robotics_utilities::voxel_grid::VoxelGrid<Eigen::Vector3d>
        watershed_map(this->GetOriginTransform(), this->GetGridSizes(),
                      default_value);
    for (int64_t x_idx = 0; x_idx < watershed_map.GetNumXCells(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < watershed_map.GetNumYCells(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < watershed_map.GetNumZCells(); z_idx++)
        {
          // We use an "unsafe" function here because we know all the indices
          // we provide it will be safe
          FollowGradientsToLocalExtremaUnsafe(
              x_idx, y_idx, z_idx, watershed_map);
        }
      }
    }
    return watershed_map;
  }
};
}  // namespace voxelized_geometry_tools
