#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/parallelism.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>
#include <voxelized_geometry_tools/vgt_namespace.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace signed_distance_field_generation
{
// To allow for faster development/improvements, the SignedDistanceField
// generation API has no stability guarantees and lives in an internal
// namespace. Use the stable API exposed by occupancy map types instead.
namespace internal
{
using common_robotics_utilities::voxel_grid::GridIndex;
using common_robotics_utilities::voxel_grid::Vector3i64;
using common_robotics_utilities::voxel_grid::VoxelGridSizes;

using EDTDistanceField =
    common_robotics_utilities::voxel_grid::VoxelGrid<double>;

void ComputeDistanceFieldTransformInPlace(
    const common_robotics_utilities::parallelism::DegreeOfParallelism&
        parallelism,
    EDTDistanceField& distance_field);

template<typename SDFScalarType>
inline SignedDistanceField<SDFScalarType> ExtractSignedDistanceField(
    const Eigen::Isometry3d& grid_origin_transform,
    const VoxelGridSizes& grid_sizes,
    const std::function<bool(const GridIndex&)>& is_filled_fn,
    const std::string& frame,
    const SignedDistanceFieldGenerationParameters<SDFScalarType>& parameters)
{
  const double marked_cell = 0.0;
  const double unmarked_cell = std::numeric_limits<double>::infinity();

  constexpr auto parallel_backend =
      common_robotics_utilities::parallelism::ParallelForBackend
          ::BEST_AVAILABLE;

  // Make two distance fields, one for distance to filled voxels, one for
  // distance to free voxels.
  EDTDistanceField filled_distance_field(
      grid_origin_transform, grid_sizes, unmarked_cell);
  auto& filled_distance_field_raw_data =
      filled_distance_field.GetMutableRawData();

  EDTDistanceField free_distance_field(
      grid_origin_transform, grid_sizes, unmarked_cell);
  auto& free_distance_field_raw_data =
      free_distance_field.GetMutableRawData();

  const auto fill_fields_work = [&](
      const common_robotics_utilities::parallelism::ThreadWorkRange& work_range)
  {
    for (int64_t data_index = work_range.GetRangeStart();
         data_index < work_range.GetRangeEnd(); data_index++)
    {
      const GridIndex grid_index = grid_sizes.DataIndexToGridIndex(data_index);

      if (is_filled_fn(grid_index))
      {
        filled_distance_field_raw_data[data_index] = marked_cell;
      }
      else
      {
        free_distance_field_raw_data[data_index] = marked_cell;
      }
    }
  };

  common_robotics_utilities::parallelism::StaticParallelForRangeLoop(
      parameters.Parallelism(), 0, grid_sizes.NumTotalVoxels(),
      fill_fields_work, parallel_backend);

  // Compute the distance transforms in place
  ComputeDistanceFieldTransformInPlace(
      parameters.Parallelism(), filled_distance_field);
  ComputeDistanceFieldTransformInPlace(
      parameters.Parallelism(), free_distance_field);

  // Generate the SDF
  SignedDistanceField<SDFScalarType> new_sdf(
      grid_origin_transform, frame, grid_sizes, parameters.OOBValue());
  const double resolution = new_sdf.Resolution();
  auto& new_sdf_raw_data = new_sdf.GetMutableRawData();

  std::vector<double> per_thread_minimums(
      parameters.Parallelism().GetNumThreads(),
      std::numeric_limits<double>::infinity());
  std::vector<double> per_thread_maximums(
      parameters.Parallelism().GetNumThreads(),
      -std::numeric_limits<double>::infinity());

  const auto fill_sdf_work = [&](
      const common_robotics_utilities::parallelism::ThreadWorkRange& work_range)
  {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

    for (int64_t data_index = work_range.GetRangeStart();
         data_index < work_range.GetRangeEnd(); data_index++)
    {
      const double filled_distance_squared =
          filled_distance_field_raw_data[data_index];
      const double free_distance_squared =
          free_distance_field_raw_data[data_index];

      const double distance1 = std::sqrt(filled_distance_squared) * resolution;
      const double distance2 = std::sqrt(free_distance_squared) * resolution;
      const double distance = distance1 - distance2;

      minimum = std::min(minimum, distance);
      maximum = std::max(maximum, distance);

      new_sdf_raw_data[data_index] = static_cast<SDFScalarType>(distance);
    }

    per_thread_minimums.at(work_range.GetThreadNum()) = minimum;
    per_thread_maximums.at(work_range.GetThreadNum()) = maximum;
  };

  common_robotics_utilities::parallelism::StaticParallelForRangeLoop(
      parameters.Parallelism(), 0, grid_sizes.NumTotalVoxels(), fill_sdf_work,
      parallel_backend);

  const double minimum =
      *std::min_element(per_thread_minimums.begin(), per_thread_minimums.end());
  const double maximum =
      *std::max_element(per_thread_maximums.begin(), per_thread_maximums.end());

  const SignedDistanceFieldMinimumMaximum minimum_maximum(
      static_cast<SDFScalarType>(minimum), static_cast<SDFScalarType>(maximum));

  // Lock & update min/max values.
  new_sdf.LockWithMinimumMaximum(minimum_maximum);
  return new_sdf;
}

template<typename SDFScalarType, typename VoxelGridType>
inline SignedDistanceField<SDFScalarType> ExtractSignedDistanceField(
    const VoxelGridType& grid,
    const std::function<bool(const GridIndex&)>& is_filled_fn,
    const std::string& frame,
    const SignedDistanceFieldGenerationParameters<SDFScalarType>& parameters)
{
  if (!grid.HasUniformVoxelSize())
  {
    throw std::invalid_argument("Grid must have uniform resolution");
  }
  if (!parameters.AddVirtualBorder())
  {
    // This is the conventional single-pass result
    return ExtractSignedDistanceField<SDFScalarType>(
        grid.OriginTransform(), grid.ControlSizes(), is_filled_fn, frame,
        parameters);
  }
  else
  {
    const int64_t x_axis_size_offset =
        (grid.NumXVoxels() > 1) ? INT64_C(2) : INT64_C(0);
    const int64_t x_axis_query_offset =
        (grid.NumXVoxels() > 1) ? INT64_C(1) : INT64_C(0);
    const int64_t y_axis_size_offset =
        (grid.NumYVoxels() > 1) ? INT64_C(2) : INT64_C(0);
    const int64_t y_axis_query_offset =
        (grid.NumYVoxels() > 1) ? INT64_C(1) : INT64_C(0);
    const int64_t z_axis_size_offset =
        (grid.NumZVoxels() > 1) ? INT64_C(2) : INT64_C(0);
    const int64_t z_axis_query_offset =
        (grid.NumZVoxels() > 1) ? INT64_C(1) : INT64_C(0);

    const Vector3i64 query_offset(
        x_axis_query_offset, y_axis_query_offset, z_axis_query_offset);

    // We need to lie about the size of the grid to add a virtual border
    const int64_t num_x_cells = grid.NumXVoxels() + x_axis_size_offset;
    const int64_t num_y_cells = grid.NumYVoxels() + y_axis_size_offset;
    const int64_t num_z_cells = grid.NumZVoxels() + z_axis_size_offset;

    // Make some deceitful helper functions that hide our lies about size
    // For the free space SDF, we lie and say the virtual border is filled
    const std::function<bool(const GridIndex&)> free_is_filled_fn
        = [&] (const GridIndex& virtual_border_grid_index)
    {
      // Is there a virtual border on our axis?
      if (x_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.X() == 0)
            || (virtual_border_grid_index.X() == (num_x_cells - 1)))
        {
          return true;
        }
      }
      // Is there a virtual border on our axis?
      if (y_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.Y() == 0)
            || (virtual_border_grid_index.Y() == (num_y_cells - 1)))
        {
          return true;
        }
      }
      // Is there a virtual border on our axis?
      if (z_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.Z() == 0)
            || (virtual_border_grid_index.Z() == (num_z_cells - 1)))
        {
          return true;
        }
      }
      const GridIndex real_grid_index =
          virtual_border_grid_index - query_offset;
      return is_filled_fn(real_grid_index);
    };

    // For the filled space SDF, we lie and say the virtual border is empty
    const std::function<bool(const GridIndex&)> filled_is_filled_fn
        = [&] (const GridIndex& virtual_border_grid_index)
    {
      // Is there a virtual border on our axis?
      if (x_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.X() == 0)
            || (virtual_border_grid_index.X() == (num_x_cells - 1)))
        {
          return false;
        }
      }
      // Is there a virtual border on our axis?
      if (y_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.Y() == 0)
            || (virtual_border_grid_index.Y() == (num_y_cells - 1)))
        {
          return false;
        }
      }
      // Is there a virtual border on our axis?
      if (z_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.Z() == 0)
            || (virtual_border_grid_index.Z() == (num_z_cells - 1)))
        {
          return false;
        }
      }
      const GridIndex real_grid_index =
          virtual_border_grid_index - query_offset;
      return is_filled_fn(real_grid_index);
    };

    // Make both SDFs
    const auto enlarged_sizes = VoxelGridSizes::FromVoxelCounts(
        grid.VoxelXSize(), Vector3i64(num_x_cells, num_y_cells, num_z_cells));
    const auto free_sdf = ExtractSignedDistanceField<SDFScalarType>(
        grid.OriginTransform(), enlarged_sizes, free_is_filled_fn, frame,
        parameters);
    const auto filled_sdf = ExtractSignedDistanceField<SDFScalarType>(
        grid.OriginTransform(), enlarged_sizes, filled_is_filled_fn, frame,
        parameters);

    // Combine to make a single SDF
    SignedDistanceField<SDFScalarType> combined_sdf(
        grid.OriginTransform(), frame, grid.ControlSizes(),
        parameters.OOBValue());
    auto& combined_sdf_raw_data = combined_sdf.GetMutableRawData();

    std::vector<SDFScalarType> per_thread_minimums(
        parameters.Parallelism().GetNumThreads(),
        std::numeric_limits<SDFScalarType>::infinity());
    std::vector<SDFScalarType> per_thread_maximums(
        parameters.Parallelism().GetNumThreads(),
        -std::numeric_limits<SDFScalarType>::infinity());

    const auto fill_sdf_work = [&](
        const common_robotics_utilities::parallelism::ThreadWorkRange& work_range)
    {
      SDFScalarType minimum = std::numeric_limits<SDFScalarType>::infinity();
      SDFScalarType maximum = -std::numeric_limits<SDFScalarType>::infinity();

      for (int64_t data_index = work_range.GetRangeStart();
           data_index < work_range.GetRangeEnd(); data_index++)
      {
        const GridIndex grid_index =
            combined_sdf.DataIndexToGridIndex(data_index);
        const GridIndex query_index = grid_index + query_offset;

        const SDFScalarType free_sdf_value =
            free_sdf.GetIndexImmutable(query_index).Value();
        const SDFScalarType filled_sdf_value =
            filled_sdf.GetIndexImmutable(query_index).Value();

        SDFScalarType distance = parameters.OOBValue();

        if (free_sdf_value >= 0.0)
        {
          distance = free_sdf_value;
        }
        else if (filled_sdf_value <= -0.0)
        {
          distance = filled_sdf_value;
        }
        else
        {
          distance = 0.0;
        }

        minimum = std::min(minimum, distance);
        maximum = std::max(maximum, distance);

        combined_sdf_raw_data[data_index] = distance;
      }

      per_thread_minimums.at(work_range.GetThreadNum()) = minimum;
      per_thread_maximums.at(work_range.GetThreadNum()) = maximum;
    };

    common_robotics_utilities::parallelism::StaticParallelForRangeLoop(
        parameters.Parallelism(), 0, combined_sdf.NumTotalVoxels(),
        fill_sdf_work,
        common_robotics_utilities::parallelism::ParallelForBackend
            ::BEST_AVAILABLE);

    const SDFScalarType minimum = *std::min_element(
        per_thread_minimums.begin(), per_thread_minimums.end());
    const SDFScalarType maximum = *std::max_element(
        per_thread_maximums.begin(), per_thread_maximums.end());

    const SignedDistanceFieldMinimumMaximum minimum_maximum(minimum, maximum);

    // Lock & update min/max values.
    combined_sdf.LockWithMinimumMaximum(minimum_maximum);
    return combined_sdf;
  }
}
}  // namespace internal
}  // namespace signed_distance_field_generation
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
