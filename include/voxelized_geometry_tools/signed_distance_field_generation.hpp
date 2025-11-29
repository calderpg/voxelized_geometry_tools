#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
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

  // Make two distance fields, one for distance to filled voxels, one for
  // distance to free voxels.
  EDTDistanceField filled_distance_field(
      grid_origin_transform, grid_sizes, unmarked_cell);
  EDTDistanceField free_distance_field(
      grid_origin_transform, grid_sizes, unmarked_cell);

  for (int64_t x_index = 0; x_index < grid_sizes.NumXVoxels(); x_index++)
  {
    for (int64_t y_index = 0; y_index < grid_sizes.NumYVoxels(); y_index++)
    {
      for (int64_t z_index = 0; z_index < grid_sizes.NumZVoxels(); z_index++)
      {
        const GridIndex current_index(x_index, y_index, z_index);
        if (is_filled_fn(current_index))
        {
          filled_distance_field.SetIndex(current_index, marked_cell);
        }
        else
        {
          free_distance_field.SetIndex(current_index, marked_cell);
        }
      }
    }
  }

  // Compute the distance transforms in place
  ComputeDistanceFieldTransformInPlace(
      parameters.Parallelism(), filled_distance_field);
  ComputeDistanceFieldTransformInPlace(
      parameters.Parallelism(), free_distance_field);

  // Generate the SDF
  SignedDistanceField<SDFScalarType> new_sdf(
      grid_origin_transform, frame, grid_sizes, parameters.OOBValue());
  for (int64_t x_index = 0; x_index < new_sdf.NumXVoxels(); x_index++)
  {
    for (int64_t y_index = 0; y_index < new_sdf.NumYVoxels(); y_index++)
    {
      for (int64_t z_index = 0; z_index < new_sdf.NumZVoxels(); z_index++)
      {
        const double filled_distance_squared =
            filled_distance_field.GetIndexImmutable(
                x_index, y_index, z_index).Value();
        const double free_distance_squared =
            free_distance_field.GetIndexImmutable(
                x_index, y_index, z_index).Value();

        const double distance1 =
            std::sqrt(filled_distance_squared) * new_sdf.Resolution();
        const double distance2 =
            std::sqrt(free_distance_squared) * new_sdf.Resolution();
        const double distance = distance1 - distance2;

        new_sdf.SetIndex(
            x_index, y_index, z_index, static_cast<SDFScalarType>(distance));
      }
    }
  }

  // Lock & update min/max values.
  new_sdf.Lock();
  return new_sdf;
}

template<typename T, typename BackingStore, typename SDFScalarType>
inline SignedDistanceField<SDFScalarType> ExtractSignedDistanceField(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& grid,
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
      const GridIndex real_grid_index(
            virtual_border_grid_index.X() - x_axis_query_offset,
            virtual_border_grid_index.Y() - y_axis_query_offset,
            virtual_border_grid_index.Z() - z_axis_query_offset);
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
      const GridIndex real_grid_index(
            virtual_border_grid_index.X() - x_axis_query_offset,
            virtual_border_grid_index.Y() - y_axis_query_offset,
            virtual_border_grid_index.Z() - z_axis_query_offset);
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
    for (int64_t x_idx = 0; x_idx < combined_sdf.NumXVoxels(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.NumYVoxels(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < combined_sdf.NumZVoxels(); z_idx++)
        {
          const int64_t query_x_idx = x_idx + x_axis_query_offset;
          const int64_t query_y_idx = y_idx + y_axis_query_offset;
          const int64_t query_z_idx = z_idx + z_axis_query_offset;
          const SDFScalarType free_sdf_value = free_sdf.GetIndexImmutable(
              query_x_idx, query_y_idx, query_z_idx).Value();
          const SDFScalarType filled_sdf_value = filled_sdf.GetIndexImmutable(
              query_x_idx, query_y_idx, query_z_idx).Value();

          if (free_sdf_value >= 0.0)
          {
            combined_sdf.SetIndex(x_idx, y_idx, z_idx, free_sdf_value);
          }
          else if (filled_sdf_value <= -0.0)
          {
            combined_sdf.SetIndex(x_idx, y_idx, z_idx, filled_sdf_value);
          }
          else
          {
            combined_sdf.SetIndex(x_idx, y_idx, z_idx, 0.0);
          }
        }
      }
    }

    // Lock & update min/max values.
    combined_sdf.Lock();
    return combined_sdf;
  }
}
}  // namespace internal
}  // namespace signed_distance_field_generation
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
