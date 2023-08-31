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

namespace voxelized_geometry_tools
{
namespace signed_distance_field_generation
{
// To allow for faster development/improvements, the SignedDistanceField
// generation API has no stability guarantees and lives in an internal
// namespace. Use the stable API exposed by CollisionMap and
// TaggedObjectCollisionMap instead.
namespace internal
{
using common_robotics_utilities::voxel_grid::GridIndex;
using common_robotics_utilities::voxel_grid::GridSizes;

struct BucketCell
{
  double distance_square = 0.0;
  int32_t update_direction = 0;
  uint32_t location[3] = {0u, 0u, 0u};
  uint32_t closest_point[3] = {0u, 0u, 0u};
};

using DistanceField =
    common_robotics_utilities::voxel_grid::VoxelGrid<BucketCell>;

DistanceField BuildDistanceField(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points,
    const common_robotics_utilities::parallelism::DegreeOfParallelism&
        parallelism);

template<typename SDFScalarType>
inline SignedDistanceField<SDFScalarType> ExtractSignedDistanceField(
    const Eigen::Isometry3d& grid_origin_tranform, const GridSizes& grid_sizes,
    const std::function<bool(const GridIndex&)>& is_filled_fn,
    const std::string& frame,
    const SignedDistanceFieldGenerationParameters<SDFScalarType>& parameters)
{
  std::vector<GridIndex> filled;
  std::vector<GridIndex> free;
  for (int64_t x_index = 0; x_index < grid_sizes.NumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < grid_sizes.NumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < grid_sizes.NumZCells(); z_index++)
      {
        const GridIndex current_index(x_index, y_index, z_index);
        if (is_filled_fn(current_index))
        {
          // Mark as filled
          filled.push_back(current_index);
        }
        else
        {
          // Mark as free space
          free.push_back(current_index);
        }
      }
    }
  }

  // Make two distance fields, one for distance to filled voxels, one for
  // distance to free voxels.
  const DistanceField filled_distance_field = BuildDistanceField(
      grid_origin_tranform, grid_sizes, filled, parameters.Parallelism());
  const DistanceField free_distance_field = BuildDistanceField(
      grid_origin_tranform, grid_sizes, free, parameters.Parallelism());

  // Generate the SDF
  SignedDistanceField<SDFScalarType> new_sdf(
      grid_origin_tranform, frame, grid_sizes, parameters.OOBValue());
  for (int64_t x_index = 0; x_index < new_sdf.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < new_sdf.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < new_sdf.GetNumZCells(); z_index++)
      {
        const double filled_distance_squared =
            filled_distance_field.GetIndexImmutable(
                x_index, y_index, z_index).Value().distance_square;
        const double free_distance_squared =
            free_distance_field.GetIndexImmutable(
                x_index, y_index, z_index).Value().distance_square;

        const double distance1 =
            std::sqrt(filled_distance_squared) * new_sdf.GetResolution();
        const double distance2 =
            std::sqrt(free_distance_squared) * new_sdf.GetResolution();
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
  if (!grid.HasUniformCellSize())
  {
    throw std::invalid_argument("Grid must have uniform resolution");
  }
  if (!parameters.AddVirtualBorder())
  {
    // This is the conventional single-pass result
    return ExtractSignedDistanceField<SDFScalarType>(
        grid.GetOriginTransform(), grid.GetGridSizes(), is_filled_fn, frame,
        parameters);
  }
  else
  {
    const int64_t x_axis_size_offset =
        (grid.GetNumXCells() > 1) ? INT64_C(2) : INT64_C(0);
    const int64_t x_axis_query_offset =
        (grid.GetNumXCells() > 1) ? INT64_C(1) : INT64_C(0);
    const int64_t y_axis_size_offset =
        (grid.GetNumYCells() > 1) ? INT64_C(2) : INT64_C(0);
    const int64_t y_axis_query_offset =
        (grid.GetNumYCells() > 1) ? INT64_C(1) : INT64_C(0);
    const int64_t z_axis_size_offset =
        (grid.GetNumZCells() > 1) ? INT64_C(2) : INT64_C(0);
    const int64_t z_axis_query_offset =
        (grid.GetNumZCells() > 1) ? INT64_C(1) : INT64_C(0);

    // We need to lie about the size of the grid to add a virtual border
    const int64_t num_x_cells = grid.GetNumXCells() + x_axis_size_offset;
    const int64_t num_y_cells = grid.GetNumYCells() + y_axis_size_offset;
    const int64_t num_z_cells = grid.GetNumZCells() + z_axis_size_offset;

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
    const common_robotics_utilities::voxel_grid::GridSizes enlarged_sizes(
        grid.GetCellSizes().x(), num_x_cells, num_y_cells, num_z_cells);
    const auto free_sdf = ExtractSignedDistanceField<SDFScalarType>(
        grid.GetOriginTransform(), enlarged_sizes, free_is_filled_fn, frame,
        parameters);
    const auto filled_sdf = ExtractSignedDistanceField<SDFScalarType>(
        grid.GetOriginTransform(), enlarged_sizes, filled_is_filled_fn, frame,
        parameters);

    // Combine to make a single SDF
    SignedDistanceField<SDFScalarType> combined_sdf(
        grid.GetOriginTransform(), frame, grid.GetGridSizes(),
        parameters.OOBValue());
    for (int64_t x_idx = 0; x_idx < combined_sdf.GetNumXCells(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.GetNumYCells(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < combined_sdf.GetNumZCells(); z_idx++)
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
}  // namespace voxelized_geometry_tools
