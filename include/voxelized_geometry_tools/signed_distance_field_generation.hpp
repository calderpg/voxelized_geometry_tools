#pragma once

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/maybe.hpp>
#include <common_robotics_utilities/openmp_helpers.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>

namespace voxelized_geometry_tools
{
namespace signed_distance_field_generation
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

typedef common_robotics_utilities::voxel_grid
    ::VoxelGrid<BucketCell, std::vector<BucketCell>> DistanceField;

inline int32_t GetDirectionNumber(
    const int32_t dx, const int32_t dy, const int32_t dz)
{
  return ((dx + 1) * 9) + ((dy + 1) * 3) + (dz + 1);
}

inline std::vector<std::vector<std::vector<std::vector<int32_t>>>>
MakeNeighborhoods()
{
  // First vector<>: 2 - the first bucket queue, the points we know are zero
  // distance, start with a complete set of neighbors to check. Every other
  // bucket queue checks fewer neighbors.
  // Second vector<>: 27 (# of source directions in fully-connected 3d grid).
  // Third vector<>:
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> neighborhoods;
  // I don't know why there are 2 initial neighborhoods.
  neighborhoods.resize(2);
  for (size_t n = 0; n < neighborhoods.size(); n++)
  {
    neighborhoods[n].resize(27);
    // Loop through the source directions.
    for (int32_t dx = -1; dx <= 1; dx++)
    {
      for (int32_t dy = -1; dy <= 1; dy++)
      {
        for (int32_t dz = -1; dz <= 1; dz++)
        {
          const int32_t direction_number = GetDirectionNumber(dx, dy, dz);
          // Loop through the target directions.
          for (int32_t tdx = -1; tdx <= 1; tdx++)
          {
            for (int32_t tdy = -1; tdy <= 1; tdy++)
            {
              for (int32_t tdz = -1; tdz <= 1; tdz++)
              {
                // Ignore the case of ourself.
                if (tdx == 0 && tdy == 0 && tdz == 0)
                {
                  continue;
                }
                // Why is one set of neighborhoods larger than the other?
                if (n >= 1)
                {
                  if ((abs(tdx) + abs(tdy) + abs(tdz)) != 1)
                  {
                    continue;
                  }
                  if ((dx * tdx) < 0 || (dy * tdy) < 0 || (dz * tdz) < 0)
                  {
                    continue;
                  }
                }
                std::vector<int32_t> new_point;
                new_point.resize(3);
                new_point[0] = tdx;
                new_point[1] = tdy;
                new_point[2] = tdz;
                neighborhoods[n][direction_number].push_back(new_point);
              }
            }
          }
        }
      }
    }
  }
  return neighborhoods;
}

inline double ComputeDistanceSquared(
    const int32_t x1, const int32_t y1, const int32_t z1,
    const int32_t x2, const int32_t y2, const int32_t z2)
{
  const int32_t dx = x1 - x2;
  const int32_t dy = y1 - y2;
  const int32_t dz = z1 - z2;
  return double((dx * dx) + (dy * dy) + (dz * dz));
}

class MultipleThreadIndexQueueWrapper
{
public:

  explicit MultipleThreadIndexQueueWrapper(const size_t max_queues)
  {
    per_thread_queues_.resize(
        common_robotics_utilities::openmp_helpers::GetNumOmpThreads(),
        ThreadIndexQueues(max_queues));
  }

  const GridIndex& Query(const int32_t distance_squared, const size_t idx) const
  {
    size_t working_index = idx;
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      const auto& current_thread_queue =
          per_thread_queues_.at(thread).at(distance_squared);
      const size_t current_thread_queue_size = current_thread_queue.size();
      if (working_index < current_thread_queue_size)
      {
        return current_thread_queue.at(working_index);
      }
      else
      {
        working_index -= current_thread_queue_size;
      }
    }
    throw std::runtime_error("Failed to find item");
  }

  size_t NumQueues() const
  {
    return per_thread_queues_.at(0).size();
  }

  size_t Size(const int32_t distance_squared) const
  {
    size_t total_size = 0;
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      total_size += per_thread_queues_.at(thread).at(distance_squared).size();
    }
    return total_size;
  }

  void Enqueue(const int32_t distance_squared, const GridIndex& index)
  {
    const int32_t thread_num
        = common_robotics_utilities::openmp_helpers::GetContextOmpThreadNum();
    per_thread_queues_.at(thread_num).at(distance_squared).push_back(index);
  }

  void ClearCompletedQueues(const int32_t distance_squared)
  {
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      per_thread_queues_.at(thread).at(distance_squared).clear();
    }
  }

private:
  typedef std::vector<std::vector<GridIndex>> ThreadIndexQueues;
  std::vector<ThreadIndexQueues> per_thread_queues_;

};

inline DistanceField BuildDistanceFieldSerial(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes, const std::vector<GridIndex>& points)
{
  if (!grid_sizes.UniformCellSize())
  {
    throw std::invalid_argument(
        "Cannot build distance field from grid with non-uniform cells");
  }
  const std::chrono::time_point<std::chrono::steady_clock> start_time
      = std::chrono::steady_clock::now();
  // Make the DistanceField container
  BucketCell default_cell;
  default_cell.distance_square = std::numeric_limits<double>::infinity();
  DistanceField distance_field(grid_origin_transform, grid_sizes, default_cell);
  // Compute maximum distance square
  const int64_t max_distance_square =
      (distance_field.GetNumXCells() * distance_field.GetNumXCells())
      + (distance_field.GetNumYCells() * distance_field.GetNumYCells())
      + (distance_field.GetNumZCells() * distance_field.GetNumZCells());
  // Make bucket queue
  std::vector<std::vector<BucketCell>> bucket_queue(max_distance_square + 1);
  bucket_queue[0].reserve(points.size());
  // Set initial update direction
  int32_t initial_update_direction = GetDirectionNumber(0, 0, 0);
  // Mark all provided points with distance zero and add to the bucket queue
  for (size_t index = 0; index < points.size(); index++)
  {
    const GridIndex& current_index = points[index];
    auto query = distance_field.GetMutable(current_index);
    if (query)
    {
      query.Value().location[0] = static_cast<uint32_t>(current_index.X());
      query.Value().location[1] = static_cast<uint32_t>(current_index.Y());
      query.Value().location[2] = static_cast<uint32_t>(current_index.Z());
      query.Value().closest_point[0] = static_cast<uint32_t>(current_index.X());
      query.Value().closest_point[1] = static_cast<uint32_t>(current_index.Y());
      query.Value().closest_point[2] = static_cast<uint32_t>(current_index.Z());
      query.Value().distance_square = 0.0;
      query.Value().update_direction = initial_update_direction;
      bucket_queue[0].push_back(query.Value());
    }
    // If the point is outside the bounds of the SDF, skip
    else
    {
      throw std::runtime_error("Point for BuildDistanceField out of bounds");
    }
  }
  // HERE BE DRAGONS
  // Process the bucket queue
  const std::vector<std::vector<std::vector<std::vector<int>>>> neighborhoods =
      MakeNeighborhoods();
  for (size_t bq_idx = 0; bq_idx < bucket_queue.size(); bq_idx++)
  {
    for (const auto& cur_cell : bucket_queue[bq_idx])
    {
      // Get the current location
      const double x = cur_cell.location[0];
      const double y = cur_cell.location[1];
      const double z = cur_cell.location[2];
      // Pick the update direction
      // Only the first bucket queue gets the larger set of neighborhoods?
      // Don't really userstand why.
      const size_t direction_switch = (bq_idx > 0) ? 1 : 0;
      // Make sure the update direction is valid
      if (cur_cell.update_direction < 0 || cur_cell.update_direction > 26)
      {
        continue;
      }
      // Get the current neighborhood list
      const std::vector<std::vector<int>>& neighborhood =
          neighborhoods[direction_switch][cur_cell.update_direction];
      // Update the distance from the neighboring cells
      for (size_t nh_idx = 0; nh_idx < neighborhood.size(); nh_idx++)
      {
        // Get the direction to check
        const int32_t dx = neighborhood[nh_idx][0];
        const int32_t dy = neighborhood[nh_idx][1];
        const int32_t dz = neighborhood[nh_idx][2];
        const int32_t nx = static_cast<int32_t>(x + dx);
        const int32_t ny = static_cast<int32_t>(y + dy);
        const int32_t nz = static_cast<int32_t>(z + dz);
        auto neighbor_query =
            distance_field.GetMutable(static_cast<int64_t>(nx),
                                      static_cast<int64_t>(ny),
                                      static_cast<int64_t>(nz));
        if (!neighbor_query)
        {
          // "Neighbor" is outside the bounds of the SDF
          continue;
        }
        // Update the neighbor's distance based on the current
        const int32_t new_distance_square =
            static_cast<int32_t>(ComputeDistanceSquared(
                                   nx, ny, nz,
                                   cur_cell.closest_point[0],
                                   cur_cell.closest_point[1],
                                   cur_cell.closest_point[2]));
        if (new_distance_square > max_distance_square)
        {
          // Skip these cases
          continue;
        }
        if (new_distance_square < neighbor_query.Value().distance_square)
        {
          // If the distance is better, time to update the neighbor
          neighbor_query.Value().distance_square = new_distance_square;
          neighbor_query.Value().closest_point[0] = cur_cell.closest_point[0];
          neighbor_query.Value().closest_point[1] = cur_cell.closest_point[1];
          neighbor_query.Value().closest_point[2] = cur_cell.closest_point[2];
          neighbor_query.Value().location[0] = nx;
          neighbor_query.Value().location[1] = ny;
          neighbor_query.Value().location[2] = nz;
          neighbor_query.Value().update_direction =
              GetDirectionNumber(dx, dy, dz);
          // Add the neighbor into the bucket queue
          bucket_queue[new_distance_square].push_back(neighbor_query.Value());
        }
      }
    }
    // Clear the current queue now that we're done with it
    bucket_queue[bq_idx].clear();
  }
  const std::chrono::time_point<std::chrono::steady_clock> end_time
      = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Computed DistanceField in " << elapsed.count() << " seconds"
            << std::endl;
  return distance_field;
}

inline DistanceField BuildDistanceFieldParallel(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points)
{
  if (!grid_sizes.UniformCellSize())
  {
    throw std::invalid_argument(
        "Cannot build distance field from grid with non-uniform cells");
  }
  const std::chrono::time_point<std::chrono::steady_clock> start_time
      = std::chrono::steady_clock::now();
  // Make the DistanceField container
  BucketCell default_cell;
  default_cell.distance_square = std::numeric_limits<double>::infinity();
  DistanceField distance_field(grid_origin_transform, grid_sizes, default_cell);
  // Compute maximum distance square
  const int64_t max_distance_square =
      (distance_field.GetNumXCells() * distance_field.GetNumXCells())
      + (distance_field.GetNumYCells() * distance_field.GetNumYCells())
      + (distance_field.GetNumZCells() * distance_field.GetNumZCells());
  // Make bucket queue
  std::vector<std::vector<BucketCell>> bucket_queue(max_distance_square + 1);
  bucket_queue[0].reserve(points.size());
  MultipleThreadIndexQueueWrapper bucket_queues(max_distance_square + 1);
  // Set initial update direction
  int32_t initial_update_direction = GetDirectionNumber(0, 0, 0);
  // Mark all provided points with distance zero and add to the bucket queues
  // points MUST NOT CONTAIN DUPLICATE ENTRIES!
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (size_t index = 0; index < points.size(); index++)
  {
    const GridIndex& current_index = points[index];
    auto query = distance_field.GetMutable(current_index);
    if (query)
    {
      query.Value().location[0] = static_cast<uint32_t>(current_index.X());
      query.Value().location[1] = static_cast<uint32_t>(current_index.Y());
      query.Value().location[2] = static_cast<uint32_t>(current_index.Z());
      query.Value().closest_point[0] = static_cast<uint32_t>(current_index.X());
      query.Value().closest_point[1] = static_cast<uint32_t>(current_index.Y());
      query.Value().closest_point[2] = static_cast<uint32_t>(current_index.Z());
      query.Value().distance_square = 0.0;
      query.Value().update_direction = initial_update_direction;
      bucket_queues.Enqueue(0, current_index);
    }
    // If the point is outside the bounds of the SDF, skip
    else
    {
      throw std::runtime_error("Point for BuildDistanceField out of bounds");
    }
  }
  // HERE BE DRAGONS
  // Process the bucket queue
  const std::vector<std::vector<std::vector<std::vector<int>>>> neighborhoods =
      MakeNeighborhoods();
  for (int32_t current_distance_square = 0;
       current_distance_square
           < static_cast<int32_t>(bucket_queues.NumQueues());
       current_distance_square++)
  {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < bucket_queues.Size(current_distance_square);
         idx++)
    {
      const GridIndex& current_index =
          bucket_queues.Query(current_distance_square, idx);
      // Get the current location
      const BucketCell& cur_cell =
          distance_field.GetImmutable(current_index).Value();
      const double x = cur_cell.location[0];
      const double y = cur_cell.location[1];
      const double z = cur_cell.location[2];
      // Pick the update direction
      // Only the first bucket queue gets the larger set of neighborhoods?
      // Don't really userstand why.
      const size_t direction_switch = (current_distance_square > 0) ? 1 : 0;
      // Make sure the update direction is valid
      if (cur_cell.update_direction < 0 || cur_cell.update_direction > 26)
      {
        continue;
      }
      // Get the current neighborhood list
      const std::vector<std::vector<int32_t>>& neighborhood =
          neighborhoods[direction_switch][cur_cell.update_direction];
      // Update the distance from the neighboring cells
      for (size_t nh_idx = 0; nh_idx < neighborhood.size(); nh_idx++)
      {
        // Get the direction to check
        const int32_t dx = neighborhood[nh_idx][0];
        const int32_t dy = neighborhood[nh_idx][1];
        const int32_t dz = neighborhood[nh_idx][2];
        const int32_t nx = static_cast<int32_t>(x + dx);
        const int32_t ny = static_cast<int32_t>(y + dy);
        const int32_t nz = static_cast<int32_t>(z + dz);
        const GridIndex neighbor_index(static_cast<int64_t>(nx),
                                       static_cast<int64_t>(ny),
                                       static_cast<int64_t>(nz));
        auto neighbor_query = distance_field.GetMutable(neighbor_index);
        if (!neighbor_query)
        {
          // "Neighbor" is outside the bounds of the SDF
          continue;
        }
        // Update the neighbor's distance based on the current
        const int32_t new_distance_square =
            static_cast<int32_t>(ComputeDistanceSquared(
                                   nx, ny, nz,
                                   cur_cell.closest_point[0],
                                   cur_cell.closest_point[1],
                                   cur_cell.closest_point[2]));
        if (new_distance_square > max_distance_square)
        {
          // Skip these cases
          continue;
        }
        if (new_distance_square < neighbor_query.Value().distance_square)
        {
          // If the distance is better, time to update the neighbor
          neighbor_query.Value().distance_square = new_distance_square;
          neighbor_query.Value().closest_point[0] = cur_cell.closest_point[0];
          neighbor_query.Value().closest_point[1] = cur_cell.closest_point[1];
          neighbor_query.Value().closest_point[2] = cur_cell.closest_point[2];
          neighbor_query.Value().location[0] = nx;
          neighbor_query.Value().location[1] = ny;
          neighbor_query.Value().location[2] = nz;
          neighbor_query.Value().update_direction =
              GetDirectionNumber(dx, dy, dz);
          // Add the neighbor into the bucket queue
          bucket_queues.Enqueue(new_distance_square, neighbor_index);
        }
      }
    }
    // Clear the current queues now that we're done with it
    bucket_queues.ClearCompletedQueues(current_distance_square);
  }
  const std::chrono::time_point<std::chrono::steady_clock> end_time
      = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Computed DistanceField in " << elapsed.count() << " seconds"
            << std::endl;
  return distance_field;
}

inline DistanceField BuildDistanceField(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points,
    const bool use_parallel)
{
  if (use_parallel)
  {
    return BuildDistanceFieldParallel(
        grid_origin_transform, grid_sizes, points);
  }
  else
  {
    return BuildDistanceFieldSerial(grid_origin_transform, grid_sizes, points);
  }
}

template<typename SDFBackingStore>
class SignedDistanceFieldResult
{
private:
  SignedDistanceField<SDFBackingStore> distance_field_;
  double maximum_ = 0.0;
  double minimum_ = 0.0;

public:
  SignedDistanceFieldResult(
      const SignedDistanceField<SDFBackingStore>& distance_field,
      const double maximum, const double minimum)
      : distance_field_(distance_field),
        maximum_(maximum), minimum_(minimum)
  {
    if (minimum_ > maximum_)
    {
      throw std::invalid_argument("minimum_ > maximum_");
    }
  }

  const SignedDistanceField<SDFBackingStore>& DistanceField() const
  {
    return distance_field_;
  }

  const SignedDistanceField<SDFBackingStore>& MutableDistanceField()
  {
    return distance_field_;
  }

  double Maximum() const { return maximum_; }

  double Minimum() const { return minimum_; }
};

template<typename SDFBackingStore>
SignedDistanceFieldResult<SDFBackingStore> MakeSignedDistanceFieldResult(
    const SignedDistanceField<SDFBackingStore>& signed_distance_field,
    const double maximum, const double minimum)
{
  return SignedDistanceFieldResult<SDFBackingStore>(
      signed_distance_field, maximum, minimum);
}


template<typename T, typename SDFBackingStore=std::vector<float>>
inline SignedDistanceFieldResult<SDFBackingStore> ExtractSignedDistanceField(
    const Eigen::Isometry3d& grid_origin_tranform, const GridSizes& grid_sizes,
    const std::function<bool(const GridIndex&)>& is_filled_fn,
    const float oob_value, const std::string& frame, const bool use_parallel)
{
  const std::chrono::time_point<std::chrono::steady_clock> start_time
      = std::chrono::steady_clock::now();
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
  const DistanceField filled_distance_field =
      BuildDistanceField(
        grid_origin_tranform, grid_sizes, filled, use_parallel);
  const DistanceField free_distance_field =
      BuildDistanceField(grid_origin_tranform, grid_sizes, free, use_parallel);
  // Generate the SDF
  SignedDistanceField<SDFBackingStore> new_sdf(
      grid_origin_tranform, frame, grid_sizes, oob_value);
  double max_distance = -std::numeric_limits<double>::infinity();
  double min_distance = std::numeric_limits<double>::infinity();
  for (int64_t x_index = 0; x_index < new_sdf.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < new_sdf.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < new_sdf.GetNumZCells(); z_index++)
      {
        const double distance1 =
            std::sqrt(
                filled_distance_field.GetImmutable(x_index, y_index, z_index)
                    .Value().distance_square)
            * new_sdf.GetResolution();
        const double distance2 =
            std::sqrt(
                free_distance_field.GetImmutable(x_index, y_index, z_index)
                    .Value().distance_square)
            * new_sdf.GetResolution();
        const double distance = distance1 - distance2;
        if (distance > max_distance)
        {
          max_distance = distance;
        }
        if (distance < min_distance)
        {
          min_distance = distance;
        }
        new_sdf.SetValue(x_index, y_index, z_index, (float)distance);
      }
    }
  }
  const std::chrono::time_point<std::chrono::steady_clock> end_time
      = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Computed SDF for grid size in " << elapsed.count() << " seconds"
            << std::endl;
  return MakeSignedDistanceFieldResult<SDFBackingStore>(
      new_sdf, max_distance, min_distance);
}

template<typename T, typename BackingStore=std::vector<T>,
         typename SDFBackingStore=std::vector<float>>
inline SignedDistanceFieldResult<SDFBackingStore> ExtractSignedDistanceField(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& grid,
    const std::function<bool(const GridIndex&)>& is_filled_fn,
    const float oob_value, const std::string& frame,
    const bool use_parallel, const bool add_virtual_border)
{
  if (!grid.HasUniformCellSize())
  {
    throw std::invalid_argument("Grid must have uniform resolution");
  }
  if (add_virtual_border == false)
  {
    // This is the conventional single-pass result
    return ExtractSignedDistanceField<T, SDFBackingStore>(
        grid.GetOriginTransform(), grid.GetGridSizes(), is_filled_fn, oob_value,
        frame, use_parallel);
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
    auto free_sdf_result
        = ExtractSignedDistanceField<T>(
            grid.GetOriginTransform(), enlarged_sizes, free_is_filled_fn,
            oob_value, frame, use_parallel);
    auto filled_sdf_result
        = ExtractSignedDistanceField<T>(
            grid.GetOriginTransform(), enlarged_sizes, filled_is_filled_fn,
            oob_value, frame, use_parallel);
    // Combine to make a single SDF
    SignedDistanceField<SDFBackingStore> combined_sdf(
          grid.GetOriginTransform(), frame, grid.GetGridSizes(), oob_value);
    for (int64_t x_idx = 0; x_idx < combined_sdf.GetNumXCells(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.GetNumYCells(); y_idx++)
      {
        for (int64_t z_idx = 0; z_idx < combined_sdf.GetNumZCells(); z_idx++)
        {
          const int64_t query_x_idx = x_idx + x_axis_query_offset;
          const int64_t query_y_idx = y_idx + y_axis_query_offset;
          const int64_t query_z_idx = z_idx + z_axis_query_offset;
          const float free_sdf_value
              = free_sdf_result.DistanceField().GetImmutable(
                  query_x_idx, query_y_idx, query_z_idx).Value();
          const float filled_sdf_value
              = filled_sdf_result.DistanceField().GetImmutable(
                  query_x_idx, query_y_idx, query_z_idx).Value();
          if (free_sdf_value >= 0.0)
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, free_sdf_value);
          }
          else if (filled_sdf_value <= -0.0)
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, filled_sdf_value);
          }
          else
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, 0.0f);
          }
        }
      }
    }
    // Get the combined max/min values
    return MakeSignedDistanceFieldResult<SDFBackingStore>(
        combined_sdf, free_sdf_result.Maximum(), filled_sdf_result.Minimum());
  }
}

template<typename T, typename BackingStore=std::vector<T>,
         typename SDFBackingStore=std::vector<float>>
inline SignedDistanceFieldResult<SDFBackingStore> ExtractSignedDistanceField(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& grid,
    const std::function<bool(const T&)>& is_filled_fn,
    const float oob_value, const std::string& frame,
    const bool use_parallel)
{
  if (!grid.HasUniformCellSize())
  {
    throw std::invalid_argument("Grid must have uniform resolution");
  }
  const std::function<bool(const GridIndex&)> real_is_filled_fn =
      [&] (const GridIndex& index)
  {
    const T& stored = grid.GetImmutable(index).Value();
    // If it matches an object to use OR there are no objects supplied
    if (is_filled_fn(stored))
    {
      // Mark as filled
      return true;
    }
    else
    {
      // Mark as free space
      return false;
    }
  };
  return ExtractSignedDistanceField<T, BackingStore, SDFBackingStore>(
        grid, real_is_filled_fn, oob_value, frame, use_parallel, false);
}
}  // namespace signed_distance_field_generation
}  // namespace voxelized_geometry_tools
