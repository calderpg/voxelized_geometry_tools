#include <voxelized_geometry_tools/signed_distance_field_generation.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/openmp_helpers.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/signed_distance_field.hpp>

namespace voxelized_geometry_tools
{
namespace signed_distance_field_generation
{
namespace internal
{
namespace
{
int32_t GetDirectionNumber(
    const int32_t dx, const int32_t dy, const int32_t dz)
{
  return ((dx + 1) * 9) + ((dy + 1) * 3) + (dz + 1);
}

std::vector<std::vector<std::vector<std::vector<int32_t>>>>
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

double ComputeDistanceSquared(
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

DistanceField BuildDistanceFieldSerial(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes, const std::vector<GridIndex>& points)
{
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
    auto query = distance_field.GetIndexMutable(current_index);
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
        auto neighbor_query = distance_field.GetIndexMutable(
            static_cast<int64_t>(nx), static_cast<int64_t>(ny),
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
  return distance_field;
}

DistanceField BuildDistanceFieldParallel(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points)
{
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
    auto query = distance_field.GetIndexMutable(current_index);
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
          distance_field.GetIndexImmutable(current_index).Value();
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
        auto neighbor_query = distance_field.GetIndexMutable(neighbor_index);
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
  return distance_field;
}
}  // namespace

DistanceField BuildDistanceField(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points,
    const bool use_parallel)
{
  if (!grid_sizes.UniformCellSize())
  {
    throw std::invalid_argument(
        "Cannot build distance field from grid with non-uniform cells");
  }
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
}  // namespace internal
}  // namespace signed_distance_field_generation
}  // namespace voxelized_geometry_tools

template class voxelized_geometry_tools::signed_distance_field_generation
    ::SignedDistanceFieldResult<double>;
template class voxelized_geometry_tools::signed_distance_field_generation
    ::SignedDistanceFieldResult<float>;
