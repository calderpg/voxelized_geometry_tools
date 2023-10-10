#include <voxelized_geometry_tools/signed_distance_field_generation.hpp>

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

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForLoop;
using common_robotics_utilities::parallelism::ThreadWorkRange;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using VectorXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
using MatrixXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

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
          const size_t direction_number =
              static_cast<size_t>(GetDirectionNumber(dx, dy, dz));
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
                  if ((std::abs(tdx) + std::abs(tdy) + std::abs(tdz)) != 1)
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

  MultipleThreadIndexQueueWrapper(
      const size_t num_threads, const size_t max_queues)
  {
    per_thread_queues_.resize(num_threads, ThreadIndexQueues(max_queues));
  }

  const GridIndex& Query(const int32_t distance_squared, const size_t idx) const
  {
    size_t working_index = idx;
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      const auto& current_thread_queue =
          per_thread_queues_.at(thread).at(
              static_cast<size_t>(distance_squared));
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
      total_size += per_thread_queues_.at(thread).at(
          static_cast<size_t>(distance_squared)).size();
    }
    return total_size;
  }

  void Enqueue(const int32_t thread_num, const int32_t distance_squared,
               const GridIndex& index)
  {
    per_thread_queues_.at(static_cast<size_t>(thread_num)).at(
        static_cast<size_t>(distance_squared)).push_back(index);
  }

  void ClearCompletedQueues(const int32_t distance_squared)
  {
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      per_thread_queues_.at(thread).at(
          static_cast<size_t>(distance_squared)).clear();
    }
  }

private:
  using ThreadIndexQueues = std::vector<std::vector<GridIndex>>;
  std::vector<ThreadIndexQueues> per_thread_queues_;

};

DistanceField BuildDistanceFieldBucketQueueSerial(
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
  std::vector<std::vector<BucketCell>> bucket_queue(
      static_cast<size_t>(max_distance_square + 1));
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
      const size_t update_direction =
          static_cast<size_t>(cur_cell.update_direction);
      const std::vector<std::vector<int>>& neighborhood =
          neighborhoods[direction_switch][update_direction];
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
                static_cast<int32_t>(cur_cell.closest_point[0]),
                static_cast<int32_t>(cur_cell.closest_point[1]),
                static_cast<int32_t>(cur_cell.closest_point[2])));
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
          neighbor_query.Value().location[0] = static_cast<uint32_t>(nx);
          neighbor_query.Value().location[1] = static_cast<uint32_t>(ny);
          neighbor_query.Value().location[2] = static_cast<uint32_t>(nz);
          neighbor_query.Value().update_direction =
              GetDirectionNumber(dx, dy, dz);
          // Add the neighbor into the bucket queue
          bucket_queue[static_cast<size_t>(new_distance_square)].push_back(
              neighbor_query.Value());
        }
      }
    }
    // Clear the current queue now that we're done with it
    bucket_queue[bq_idx].clear();
  }
  return distance_field;
}

DistanceField BuildDistanceFieldBucketQueueParallel(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points,
    const DegreeOfParallelism& parallelism)
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
  std::vector<std::vector<BucketCell>> bucket_queue(
      static_cast<size_t>(max_distance_square + 1));
  bucket_queue[0].reserve(points.size());
  MultipleThreadIndexQueueWrapper bucket_queues(
      static_cast<size_t>(parallelism.GetNumThreads()),
      static_cast<size_t>(max_distance_square + 1));

  // Set initial update direction
  int32_t initial_update_direction = GetDirectionNumber(0, 0, 0);

  // Mark all provided points with distance zero and add to the bucket queues
  // points MUST NOT CONTAIN DUPLICATE ENTRIES!
  const auto mark_thread_work = [&](const ThreadWorkRange& work_range)
  {
    for (size_t index = static_cast<size_t>(work_range.GetRangeStart());
         index < static_cast<size_t>(work_range.GetRangeEnd());
         index++)
    {
      const GridIndex& current_index = points[index];
      auto query = distance_field.GetIndexMutable(current_index);
      if (query)
      {
        query.Value().location[0] = static_cast<uint32_t>(current_index.X());
        query.Value().location[1] = static_cast<uint32_t>(current_index.Y());
        query.Value().location[2] = static_cast<uint32_t>(current_index.Z());
        query.Value().closest_point[0] =
            static_cast<uint32_t>(current_index.X());
        query.Value().closest_point[1] =
            static_cast<uint32_t>(current_index.Y());
        query.Value().closest_point[2] =
            static_cast<uint32_t>(current_index.Z());
        query.Value().distance_square = 0.0;
        query.Value().update_direction = initial_update_direction;
        bucket_queues.Enqueue(work_range.GetThreadNum(), 0, current_index);
      }
      // If the point is outside the bounds of the SDF, skip
      else
      {
        throw std::runtime_error("Point for BuildDistanceField out of bounds");
      }
    }
  };

  StaticParallelForLoop(
      parallelism, 0, static_cast<int64_t>(points.size()), mark_thread_work,
      ParallelForBackend::BEST_AVAILABLE);

  // HERE BE DRAGONS
  // Process the bucket queue
  const std::vector<std::vector<std::vector<std::vector<int>>>> neighborhoods =
      MakeNeighborhoods();
  for (int32_t current_distance_square = 0;
       current_distance_square
           < static_cast<int32_t>(bucket_queues.NumQueues());
       current_distance_square++)
  {
    const auto process_thread_work = [&](const ThreadWorkRange& work_range)
    {
      for (size_t idx = static_cast<size_t>(work_range.GetRangeStart());
           idx < static_cast<size_t>(work_range.GetRangeEnd());
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
        const size_t update_direction =
            static_cast<size_t>(cur_cell.update_direction);
        const std::vector<std::vector<int32_t>>& neighborhood =
            neighborhoods[direction_switch][update_direction];
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
                  static_cast<int32_t>(cur_cell.closest_point[0]),
                  static_cast<int32_t>(cur_cell.closest_point[1]),
                  static_cast<int32_t>(cur_cell.closest_point[2])));
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
            neighbor_query.Value().location[0] = static_cast<uint32_t>(nx);
            neighbor_query.Value().location[1] = static_cast<uint32_t>(ny);
            neighbor_query.Value().location[2] = static_cast<uint32_t>(nz);
            neighbor_query.Value().update_direction =
                GetDirectionNumber(dx, dy, dz);
            // Add the neighbor into the bucket queue
            bucket_queues.Enqueue(
                work_range.GetThreadNum(), new_distance_square, neighbor_index);
          }
        }
      }
    };

    StaticParallelForLoop(
        parallelism, 0,
        static_cast<int64_t>(bucket_queues.Size(current_distance_square)),
        process_thread_work, ParallelForBackend::BEST_AVAILABLE);

    // Clear the current queues now that we're done with it
    bucket_queues.ClearCompletedQueues(current_distance_square);
  }
  return distance_field;
}
}  // namespace

DistanceField BuildDistanceFieldBucketQueue(
    const Eigen::Isometry3d& grid_origin_transform,
    const GridSizes& grid_sizes,
    const std::vector<GridIndex>& points,
    const DegreeOfParallelism& parallelism)
{
  if (!grid_sizes.UniformCellSize())
  {
    throw std::invalid_argument(
        "Cannot build distance field from grid with non-uniform cells");
  }
  if (parallelism.IsParallel())
  {
    return BuildDistanceFieldBucketQueueParallel(
        grid_origin_transform, grid_sizes, points, parallelism);
  }
  else
  {
    return BuildDistanceFieldBucketQueueSerial(
        grid_origin_transform, grid_sizes, points);
  }
}

namespace
{
class XIndexer
{
public:
  XIndexer(const int64_t y_index, const int64_t z_index)
      : y_index_(y_index), z_index_(z_index) {}

  GridIndex operator()(const int64_t x_index) const
  {
    return GridIndex(x_index, y_index_, z_index_);
  }

private:
  const int64_t y_index_;
  const int64_t z_index_;
};

class YIndexer
{
public:
  YIndexer(const int64_t x_index, const int64_t z_index)
      : x_index_(x_index), z_index_(z_index) {}

  GridIndex operator()(const int64_t y_index) const
  {
    return GridIndex(x_index_, y_index, z_index_);
  }

private:
  const int64_t x_index_;
  const int64_t z_index_;
};

class ZIndexer
{
public:
  ZIndexer(const int64_t x_index, const int64_t y_index)
      : x_index_(x_index), y_index_(y_index) {}

  GridIndex operator()(const int64_t z_index) const
  {
    return GridIndex(x_index_, y_index_, z_index);
  }

private:
  const int64_t x_index_;
  const int64_t y_index_;
};

template<typename Indexer>
void ComputeOneDimensionDistanceTransformInPlaceBruteForce(
    const int64_t num_elements, const Indexer& indexer,
    Eigen::Ref<VectorXd> scratch_d, EDTDistanceField& distance_field)
{
  const auto f = [&](const int64_t element)
  {
    return distance_field.GetIndexImmutable(indexer(element)).Value();
  };

  const auto set_f = [&](const int64_t element, const double value)
  {
    return distance_field.SetIndex(indexer(element), value);
  };

  const auto square =
      [](const int64_t value) { return static_cast<double>(value * value); };

  // Reset elements in scratch space
  scratch_d.setConstant(std::numeric_limits<double>::infinity());

  for (int64_t q = 0; q < num_elements; q++)
  {
    for (int64_t other = 0; other < num_elements; other++)
    {
      const double dist = square(q - other) + f(other);
      if (dist < scratch_d(q))
      {
        scratch_d(q) = dist;
      }
    }
  }

  for (int64_t q = 0; q < num_elements; q++)
  {
    set_f(q, scratch_d(q));
  }
}

template<typename Indexer>
void ComputeOneDimensionDistanceTransformInPlaceLinear(
    const int64_t num_elements, const Indexer& indexer,
    Eigen::Ref<VectorXd> scratch_z, Eigen::Ref<VectorXi64> scratch_v,
    Eigen::Ref<VectorXd> scratch_d, EDTDistanceField& distance_field)
{
  const auto f = [&](const int64_t element)
  {
    return distance_field.GetIndexImmutable(indexer(element)).Value();
  };

  const auto set_f = [&](const int64_t element, const double value)
  {
    return distance_field.SetIndex(indexer(element), value);
  };

  const auto square =
      [](const int64_t value) { return static_cast<double>(value * value); };

  // Reset elements in scratch space
  scratch_z.setZero();
  scratch_v.setZero();
  scratch_d.setZero();

  // Initialize
  scratch_z(0) = -std::numeric_limits<double>::infinity();
  scratch_z(1) = std::numeric_limits<double>::infinity();

  // Helper to compute subtraction without producing NaN values.
  const auto sub = [](const double a, const double b)
  {
    constexpr double infinity = std::numeric_limits<double>::infinity();
    if (a == infinity && b == infinity)
    {
      return 0.0;
    }
    else if (a == infinity)
    {
      return infinity;
    }
    else if (b == infinity)
    {
      return -infinity;
    }
    else
    {
      return a - b;
    }
  };

  // Helper for computing intermediate value s
  const auto compute_s = [&](const int64_t q, const int64_t k)
  {
    const int64_t v_k = scratch_v(k);
    const double fq = f(q);
    const double fv_k = f(v_k);
    const double top = sub((fq + square(q)), (fv_k + square(v_k)));
    const double bottom = static_cast<double>((2 * q) - (2 * v_k));
    const double s = top / bottom;
    return s;
  };

  // Phase 1
  {
    int64_t k = 0;

    for (int64_t q = 1; q < num_elements; q++)
    {
      double s = compute_s(q, k);
      while (k > 0 && s <= scratch_z(k))
      {
        k--;
        s = compute_s(q, k);
      }

      k++;
      scratch_v(k) = q;
      scratch_z(k) = s;
      scratch_z(k + 1) = std::numeric_limits<double>::infinity();
    }
  }

  // Phase 2
  {
    int64_t k = 0;

    for (int q = 0; q < num_elements; q++)
    {
      while (scratch_z(k + 1) < q)
      {
        k++;
      }

      const int64_t v_k = scratch_v(k);
      scratch_d(q) = square(q - v_k) + f(v_k);
    }

    for (int q = 0; q < num_elements; q++)
    {
      set_f(q, scratch_d(q));
    }
  }
}

template<typename Indexer>
void ComputeOneDimensionDistanceTransformInPlace(
    const int64_t num_elements, const Indexer& indexer,
    Eigen::Ref<VectorXd> scratch_z, Eigen::Ref<VectorXi64> scratch_v,
    Eigen::Ref<VectorXd> scratch_d, EDTDistanceField& distance_field)
{
  // We expect the brute force O(num_elements^2) strategy to be faster for small
  // numbers of elements than the more complex O(num_elements) strategy.
  constexpr int64_t kStrategyThreshold = 8;

  if (num_elements > kStrategyThreshold)
  {
    ComputeOneDimensionDistanceTransformInPlaceLinear(
        num_elements, indexer, scratch_z, scratch_v, scratch_d, distance_field);
  }
  else
  {
    ComputeOneDimensionDistanceTransformInPlaceBruteForce(
        num_elements, indexer, scratch_d, distance_field);
  }
}
}  // namespace


void ComputeDistanceFieldTransformInPlace(
    const DegreeOfParallelism& parallelism, EDTDistanceField& distance_field)
{
  const int64_t num_x_cells = distance_field.GetNumXCells();
  const int64_t num_y_cells = distance_field.GetNumYCells();
  const int64_t num_z_cells = distance_field.GetNumZCells();

  const auto get_indicies_from_iteration = [&](
      const int64_t step, const int64_t iteration)
  {
    const int64_t first_index = iteration / step;
    const int64_t second_index = iteration % step;
    return std::make_pair(first_index, second_index);
  };

  const int32_t num_threads = parallelism.GetNumThreads();

  // Transform along X axis
  if (num_x_cells > 1)
  {
    // Allocate scratch space (don't need to initialize, since transform
    // functions reset their scratch space as needed).

    // Note, z requires an additional element.
    MatrixXd full_scratch_z(num_x_cells + 1, num_threads);
    MatrixXi64 full_scratch_v(num_x_cells, num_threads);
    MatrixXd full_scratch_d(num_x_cells, num_threads);

    const auto thread_work = [&](const ThreadWorkRange& work_range)
    {
      const int32_t thread_num = work_range.GetThreadNum();
      auto scratch_z = full_scratch_z.col(thread_num);
      auto scratch_v = full_scratch_v.col(thread_num);
      auto scratch_d = full_scratch_d.col(thread_num);

      for (int64_t iteration = work_range.GetRangeStart();
           iteration < work_range.GetRangeEnd();
           iteration++)
      {
        const auto indices =
            get_indicies_from_iteration(num_y_cells, iteration);
        const int64_t y_index = indices.first;
        const int64_t z_index = indices.second;

        ComputeOneDimensionDistanceTransformInPlace(
            num_x_cells, XIndexer(y_index, z_index), scratch_z, scratch_v,
            scratch_d, distance_field);
      }
    };

    const int64_t iterations = num_y_cells * num_z_cells;
    StaticParallelForLoop(
        parallelism, 0, iterations, thread_work,
        ParallelForBackend::BEST_AVAILABLE);
  }

  // Transform along Y axis
  if (num_y_cells > 1)
  {
    // Allocate scratch space (don't need to initialize, since transform
    // functions reset their scratch space as needed).

    // Note, z requires an additional element.
    MatrixXd full_scratch_z(num_y_cells + 1, num_threads);
    MatrixXi64 full_scratch_v(num_y_cells, num_threads);
    MatrixXd full_scratch_d(num_y_cells, num_threads);

    const auto thread_work = [&](const ThreadWorkRange& work_range)
    {
      const int32_t thread_num = work_range.GetThreadNum();
      auto scratch_z = full_scratch_z.col(thread_num);
      auto scratch_v = full_scratch_v.col(thread_num);
      auto scratch_d = full_scratch_d.col(thread_num);

      for (int64_t iteration = work_range.GetRangeStart();
           iteration < work_range.GetRangeEnd();
           iteration++)
      {
        const auto indices =
            get_indicies_from_iteration(num_x_cells, iteration);
        const int64_t x_index = indices.first;
        const int64_t z_index = indices.second;

        ComputeOneDimensionDistanceTransformInPlace(
            num_y_cells, YIndexer(x_index, z_index), scratch_z, scratch_v,
            scratch_d, distance_field);
      }
    };

    const int64_t iterations = num_x_cells * num_z_cells;
    StaticParallelForLoop(
        parallelism, 0, iterations, thread_work,
        ParallelForBackend::BEST_AVAILABLE);
  }

  // Transform along Z axis
  if (num_z_cells > 1)
  {
    // Allocate scratch space (don't need to initialize, since transform
    // functions reset their scratch space as needed).

    // Note, z requires an additional element.
    MatrixXd full_scratch_z(num_z_cells + 1, num_threads);
    MatrixXi64 full_scratch_v(num_z_cells, num_threads);
    MatrixXd full_scratch_d(num_z_cells, num_threads);

    const auto thread_work = [&](const ThreadWorkRange& work_range)
    {
      const int32_t thread_num = work_range.GetThreadNum();
      auto scratch_z = full_scratch_z.col(thread_num);
      auto scratch_v = full_scratch_v.col(thread_num);
      auto scratch_d = full_scratch_d.col(thread_num);

      for (int64_t iteration = work_range.GetRangeStart();
           iteration < work_range.GetRangeEnd();
           iteration++)
      {
        const auto indices =
            get_indicies_from_iteration(num_x_cells, iteration);
        const int64_t x_index = indices.first;
        const int64_t y_index = indices.second;

        ComputeOneDimensionDistanceTransformInPlace(
            num_z_cells, ZIndexer(x_index, y_index), scratch_z, scratch_v,
            scratch_d, distance_field);
      }
    };

    const int64_t iterations = num_x_cells * num_y_cells;
    StaticParallelForLoop(
        parallelism, 0, iterations, thread_work,
        ParallelForBackend::BEST_AVAILABLE);
  }
}
}  // namespace internal
}  // namespace signed_distance_field_generation
}  // namespace voxelized_geometry_tools
