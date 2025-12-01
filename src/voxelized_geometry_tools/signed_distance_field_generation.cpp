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
using common_robotics_utilities::parallelism::StaticParallelForRangeLoop;
using common_robotics_utilities::parallelism::ThreadWorkRange;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using VectorXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
using MatrixXi64 = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
namespace signed_distance_field_generation
{
namespace internal
{
namespace
{
// Helpers for Euclidean Distance Transform implementation below

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

// Implementation of the 3-dimensional Euclidean Distance Transform based on a
// sequence of 1-dimensonal distance transforms adapted from
// "Distance Transforms of Sampled Functions",Felzenszwalb & Huttenlocker, 2012.
// Variable names match those used in the paper pseudocode (e.g. f, k, z, v, d),
// with adaptations to allow scratch space reuse between iterations to reduce
// the need for dynamic memory allocations. Parallelism is achieved by
// dispatching multiple 1-dimensional distance transforms simultaneuously.
void ComputeDistanceFieldTransformInPlace(
    const DegreeOfParallelism& parallelism, EDTDistanceField& distance_field)
{
  const int64_t num_x_cells = distance_field.NumXVoxels();
  const int64_t num_y_cells = distance_field.NumYVoxels();
  const int64_t num_z_cells = distance_field.NumZVoxels();

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

    const auto per_range_work = [&](const ThreadWorkRange& work_range)
    {
      const int32_t thread_num = work_range.GetThreadNum();
      Eigen::Ref<VectorXd> scratch_z = full_scratch_z.col(thread_num);
      Eigen::Ref<VectorXi64> scratch_v = full_scratch_v.col(thread_num);
      Eigen::Ref<VectorXd> scratch_d = full_scratch_d.col(thread_num);

      for (int64_t iteration = work_range.GetRangeStart();
           iteration < work_range.GetRangeEnd();
           iteration++)
      {
        const auto indices =
            get_indicies_from_iteration(num_z_cells, iteration);
        const int64_t y_index = indices.first;
        const int64_t z_index = indices.second;

        ComputeOneDimensionDistanceTransformInPlace(
            num_x_cells, XIndexer(y_index, z_index), scratch_z, scratch_v,
            scratch_d, distance_field);
      }
    };

    const int64_t iterations = num_y_cells * num_z_cells;
    StaticParallelForRangeLoop(
        parallelism, 0, iterations, per_range_work,
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

    const auto per_range_work = [&](const ThreadWorkRange& work_range)
    {
      const int32_t thread_num = work_range.GetThreadNum();
      Eigen::Ref<VectorXd> scratch_z = full_scratch_z.col(thread_num);
      Eigen::Ref<VectorXi64> scratch_v = full_scratch_v.col(thread_num);
      Eigen::Ref<VectorXd> scratch_d = full_scratch_d.col(thread_num);

      for (int64_t iteration = work_range.GetRangeStart();
           iteration < work_range.GetRangeEnd();
           iteration++)
      {
        const auto indices =
            get_indicies_from_iteration(num_z_cells, iteration);
        const int64_t x_index = indices.first;
        const int64_t z_index = indices.second;

        ComputeOneDimensionDistanceTransformInPlace(
            num_y_cells, YIndexer(x_index, z_index), scratch_z, scratch_v,
            scratch_d, distance_field);
      }
    };

    const int64_t iterations = num_x_cells * num_z_cells;
    StaticParallelForRangeLoop(
        parallelism, 0, iterations, per_range_work,
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

    const auto per_range_work = [&](const ThreadWorkRange& work_range)
    {
      const int32_t thread_num = work_range.GetThreadNum();
      Eigen::Ref<VectorXd> scratch_z = full_scratch_z.col(thread_num);
      Eigen::Ref<VectorXi64> scratch_v = full_scratch_v.col(thread_num);
      Eigen::Ref<VectorXd> scratch_d = full_scratch_d.col(thread_num);

      for (int64_t iteration = work_range.GetRangeStart();
           iteration < work_range.GetRangeEnd();
           iteration++)
      {
        const auto indices =
            get_indicies_from_iteration(num_y_cells, iteration);
        const int64_t x_index = indices.first;
        const int64_t y_index = indices.second;

        ComputeOneDimensionDistanceTransformInPlace(
            num_z_cells, ZIndexer(x_index, y_index), scratch_z, scratch_v,
            scratch_d, distance_field);
      }
    };

    const int64_t iterations = num_x_cells * num_y_cells;
    StaticParallelForRangeLoop(
        parallelism, 0, iterations, per_range_work,
        ParallelForBackend::BEST_AVAILABLE);
  }
}
}  // namespace internal
}  // namespace signed_distance_field_generation
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
