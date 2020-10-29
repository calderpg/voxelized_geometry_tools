#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
class DummyCudaVoxelizationHelperInterface
    : public CudaVoxelizationHelperInterface
{
public:
  bool IsAvailable() const override { return false; }

  std::vector<int64_t> PrepareTrackingGrids(
      const int64_t, const int32_t) override
  {
    return std::vector<int64_t>();
  }

  void RaycastPoints(
      const std::vector<float>&, const float* const, const float,
      const float * const, const float, const float, const int32_t,
      const int32_t, const int32_t, const int64_t) override {}

  void PrepareFilterGrid(const int64_t, const void*) override {}

  void FilterTrackingGrids(
       const int64_t, const int32_t, const float, const int32_t,
       const int32_t) override {}

  void RetrieveTrackingGrid(const int64_t, const int64_t, void*) override {}

  void RetrieveFilteredGrid(const int64_t, void*) override {}

  void CleanupAllocatedMemory() override {}
};

CudaVoxelizationHelperInterface* MakeHelperInterface(
    const std::map<std::string, int32_t>&)
{
  return new DummyCudaVoxelizationHelperInterface();
}
}  // namespace cuda_helpers
}  // namespace pointcloud_voxelization
}  // namespace voxelized_geometry_tools
