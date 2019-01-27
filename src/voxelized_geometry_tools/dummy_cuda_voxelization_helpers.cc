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
/*
bool IsAvailable() { return false; }

float* PreparePointCloud(const int32_t, const float*)
{
  return nullptr;
}

int32_t* PrepareTrackingGrid(const int64_t)
{
  return nullptr;
}

void RaycastPoints(
    const float*, const int32_t, const float* const, const float* const,
    const float, const float, const int32_t, const int32_t, const int32_t,
    int32_t* const) {}

float* PrepareFilterGrid(const int64_t, const void*)
{
  return nullptr;
}

void FilterTrackingGrids(
    const int64_t, const int32_t, int32_t* const *, float* const, const float,
    const int32_t, const int32_t) {}

void RetrieveTrackingGrid(const int64_t, const int32_t* const, void*) {}

void RetrieveFilteredGrid(const int64_t, const float* const, void*) {}

void CleanupDeviceMemory(
    const int32_t, float* const*, const int32_t, int32_t* const*, float*) {}

void SetCudaDevice(const int32_t) {}
*/

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
      const std::vector<float>&, const float* const, const float * const,
      const float, const float, const int32_t, const int32_t, const int32_t,
      const int64_t) override {}

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
