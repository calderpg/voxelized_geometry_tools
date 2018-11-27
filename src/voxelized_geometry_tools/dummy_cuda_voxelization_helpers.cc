#include <voxelized_geometry_tools/cuda_voxelization_helpers.h>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace cuda_helpers
{
bool IsAvailable() { return false; }

float* PrepareTrackingGrid(const int64_t)
{
  return nullptr;
}

void RaycastPoints(
    const float*, const int32_t, const float* const, const float* const,
    const float, const int32_t, const int32_t, const int32_t, float* const) {}

float* PrepareFilterGrid(const int64_t, const void*)
{
  return nullptr;
}

void FilterTrackingGrids(
    const int64_t, const int32_t, float* const *, float* const, const float,
    const float, const float);

void RetrieveFilteredGrid(const int64_t, const float* const, void*) {}

void CleanupDeviceMemory(const int32_t, float* const*, float*) {}
}
}
}
