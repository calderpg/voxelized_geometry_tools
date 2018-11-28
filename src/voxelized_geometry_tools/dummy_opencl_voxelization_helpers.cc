#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
bool IsAvailable() { return false; }

int32_t* PrepareTrackingGrid(const int64_t)
{
  return nullptr;
}

void RaycastPoints(
    const float*, const int32_t, const float* const, const float* const,
    const float, const int32_t, const int32_t, const int32_t, int32_t* const) {}

float* PrepareFilterGrid(const int64_t, const void*)
{
  return nullptr;
}

void FilterTrackingGrids(
    const int64_t, const int32_t, int32_t* const *, float* const, const float,
    const int32_t, const int32_t) {}

void RetrieveTrackingGrid(const int64_t, const int32_t* const, void*) {}

void RetrieveFilteredGrid(const int64_t, const float* const, void*) {}

void CleanupDeviceMemory(const int32_t, int32_t* const*, float*) {}
}
}
}
