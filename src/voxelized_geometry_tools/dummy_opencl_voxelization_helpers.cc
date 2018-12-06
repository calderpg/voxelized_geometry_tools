#include <voxelized_geometry_tools/opencl_voxelization_helpers.h>

#include <cstdint>
#include <vector>

#include <Eigen/Geometry>

namespace voxelized_geometry_tools
{
namespace pointcloud_voxelization
{
namespace opencl_helpers
{
class DummyOpenCLVoxelizationHelperInterface
    : public OpenCLVoxelizationHelperInterface
{
public:
  bool IsAvailable() const override { return false; }

  std::vector<int64_t> PrepareTrackingGrids(
      const int64_t, const int32_t) override
  {
    return std::vector<int64_t>();
  }

  void RaycastPoints(
      const std::vector<float>&, const Eigen::Isometry3f&,
      const Eigen::Isometry3f&, const float, const float, const int32_t,
      const int32_t, const int32_t, const int64_t) override {}

  bool PrepareFilterGrid(const int64_t, const void*) override { return false; }

  void FilterTrackingGrids(
       const int64_t, const int32_t, const float, const int32_t,
       const int32_t) override {}

  void RetrieveTrackingGrid(const int64_t, const int64_t, void*) override {}

  void RetrieveFilteredGrid(const int64_t, void*) override {}

  void CleanupAllocatedMemory() override {}
};

OpenCLVoxelizationHelperInterface* MakeHelperInterface()
{
  return new DummyOpenCLVoxelizationHelperInterface();
}
}
}
}
