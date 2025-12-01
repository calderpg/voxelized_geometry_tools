#include <voxelized_geometry_tools/occupancy_map_conversions.hpp>

#include <stdexcept>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN

namespace
{

template<typename FromMapType, typename ToMapType, typename CellConverter>
ToMapType ConvertOccupancyMapImpl(
    const FromMapType& from_map, const CellConverter& cell_converter)
{
  if (from_map.IsInitialized())
  {
    ToMapType to_map(
        from_map.OriginTransform(), from_map.Frame(), from_map.ControlSizes(),
        cell_converter(from_map.DefaultValue()),
        cell_converter(from_map.OOBValue()));

    const auto& from_backing_store = from_map.GetImmutableRawData();
    auto& to_backing_store = to_map.GetMutableRawData();

    const size_t num_from_cells = from_backing_store.size();
    const size_t num_to_cells = to_backing_store.size();

    if (num_from_cells != num_to_cells)
    {
      throw std::runtime_error("Incompatible backing store sizes");
    }

    for (size_t index = 0; index < num_from_cells; index++)
    {
      const auto& from_cell = from_backing_store.at(index);
      to_backing_store.at(index) = cell_converter(from_cell);
    }

    return to_map;
  }
  else
  {
    return ToMapType();
  }
}

}  // namespace

OccupancyComponentMap ConvertToOccupancyComponentMap(
    const OccupancyMap& occupancy_map)
{
  return ConvertOccupancyMapImpl<OccupancyMap, OccupancyComponentMap>(
      occupancy_map, ConvertToOccupancyComponentCell);
}

OccupancyMap ConvertFromOccupancyComponentMap(
    const OccupancyComponentMap& occupancy_component_map)
{
  return ConvertOccupancyMapImpl<OccupancyComponentMap, OccupancyMap>(
      occupancy_component_map, ConvertFromOccupancyComponentCell);
}

TaggedObjectOccupancyComponentMap ConvertToTaggedObjectOccupancyComponentMap(
    const TaggedObjectOccupancyMap& occupancy_map)
{
  return ConvertOccupancyMapImpl
      <TaggedObjectOccupancyMap, TaggedObjectOccupancyComponentMap>(
          occupancy_map, ConvertToTaggedObjectOccupancyComponentCell);
}

TaggedObjectOccupancyMap ConvertFromTaggedObjectOccupancyComponentMap(
    const TaggedObjectOccupancyComponentMap& occupancy_component_map)
{
  return ConvertOccupancyMapImpl
      <TaggedObjectOccupancyComponentMap, TaggedObjectOccupancyMap>(
          occupancy_component_map,
          ConvertFromTaggedObjectOccupancyComponentCell);
}

VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
