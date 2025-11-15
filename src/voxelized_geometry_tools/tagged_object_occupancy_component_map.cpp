#include <voxelized_geometry_tools/tagged_object_occupancy_component_map.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <voxelized_geometry_tools/topology_computation.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
uint64_t TaggedObjectOccupancyComponentCell::Serialize(
    const TaggedObjectOccupancyComponentCell& cell,
    std::vector<uint8_t>& buffer)
{
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeMemcpyable<float>(
      cell.Occupancy(), buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      cell.ObjectId(), buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      cell.Component(), buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      cell.SpatialSegment(), buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

TaggedObjectOccupancyComponentCell::
    DeserializedTaggedObjectOccupancyComponentCell
TaggedObjectOccupancyComponentCell::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  uint64_t current_position = starting_offset;
  const auto occupancy_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<float>(buffer, current_position);
  current_position += occupancy_deserialized.BytesRead();
  const auto object_id_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint32_t>(buffer, current_position);
  current_position += object_id_deserialized.BytesRead();
  const auto component_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint32_t>(buffer, current_position);
  current_position += component_deserialized.BytesRead();
  const auto spatial_segment_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint32_t>(buffer, current_position);
  current_position += spatial_segment_deserialized.BytesRead();
  const TaggedObjectOccupancyComponentCell cell(
      occupancy_deserialized.Value(), object_id_deserialized.Value(),
      component_deserialized.Value(), spatial_segment_deserialized.Value());
  // Figure out how many bytes were read
  const uint64_t bytes_read = current_position - starting_offset;
  return common_robotics_utilities::serialization::MakeDeserialized(
      cell, bytes_read);
}

/// We need to implement cloning.
std::unique_ptr<common_robotics_utilities::voxel_grid::VoxelGridBase
    <TaggedObjectOccupancyComponentCell,
     std::vector<TaggedObjectOccupancyComponentCell>>>
TaggedObjectOccupancyComponentMap::DoClone() const
{
  return std::unique_ptr<TaggedObjectOccupancyComponentMap>(
      new TaggedObjectOccupancyComponentMap(*this));
}

/// We need to serialize the frame and locked flag.
uint64_t TaggedObjectOccupancyComponentMap::DerivedSerializeSelf(
    std::vector<uint8_t>& buffer,
    const TaggedObjectOccupancyComponentCellSerializer& value_serializer) const
{
  CRU_UNUSED(value_serializer);
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      number_of_components_, buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      number_of_spatial_segments_, buffer);
  common_robotics_utilities::serialization::SerializeString(frame_, buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint8_t>(
      static_cast<uint8_t>(components_valid_.load()), buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint8_t>(
      static_cast<uint8_t>(spatial_segments_valid_.load()), buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

/// We need to deserialize the frame and locked flag.
uint64_t TaggedObjectOccupancyComponentMap::DerivedDeserializeSelf(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
    const TaggedObjectOccupancyComponentCellDeserializer& value_deserializer)
{
  CRU_UNUSED(value_deserializer);
  uint64_t current_position = starting_offset;
  const auto number_of_components_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint32_t>(buffer, current_position);
  number_of_components_ = number_of_components_deserialized.Value();
  current_position += number_of_components_deserialized.BytesRead();
  const auto number_of_spatial_segments_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint32_t>(buffer, current_position);
  number_of_spatial_segments_ = number_of_spatial_segments_deserialized.Value();
  current_position += number_of_spatial_segments_deserialized.BytesRead();
  const auto frame_deserialized
      = common_robotics_utilities::serialization::DeserializeString<char>(
          buffer, current_position);
  frame_ = frame_deserialized.Value();
  current_position += frame_deserialized.BytesRead();
  const auto components_valid_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint8_t>(buffer, current_position);
  components_valid_.store(
      static_cast<bool>(components_valid_deserialized.Value()));
  current_position += components_valid_deserialized.BytesRead();
  const auto spatial_segments_valid_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint8_t>(buffer, current_position);
  spatial_segments_valid_.store(
      static_cast<bool>(spatial_segments_valid_deserialized.Value()));
  current_position += spatial_segments_valid_deserialized.BytesRead();
  // Figure out how many bytes were read
  const uint64_t bytes_read = current_position - starting_offset;
  return bytes_read;
}

/// Invalidate connected components and spatial segments on mutable access.
bool TaggedObjectOccupancyComponentMap::OnMutableAccess(
    const int64_t x_index, const int64_t y_index, const int64_t z_index)
{
  CRU_UNUSED(x_index);
  CRU_UNUSED(y_index);
  CRU_UNUSED(z_index);
  components_valid_.store(false);
  spatial_segments_valid_.store(false);
  return true;
}

/// Invalidate connected components and spatial segments on mutable raw access.
bool TaggedObjectOccupancyComponentMap::OnMutableRawAccess()
{
  components_valid_.store(false);
  spatial_segments_valid_.store(false);
  return true;
}

uint64_t TaggedObjectOccupancyComponentMap::Serialize(
    const TaggedObjectOccupancyComponentMap& map, std::vector<uint8_t>& buffer)
{
  return map.SerializeSelf(
      buffer, TaggedObjectOccupancyComponentCell::Serialize);
}

TaggedObjectOccupancyComponentMap::DeserializedTaggedObjectOccupancyComponentMap
TaggedObjectOccupancyComponentMap::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  TaggedObjectOccupancyComponentMap temp_map;
  const uint64_t bytes_read = temp_map.DeserializeSelf(
      buffer, starting_offset, TaggedObjectOccupancyComponentCell::Deserialize);
  return common_robotics_utilities::serialization::MakeDeserialized(
      temp_map, bytes_read);
}

void TaggedObjectOccupancyComponentMap::SaveToFile(
    const TaggedObjectOccupancyComponentMap& map,
    const std::string& filepath,
    const bool compress)
{
  std::vector<uint8_t> buffer;
  TaggedObjectOccupancyComponentMap::Serialize(map, buffer);
  std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
  if (compress)
  {
    output_file.write("TMGZ", 4);
    const std::vector<uint8_t> compressed
        = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
    const size_t serialized_size = compressed.size();
    output_file.write(
        reinterpret_cast<const char*>(compressed.data()),
        static_cast<std::streamsize>(serialized_size));
  }
  else
  {
    output_file.write("TMGR", 4);
    const size_t serialized_size = buffer.size();
    output_file.write(
        reinterpret_cast<const char*>(buffer.data()),
        static_cast<std::streamsize>(serialized_size));
  }
  output_file.close();
}

TaggedObjectOccupancyComponentMap
TaggedObjectOccupancyComponentMap::LoadFromFile(const std::string& filepath)
{
  std::ifstream input_file(
      filepath, std::ios::in | std::ios::binary | std::ios::ate);
  if (input_file.good() == false)
  {
    throw std::invalid_argument("File does not exist");
  }
  const std::streampos end = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  const std::streampos begin = input_file.tellg();
  const std::streamsize serialized_size = end - begin;
  const std::streamsize header_size = 4;
  if (serialized_size >= header_size)
  {
    // Load the header
    std::vector<uint8_t> file_header(header_size + 1, 0x00);
    input_file.read(reinterpret_cast<char*>(file_header.data()),
                    header_size);
    const std::string header_string(
          reinterpret_cast<const char*>(file_header.data()));
    // Load the rest of the file
    std::vector<uint8_t> file_buffer(
          static_cast<size_t>(serialized_size - header_size), 0x00);
    input_file.read(reinterpret_cast<char*>(file_buffer.data()),
                    serialized_size - header_size);
    // Deserialize
    if (header_string == "TMGZ")
    {
      const std::vector<uint8_t> decompressed
          = common_robotics_utilities::zlib_helpers
              ::DecompressBytes(file_buffer);
      return TaggedObjectOccupancyComponentMap::Deserialize(
          decompressed, 0).Value();
    }
    else if (header_string == "TMGR")
    {
      return TaggedObjectOccupancyComponentMap::Deserialize(
          file_buffer, 0).Value();
    }
    else
    {
      throw std::invalid_argument(
            "File has invalid header [" + header_string + "]");
    }
  }
  else
  {
    throw std::invalid_argument("File is too small");
  }
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::IsSurfaceIndex(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return IsSurfaceIndex(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::IsSurfaceIndex(
    const int64_t x_index, const int64_t y_index,
    const int64_t z_index) const
{
  // First, we make sure that indices are within bounds
  // Out of bounds indices are NOT surface cells
  if (IndexInBounds(x_index, y_index, z_index) == false)
  {
    return common_robotics_utilities::OwningMaybe<bool>();
  }
  // Check all 26 possible neighbors
  const int64_t min_x_check = std::max(INT64_C(0), x_index - 1);
  const int64_t max_x_check = std::min(GetNumXCells() - 1, x_index + 1);
  const int64_t min_y_check = std::max(INT64_C(0), y_index - 1);
  const int64_t max_y_check = std::min(GetNumYCells() - 1, y_index + 1);
  const int64_t min_z_check = std::max(INT64_C(0), z_index - 1);
  const int64_t max_z_check = std::min(GetNumZCells() - 1, z_index + 1);
  const float our_occupancy
      = GetIndexImmutable(x_index, y_index, z_index).Value().Occupancy();
  for (int64_t x_idx = min_x_check; x_idx <= max_x_check; x_idx++)
  {
    for (int64_t y_idx = min_y_check; y_idx <= max_y_check; y_idx++)
    {
      for (int64_t z_idx = min_z_check; z_idx <= max_z_check; z_idx++)
      {
        // Skip ourselves
        if ((x_idx != x_index) || (y_idx != y_index) || (z_idx != z_index))
        {
          const float other_occupancy
              = GetIndexImmutable(x_idx, y_idx, z_idx).Value().Occupancy();
          if ((our_occupancy < 0.5) && (other_occupancy >= 0.5))
          {
            return common_robotics_utilities::OwningMaybe<bool>(true);
          }
          else if ((our_occupancy > 0.5) && (other_occupancy <= 0.5))
          {
            return common_robotics_utilities::OwningMaybe<bool>(true);
          }
          else if ((our_occupancy == 0.5) && (other_occupancy != 0.5))
          {
            return common_robotics_utilities::OwningMaybe<bool>(true);
          }
        }
      }
    }
  }
  return common_robotics_utilities::OwningMaybe<bool>(false);
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::IsConnectedComponentSurfaceIndex(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return IsConnectedComponentSurfaceIndex(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::IsConnectedComponentSurfaceIndex(
    const int64_t x_index, const int64_t y_index,
    const int64_t z_index) const
{
  // First, we make sure that indices are within bounds
  // Out of bounds indices are NOT surface cells
  if (!IndexInBounds(x_index, y_index, z_index))
  {
    return common_robotics_utilities::OwningMaybe<bool>();
  }
  // Edge indices are automatically surface cells
  if (x_index == 0 || y_index == 0 || z_index == 0
      || x_index == (GetNumXCells() - 1) || y_index == (GetNumYCells() - 1)
      || z_index == (GetNumZCells() - 1))
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // If the cell is inside the grid, we check the neighbors
  // Note that we must check all 26 neighbors
  const uint32_t our_component
      = GetIndexImmutable(x_index, y_index, z_index).Value().Component();
  // Check neighbor 1
  if (our_component !=
      GetIndexImmutable(x_index, y_index, z_index - 1).Value().Component())
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // Check neighbor 2
  else if (our_component !=
           GetIndexImmutable(x_index, y_index, z_index + 1).Value().Component())
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // Check neighbor 3
  else if (our_component !=
           GetIndexImmutable(x_index, y_index - 1, z_index).Value().Component())
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // Check neighbor 4
  else if (our_component !=
           GetIndexImmutable(x_index, y_index + 1, z_index).Value().Component())
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // Check neighbor 5
  else if (our_component !=
           GetIndexImmutable(x_index - 1, y_index, z_index).Value().Component())
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // Check neighbor 6
  else if (our_component !=
           GetIndexImmutable(x_index + 1, y_index, z_index).Value().Component())
  {
    return common_robotics_utilities::OwningMaybe<bool>(true);
  }
  // If none of the faces are exposed, it's not a surface voxel
  return common_robotics_utilities::OwningMaybe<bool>(false);
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::CheckIfCandidateCorner(
    const double x, const double y, const double z) const
{
  return CheckIfCandidateCorner4d(Eigen::Vector4d(x, y, z, 1.0));
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::CheckIfCandidateCorner3d(
    const Eigen::Vector3d& location) const
{
  return CheckIfCandidateCorner(LocationToGridIndex3d(location));
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::CheckIfCandidateCorner4d(
    const Eigen::Vector4d& location) const
{
  return CheckIfCandidateCorner(LocationToGridIndex4d(location));
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::CheckIfCandidateCorner(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return CheckIfCandidateCorner(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyComponentMap::CheckIfCandidateCorner(
    const int64_t x_index, const int64_t y_index, const int64_t z_index) const
{
  const auto current_cell = GetIndexImmutable(x_index, y_index, z_index);
  if (current_cell)
  {
    const auto& current_cell_value = current_cell.Value();
    // Grab the six neighbors & check if they belong to a different component
    uint32_t different_neighbors = 0u;
    const auto xm1yz_cell = GetIndexImmutable(x_index - 1, y_index, z_index);
    if (xm1yz_cell
        && (xm1yz_cell.Value().Component() != current_cell_value.Component()))
    {
      different_neighbors++;
    }
    const auto xp1yz_cell = GetIndexImmutable(x_index + 1, y_index, z_index);
    if (xp1yz_cell
        && (xp1yz_cell.Value().Component() != current_cell_value.Component()))
    {
      different_neighbors++;
    }
    const auto xym1z_cell = GetIndexImmutable(x_index, y_index - 1, z_index);
    if (xym1z_cell
        && (xym1z_cell.Value().Component() != current_cell_value.Component()))
    {
      different_neighbors++;
    }
    const auto xyp1z_cell = GetIndexImmutable(x_index, y_index + 1, z_index);
    if (xyp1z_cell
        && (xyp1z_cell.Value().Component() != current_cell_value.Component()))
    {
      different_neighbors++;
    }
    const auto xyzm1_cell = GetIndexImmutable(x_index, y_index, z_index - 1);
    if (xyzm1_cell
        && (xyzm1_cell.Value().Component() != current_cell_value.Component()))
    {
      different_neighbors++;
    }
    const auto xyzp1_cell = GetIndexImmutable(x_index, y_index, z_index + 1);
    if (xyzp1_cell
        && (xyzp1_cell.Value().Component() != current_cell_value.Component()))
    {
      different_neighbors++;
    }
    // We now have between zero and six neighbors to work with
    if (different_neighbors <= 1u)
    {
      // If there is one or fewer neighbors to work with,
      // we are clearly not a corner
      return common_robotics_utilities::OwningMaybe<bool>(false);
    }
    else
    {
      // If there are 2 or more neighbors to work with,
      // we are a candidate corner
      return common_robotics_utilities::OwningMaybe<bool>(true);
    }
  }
  else
  {
    // Not in the grid
    return common_robotics_utilities::OwningMaybe<bool>();
  }
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
TaggedObjectOccupancyComponentMap::ExtractComponentSurfaces(
    const COMPONENT_TYPES component_types_to_extract) const
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  // Make the helper functions
  const std::function<int64_t(const GridIndex&)> get_component_fn
      = [&] (const GridIndex& index)
  {
    const auto query = GetIndexImmutable(index);
    if (query)
    {
      return static_cast<int64_t>(query.Value().Component());
    }
    else
    {
      return INT64_C(-1);
    }
  };
  const std::function<bool(const GridIndex&)> is_surface_index_fn
      = [&] (const GridIndex& index)
  {
    const auto query = GetIndexImmutable(index);
    const TaggedObjectOccupancyComponentCell& current_cell = query.Value();
    if (current_cell.Occupancy() > 0.5)
    {
      if ((component_types_to_extract & FILLED_COMPONENTS) > 0x00)
      {
        if (IsConnectedComponentSurfaceIndex(index))
        {
          return true;
        }
      }
    }
    else if (current_cell.Occupancy() < 0.5)
    {
      if ((component_types_to_extract & EMPTY_COMPONENTS) > 0x00)
      {
        if (IsConnectedComponentSurfaceIndex(index))
        {
          return true;
        }
      }
    }
    else
    {
      if ((component_types_to_extract & UNKNOWN_COMPONENTS) > 0x00)
      {
        if (IsConnectedComponentSurfaceIndex(index))
        {
          return true;
        }
      }
    }
    return false;
  };
  return topology_computation::ExtractComponentSurfaces(
      *this, get_component_fn, is_surface_index_fn);
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
TaggedObjectOccupancyComponentMap::ExtractFilledComponentSurfaces() const
{
  return ExtractComponentSurfaces(FILLED_COMPONENTS);
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
TaggedObjectOccupancyComponentMap::ExtractUnknownComponentSurfaces() const
{
  return ExtractComponentSurfaces(UNKNOWN_COMPONENTS);
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
TaggedObjectOccupancyComponentMap::ExtractEmptyComponentSurfaces() const
{
  return ExtractComponentSurfaces(EMPTY_COMPONENTS);
}

topology_computation::TopologicalInvariants
TaggedObjectOccupancyComponentMap::ComputeComponentTopology(
    const COMPONENT_TYPES component_types_to_use,
    const bool connect_across_objects,
    const common_robotics_utilities::utility::LoggingFunction& logging_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  UpdateConnectedComponents(connect_across_objects);
  const std::function<int64_t(const GridIndex&)> get_component_fn
      = [&] (const GridIndex& index)
  {
    const auto query = GetIndexImmutable(index);
    if (query)
    {
      return static_cast<int64_t>(query.Value().Component());
    }
    else
    {
      return INT64_C(-1);
    }
  };
  const std::function<bool(const GridIndex&)> is_surface_index_fn
      = [&] (const GridIndex& index)
  {
    const auto query = GetIndexImmutable(index);
    const TaggedObjectOccupancyComponentCell& current_cell = query.Value();
    if (current_cell.Occupancy() > 0.5)
    {
      if ((component_types_to_use & FILLED_COMPONENTS) > 0x00)
      {
        if (IsConnectedComponentSurfaceIndex(index))
        {
          return true;
        }
      }
    }
    else if (current_cell.Occupancy() < 0.5)
    {
      if ((component_types_to_use & EMPTY_COMPONENTS) > 0x00)
      {
        if (IsConnectedComponentSurfaceIndex(index))
        {
          return true;
        }
      }
    }
    else
    {
      if ((component_types_to_use & UNKNOWN_COMPONENTS) > 0x00)
      {
        if (IsConnectedComponentSurfaceIndex(index))
        {
          return true;
        }
      }
    }
    return false;
  };
  return topology_computation::ComputeComponentTopology(
      *this, get_component_fn, is_surface_index_fn, logging_fn);
}

SignedDistanceField<double>
TaggedObjectOccupancyComponentMap::ExtractSignedDistanceFieldDouble(
    const std::vector<uint32_t>& objects_to_use,
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return ExtractSignedDistanceField<double>(objects_to_use, parameters);
}

SignedDistanceField<float>
TaggedObjectOccupancyComponentMap::ExtractSignedDistanceFieldFloat(
    const std::vector<uint32_t>& objects_to_use,
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return ExtractSignedDistanceField<float>(objects_to_use, parameters);
}

std::map<uint32_t, SignedDistanceField<double>>
TaggedObjectOccupancyComponentMap::MakeSeparateObjectSDFsDouble(
    const std::vector<uint32_t>& object_ids,
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return MakeSeparateObjectSDFs<double>(object_ids, parameters);
}

std::map<uint32_t, SignedDistanceField<float>>
TaggedObjectOccupancyComponentMap::MakeSeparateObjectSDFsFloat(
    const std::vector<uint32_t>& object_ids,
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return MakeSeparateObjectSDFs<float>(object_ids, parameters);
}

std::map<uint32_t, SignedDistanceField<double>>
TaggedObjectOccupancyComponentMap::MakeAllObjectSDFsDouble(
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return MakeAllObjectSDFs<double>(parameters);
}

std::map<uint32_t, SignedDistanceField<float>>
TaggedObjectOccupancyComponentMap::MakeAllObjectSDFsFloat(
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return MakeAllObjectSDFs<float>(parameters);
}

SignedDistanceField<double>
TaggedObjectOccupancyComponentMap::
ExtractFreeAndNamedObjectsSignedDistanceFieldDouble(
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return ExtractFreeAndNamedObjectsSignedDistanceField<double>(parameters);
}

SignedDistanceField<float>
TaggedObjectOccupancyComponentMap::
ExtractFreeAndNamedObjectsSignedDistanceFieldFloat(
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return ExtractFreeAndNamedObjectsSignedDistanceField<float>(parameters);
}

uint32_t TaggedObjectOccupancyComponentMap::UpdateConnectedComponents(
    const bool connect_across_objects)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  // If the connected components are already valid, skip computing them again
  if (components_valid_.load())
  {
    return number_of_components_;
  }
  components_valid_.store(false);
  // Make the helper functions
  const std::function<bool(const GridIndex&, const GridIndex&)>
    are_connected_fn = [&] (const GridIndex& index1, const GridIndex& index2)
  {
    const auto query1 = GetIndexImmutable(index1);
    const auto query2 = GetIndexImmutable(index2);
    if ((query1.Value().Occupancy() > 0.5)
        && (query2.Value().Occupancy() > 0.5))
    {
      if (connect_across_objects)
      {
        return true;
      }
      else
      {
        return (query1.Value().ObjectId() == query2.Value().ObjectId());
      }
    }
    else if ((query1.Value().Occupancy() < 0.5)
             && (query2.Value().Occupancy() < 0.5))
    {
      if (connect_across_objects)
      {
        return true;
      }
      else
      {
        return (query1.Value().ObjectId() == query2.Value().ObjectId());
      }
    }
    else if ((query1.Value().Occupancy() == 0.5)
             && (query2.Value().Occupancy() == 0.5))
    {
      if (connect_across_objects)
      {
        return true;
      }
      else
      {
        return (query1.Value().ObjectId() == query2.Value().ObjectId());
      }
    }
    else
    {
      return false;
    }
  };
  const std::function<int64_t(const GridIndex&)> get_component_fn
      = [&] (const GridIndex& index)
  {
    const auto query = GetIndexImmutable(index);
    if (query)
    {
      return static_cast<int64_t>(query.Value().Component());
    }
    else
    {
      return INT64_C(-1);
    }
  };
  const std::function<void(const GridIndex&, const uint32_t)> mark_component_fn
      = [&] (const GridIndex& index, const uint32_t component)
  {
    auto query = GetIndexMutable(index);
    if (query)
    {
      query.Value().SetComponent(component);
    }
  };
  number_of_components_
      = topology_computation::ComputeConnectedComponents(
          *this, are_connected_fn, get_component_fn, mark_component_fn);
  components_valid_.store(true);
  return number_of_components_;
}

uint32_t TaggedObjectOccupancyComponentMap::UpdateSpatialSegments(
    const double connected_threshold,
    const SignedDistanceFieldGenerationParameters<float>& sdf_parameters)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  // If the connected components are already valid, skip computing them again
  if (spatial_segments_valid_.load())
  {
    return number_of_spatial_segments_;
  }
  spatial_segments_valid_.store(false);
  const auto sdf
      = (sdf_parameters.AddVirtualBorder())
        ? ExtractSignedDistanceFieldFloat({}, sdf_parameters)
        : ExtractFreeAndNamedObjectsSignedDistanceFieldFloat(sdf_parameters);
  const auto extrema_map = sdf.ComputeLocalExtremaMap();
  // Make the helper functions
  // This is not enough, we also need to limit the curvature of the
  // segment/local extrema cluster! Otherwise thin objects will always have
  // their local extrema dominated by their surroundings rather than their
  // own structure!
  const std::function<bool(const GridIndex&, const GridIndex&)> are_connected_fn
      = [&] (const GridIndex& index1, const GridIndex& index2)
  {
    const auto query1 = GetIndexImmutable(index1);
    const auto query2 = GetIndexImmutable(index2);
    if (query1.Value().ObjectId() == query2.Value().ObjectId())
    {
      const auto exmap_query1 = extrema_map.GetIndexImmutable(index1);
      const auto examp_query2 = extrema_map.GetIndexImmutable(index2);
      const double maxima_distance
          = (exmap_query1.Value() - examp_query2.Value()).norm();
      if (maxima_distance < connected_threshold)
      {
        return true;
      }
      else
      {
        return false;
      }
    }
    else
    {
      return false;
    }
  };
  const std::function<int64_t(const GridIndex&)> get_component_fn
      = [&] (const GridIndex& index)
  {
    const auto query = GetIndexImmutable(index);
    const auto extrema_query = extrema_map.GetIndexImmutable(index);
    if (query)
    {
      if ((query.Value().Occupancy() < 0.5f) || (query.Value().ObjectId() > 0u))
      {
        const Eigen::Vector3d& extrema = extrema_query.Value();
        if (!std::isinf(extrema.x())
            && !std::isinf(extrema.y())
            && !std::isinf(extrema.z()))
        {
          return static_cast<int64_t>(query.Value().SpatialSegment());
        }
        else
        {
          // Ignore cells with infinite extrema
          return INT64_C(-1);
        }
      }
      else
      {
        // Ignore filled cells that don't belong to an object
        return INT64_C(-1);
      }
    }
    else
    {
      return INT64_C(-1);
    }
  };
  const std::function<void(const GridIndex&, const uint32_t)> mark_component_fn
      = [&] (const GridIndex& index, const uint32_t component)
  {
    auto query = GetIndexMutable(index);
    if (query)
    {
      query.Value().SetSpatialSegment(component);
    }
  };
  number_of_spatial_segments_
      = topology_computation::ComputeConnectedComponents(
          *this, are_connected_fn, get_component_fn, mark_component_fn);
  spatial_segments_valid_.store(true);
  return number_of_spatial_segments_;
}
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
