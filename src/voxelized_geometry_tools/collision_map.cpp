#include <voxelized_geometry_tools/collision_map.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iostream>
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
/// We need to implement cloning.
std::unique_ptr<common_robotics_utilities::voxel_grid
    ::VoxelGridBase<CollisionCell, std::vector<CollisionCell>>>
CollisionMap::DoClone() const
{
  return std::unique_ptr<CollisionMap>(new CollisionMap(*this));
}

/// We need to serialize the frame and locked flag.
uint64_t CollisionMap::DerivedSerializeSelf(
    std::vector<uint8_t>& buffer,
    const CollisionCellSerializer& value_serializer) const
{
  CRU_UNUSED(value_serializer);
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      number_of_components_, buffer);
  common_robotics_utilities::serialization::SerializeString(frame_, buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint8_t>(
      static_cast<uint8_t>(components_valid_), buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

/// We need to deserialize the frame and locked flag.
uint64_t CollisionMap::DerivedDeserializeSelf(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
    const CollisionCellDeserializer& value_deserializer)
{
  CRU_UNUSED(value_deserializer);
  uint64_t current_position = starting_offset;
  const auto number_of_components_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint32_t>(buffer, current_position);
  number_of_components_ = number_of_components_deserialized.Value();
  current_position += number_of_components_deserialized.BytesRead();
  const auto frame_deserialized
      = common_robotics_utilities::serialization::DeserializeString<char>(
          buffer, current_position);
  frame_ = frame_deserialized.Value();
  current_position += frame_deserialized.BytesRead();
  const auto components_valid_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<uint8_t>(buffer, current_position);
  components_valid_ = static_cast<bool>(components_valid_deserialized.Value());
  current_position += components_valid_deserialized.BytesRead();
  // Figure out how many bytes were read
  const uint64_t bytes_read = current_position - starting_offset;
  return bytes_read;
}

/// Invalidate connected components on mutable access.
bool CollisionMap::OnMutableAccess(const int64_t x_index,
                                       const int64_t y_index,
                                       const int64_t z_index)
{
  CRU_UNUSED(x_index);
  CRU_UNUSED(y_index);
  CRU_UNUSED(z_index);
  components_valid_ = false;
  return true;
}

uint64_t CollisionMap::Serialize(
    const CollisionMap& map, std::vector<uint8_t>& buffer)
{
  return map.SerializeSelf(buffer, common_robotics_utilities::serialization
                                       ::SerializeMemcpyable<CollisionCell>);
}

CollisionMap::DeserializedCollisionMap CollisionMap::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  CollisionMap temp_map;
  const uint64_t bytes_read
      = temp_map.DeserializeSelf(
          buffer, starting_offset,
          common_robotics_utilities::serialization
              ::DeserializeMemcpyable<CollisionCell>);
  return common_robotics_utilities::serialization::MakeDeserialized(
      temp_map, bytes_read);
}

void CollisionMap::SaveToFile(
    const CollisionMap& map,
    const std::string& filepath,
    const bool compress)
{
  std::vector<uint8_t> buffer;
  CollisionMap::Serialize(map, buffer);
  std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
  if (compress)
  {
    output_file.write("CMGZ", 4);
    const std::vector<uint8_t> compressed
        = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
    const size_t serialized_size = compressed.size();
    output_file.write(
        reinterpret_cast<const char*>(compressed.data()),
        static_cast<std::streamsize>(serialized_size));
  }
  else
  {
    output_file.write("CMGR", 4);
    const size_t serialized_size = buffer.size();
    output_file.write(
        reinterpret_cast<const char*>(buffer.data()),
        static_cast<std::streamsize>(serialized_size));
  }
  output_file.close();
}

CollisionMap CollisionMap::LoadFromFile(const std::string& filepath)
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
    if (header_string == "CMGZ")
    {
      const std::vector<uint8_t> decompressed
          = common_robotics_utilities::zlib_helpers
              ::DecompressBytes(file_buffer);
      return CollisionMap::Deserialize(decompressed, 0).Value();
    }
    else if (header_string == "CMGR")
    {
      return CollisionMap::Deserialize(file_buffer, 0).Value();
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

common_robotics_utilities::OwningMaybe<bool> CollisionMap::IsSurfaceIndex(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return IsSurfaceIndex(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool> CollisionMap::IsSurfaceIndex(
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
CollisionMap::IsConnectedComponentSurfaceIndex(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return IsConnectedComponentSurfaceIndex(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool>
CollisionMap::IsConnectedComponentSurfaceIndex(
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
CollisionMap::CheckIfCandidateCorner(
    const double x, const double y, const double z) const
{
  return CheckIfCandidateCorner4d(Eigen::Vector4d(x, y, z, 1.0));
}

common_robotics_utilities::OwningMaybe<bool>
CollisionMap::CheckIfCandidateCorner3d(
    const Eigen::Vector3d& location) const
{
  return CheckIfCandidateCorner(LocationToGridIndex3d(location));
}

common_robotics_utilities::OwningMaybe<bool>
CollisionMap::CheckIfCandidateCorner4d(
    const Eigen::Vector4d& location) const
{
  return CheckIfCandidateCorner(LocationToGridIndex4d(location));
}

common_robotics_utilities::OwningMaybe<bool>
CollisionMap::CheckIfCandidateCorner(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return CheckIfCandidateCorner(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool>
CollisionMap::CheckIfCandidateCorner(
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

uint32_t CollisionMap::UpdateConnectedComponents()
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  // If the connected components are already valid, skip computing them again
  if (components_valid_)
  {
    return number_of_components_;
  }
  components_valid_ = false;
  // Make the helper functions
  const std::function<bool(const GridIndex&, const GridIndex&)>
    are_connected_fn = [&] (const GridIndex& index1, const GridIndex& index2)
  {
    const auto query1 = GetIndexImmutable(index1);
    const auto query2 = GetIndexImmutable(index2);
    if ((query1.Value().Occupancy() > 0.5)
        && (query2.Value().Occupancy() > 0.5))
    {
      return true;
    }
    else if ((query1.Value().Occupancy() < 0.5)
             && (query2.Value().Occupancy() < 0.5))
    {
      return true;
    }
    else if ((query1.Value().Occupancy() == 0.5)
             && (query2.Value().Occupancy() == 0.5))
    {
      return true;
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
      query.Value().Component() = component;
    }
  };
  number_of_components_
      = topology_computation::ComputeConnectedComponents(
          *this, are_connected_fn, get_component_fn, mark_component_fn);
  components_valid_ = true;
  return number_of_components_;
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
CollisionMap::ExtractComponentSurfaces(
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
    const CollisionCell& current_cell = GetIndexImmutable(index).Value();
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
  return topology_computation::ExtractComponentSurfaces(*this,
                                                        get_component_fn,
                                                        is_surface_index_fn);
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
CollisionMap::ExtractFilledComponentSurfaces() const
{
  return ExtractComponentSurfaces(FILLED_COMPONENTS);
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
CollisionMap::ExtractUnknownComponentSurfaces() const
{
  return ExtractComponentSurfaces(UNKNOWN_COMPONENTS);
}

std::map<uint32_t, std::unordered_map<
    common_robotics_utilities::voxel_grid::GridIndex, uint8_t>>
CollisionMap::ExtractEmptyComponentSurfaces() const
{
  return ExtractComponentSurfaces(EMPTY_COMPONENTS);
}

topology_computation::TopologicalInvariants
CollisionMap::ComputeComponentTopology(
    const COMPONENT_TYPES component_types_to_use,
    const common_robotics_utilities::utility::LoggingFunction& logging_fn)
{
  using common_robotics_utilities::voxel_grid::GridIndex;
  UpdateConnectedComponents();
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
    const CollisionCell& current_cell = GetIndexImmutable(index).Value();
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

SignedDistanceField<double> CollisionMap::ExtractSignedDistanceFieldDouble(
    const double oob_value, const bool unknown_is_filled,
    const bool use_parallel, const bool add_virtual_border) const
{
  return ExtractSignedDistanceField<double>(
      oob_value, unknown_is_filled, use_parallel, add_virtual_border);
}

SignedDistanceField<float> CollisionMap::ExtractSignedDistanceFieldFloat(
    const float oob_value, const bool unknown_is_filled,
    const bool use_parallel, const bool add_virtual_border) const
{
  return ExtractSignedDistanceField<float>(
      oob_value, unknown_is_filled, use_parallel, add_virtual_border);
}
}  // namespace voxelized_geometry_tools
