#include <voxelized_geometry_tools/occupancy_map.hpp>

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

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
uint64_t OccupancyCell::Serialize(
    const OccupancyCell& cell, std::vector<uint8_t>& buffer)
{
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeMemcpyable<float>(
      cell.Occupancy(), buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

OccupancyCell::DeserializedOccupancyCell OccupancyCell::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  uint64_t current_position = starting_offset;
  const auto occupancy_deserialized
      = common_robotics_utilities::serialization
          ::DeserializeMemcpyable<float>(buffer, current_position);
  current_position += occupancy_deserialized.BytesRead();
  const OccupancyCell cell(occupancy_deserialized.Value());
  // Figure out how many bytes were read
  const uint64_t bytes_read = current_position - starting_offset;
  return common_robotics_utilities::serialization::MakeDeserialized(
      cell, bytes_read);
}

/// We need to implement cloning.
std::unique_ptr<common_robotics_utilities::voxel_grid
    ::VoxelGridBase<OccupancyCell, std::vector<OccupancyCell>>>
OccupancyMap::DoClone() const
{
  return std::unique_ptr<OccupancyMap>(new OccupancyMap(*this));
}

uint64_t OccupancyMap::DerivedSerializeSelf(
    std::vector<uint8_t>& buffer,
    const OccupancyCellSerializer& value_serializer) const
{
  CRU_UNUSED(value_serializer);
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeString(frame_, buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

uint64_t OccupancyMap::DerivedDeserializeSelf(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
    const OccupancyCellDeserializer& value_deserializer)
{
  CRU_UNUSED(value_deserializer);
  // Enforce uniform voxel sizes
  EnforceUniformVoxelSize();
  // Deserialize our additional members
  uint64_t current_position = starting_offset;
  const auto frame_deserialized
      = common_robotics_utilities::serialization::DeserializeString<char>(
          buffer, current_position);
  frame_ = frame_deserialized.Value();
  current_position += frame_deserialized.BytesRead();
  // Figure out how many bytes were read
  const uint64_t bytes_read = current_position - starting_offset;
  return bytes_read;
}

bool OccupancyMap::OnMutableAccess(
    const int64_t x_index, const int64_t y_index, const int64_t z_index)
{
  CRU_UNUSED(x_index);
  CRU_UNUSED(y_index);
  CRU_UNUSED(z_index);
  return true;
}

bool OccupancyMap::OnMutableRawAccess()
{
  return true;
}

uint64_t OccupancyMap::Serialize(
    const OccupancyMap& map, std::vector<uint8_t>& buffer)
{
  return map.SerializeSelf(buffer, OccupancyCell::Serialize);
}

OccupancyMap::DeserializedOccupancyMap OccupancyMap::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  OccupancyMap temp_map;
  const uint64_t bytes_read = temp_map.DeserializeSelf(
      buffer, starting_offset, OccupancyCell::Deserialize);
  return common_robotics_utilities::serialization::MakeDeserialized(
      temp_map, bytes_read);
}

void OccupancyMap::SaveToFile(
    const OccupancyMap& map,
    const std::string& filepath,
    const bool compress)
{
  std::vector<uint8_t> buffer;
  OccupancyMap::Serialize(map, buffer);
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

OccupancyMap OccupancyMap::LoadFromFile(const std::string& filepath)
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
      return OccupancyMap::Deserialize(decompressed, 0).Value();
    }
    else if (header_string == "CMGR")
    {
      return OccupancyMap::Deserialize(file_buffer, 0).Value();
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

common_robotics_utilities::OwningMaybe<bool> OccupancyMap::IsSurfaceIndex(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return IsSurfaceIndex(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool> OccupancyMap::IsSurfaceIndex(
    const int64_t x_index, const int64_t y_index,
    const int64_t z_index) const
{
  // First, we make sure that indices are within bounds
  // Out of bounds indices are NOT surface cells
  if (!CheckGridIndexInBounds(x_index, y_index, z_index))
  {
    return common_robotics_utilities::OwningMaybe<bool>();
  }
  // Check all 26 possible neighbors
  const int64_t min_x_check = std::max(INT64_C(0), x_index - 1);
  const int64_t max_x_check = std::min(NumXVoxels() - 1, x_index + 1);
  const int64_t min_y_check = std::max(INT64_C(0), y_index - 1);
  const int64_t max_y_check = std::min(NumYVoxels() - 1, y_index + 1);
  const int64_t min_z_check = std::max(INT64_C(0), z_index - 1);
  const int64_t max_z_check = std::min(NumZVoxels() - 1, z_index + 1);
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

SignedDistanceField<double> OccupancyMap::ExtractSignedDistanceFieldDouble(
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return ExtractSignedDistanceField<double>(parameters);
}

SignedDistanceField<float> OccupancyMap::ExtractSignedDistanceFieldFloat(
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return ExtractSignedDistanceField<float>(parameters);
}
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
