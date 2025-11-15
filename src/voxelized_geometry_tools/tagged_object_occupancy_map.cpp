#include <voxelized_geometry_tools/tagged_object_occupancy_map.hpp>

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
uint64_t TaggedObjectOccupancyCell::Serialize(
    const TaggedObjectOccupancyCell& cell, std::vector<uint8_t>& buffer)
{
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeMemcpyable<float>(
      cell.Occupancy(), buffer);
  common_robotics_utilities::serialization::SerializeMemcpyable<uint32_t>(
      cell.ObjectId(), buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

TaggedObjectOccupancyCell::DeserializedTaggedObjectOccupancyCell
TaggedObjectOccupancyCell::Deserialize(
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
  const TaggedObjectOccupancyCell cell(
      occupancy_deserialized.Value(), object_id_deserialized.Value());
  // Figure out how many bytes were read
  const uint64_t bytes_read = current_position - starting_offset;
  return common_robotics_utilities::serialization::MakeDeserialized(
      cell, bytes_read);
}

/// We need to implement cloning.
std::unique_ptr<common_robotics_utilities::voxel_grid
    ::VoxelGridBase<TaggedObjectOccupancyCell,
                    std::vector<TaggedObjectOccupancyCell>>>
TaggedObjectOccupancyMap::DoClone() const
{
  return std::unique_ptr<TaggedObjectOccupancyMap>(
      new TaggedObjectOccupancyMap(*this));
}

/// We need to serialize the frame and locked flag.
uint64_t TaggedObjectOccupancyMap::DerivedSerializeSelf(
    std::vector<uint8_t>& buffer,
    const TaggedObjectOccupancyCellSerializer& value_serializer) const
{
  CRU_UNUSED(value_serializer);
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeString(frame_, buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

/// We need to deserialize the frame and locked flag.
uint64_t TaggedObjectOccupancyMap::DerivedDeserializeSelf(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
    const TaggedObjectOccupancyCellDeserializer& value_deserializer)
{
  CRU_UNUSED(value_deserializer);
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

bool TaggedObjectOccupancyMap::OnMutableAccess(
    const int64_t x_index, const int64_t y_index, const int64_t z_index)
{
  CRU_UNUSED(x_index);
  CRU_UNUSED(y_index);
  CRU_UNUSED(z_index);
  return true;
}

bool TaggedObjectOccupancyMap::OnMutableRawAccess()
{
  return true;
}

uint64_t TaggedObjectOccupancyMap::Serialize(
    const TaggedObjectOccupancyMap& map, std::vector<uint8_t>& buffer)
{
  return map.SerializeSelf(buffer, TaggedObjectOccupancyCell::Serialize);
}

TaggedObjectOccupancyMap::DeserializedTaggedObjectOccupancyMap
TaggedObjectOccupancyMap::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  TaggedObjectOccupancyMap temp_map;
  const uint64_t bytes_read = temp_map.DeserializeSelf(
      buffer, starting_offset, TaggedObjectOccupancyCell::Deserialize);
  return common_robotics_utilities::serialization::MakeDeserialized(
      temp_map, bytes_read);
}

void TaggedObjectOccupancyMap::SaveToFile(
    const TaggedObjectOccupancyMap& map,
    const std::string& filepath,
    const bool compress)
{
  std::vector<uint8_t> buffer;
  TaggedObjectOccupancyMap::Serialize(map, buffer);
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

TaggedObjectOccupancyMap TaggedObjectOccupancyMap::LoadFromFile(
    const std::string& filepath)
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
      return TaggedObjectOccupancyMap::Deserialize(decompressed, 0).Value();
    }
    else if (header_string == "TMGR")
    {
      return TaggedObjectOccupancyMap::Deserialize(file_buffer, 0).Value();
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
TaggedObjectOccupancyMap::IsSurfaceIndex(
    const common_robotics_utilities::voxel_grid::GridIndex& index) const
{
  return IsSurfaceIndex(index.X(), index.Y(), index.Z());
}

common_robotics_utilities::OwningMaybe<bool>
TaggedObjectOccupancyMap::IsSurfaceIndex(
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

SignedDistanceField<double>
TaggedObjectOccupancyMap::ExtractSignedDistanceFieldDouble(
    const std::vector<uint32_t>& objects_to_use,
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return ExtractSignedDistanceField<double>(objects_to_use, parameters);
}

SignedDistanceField<float>
TaggedObjectOccupancyMap::ExtractSignedDistanceFieldFloat(
    const std::vector<uint32_t>& objects_to_use,
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return ExtractSignedDistanceField<float>(objects_to_use, parameters);
}

std::map<uint32_t, SignedDistanceField<double>>
TaggedObjectOccupancyMap::MakeSeparateObjectSDFsDouble(
    const std::vector<uint32_t>& object_ids,
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return MakeSeparateObjectSDFs<double>(object_ids, parameters);
}

std::map<uint32_t, SignedDistanceField<float>>
TaggedObjectOccupancyMap::MakeSeparateObjectSDFsFloat(
    const std::vector<uint32_t>& object_ids,
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return MakeSeparateObjectSDFs<float>(object_ids, parameters);
}

std::map<uint32_t, SignedDistanceField<double>>
TaggedObjectOccupancyMap::MakeAllObjectSDFsDouble(
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return MakeAllObjectSDFs<double>(parameters);
}

std::map<uint32_t, SignedDistanceField<float>>
TaggedObjectOccupancyMap::MakeAllObjectSDFsFloat(
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return MakeAllObjectSDFs<float>(parameters);
}

SignedDistanceField<double>
TaggedObjectOccupancyMap::ExtractFreeAndNamedObjectsSignedDistanceFieldDouble(
    const SignedDistanceFieldGenerationParameters<double>& parameters) const
{
  return ExtractFreeAndNamedObjectsSignedDistanceField<double>(parameters);
}

SignedDistanceField<float>
TaggedObjectOccupancyMap::ExtractFreeAndNamedObjectsSignedDistanceFieldFloat(
    const SignedDistanceFieldGenerationParameters<float>& parameters) const
{
  return ExtractFreeAndNamedObjectsSignedDistanceField<float>(parameters);
}
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
