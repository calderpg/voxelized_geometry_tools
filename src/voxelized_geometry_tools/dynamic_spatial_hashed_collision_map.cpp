#include <voxelized_geometry_tools/dynamic_spatial_hashed_collision_map.hpp>

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/dynamic_spatial_hashed_voxel_grid.hpp>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>

namespace voxelized_geometry_tools
{
VGT_NAMESPACE_BEGIN
/// We need to implement cloning.
std::unique_ptr<common_robotics_utilities::voxel_grid
    ::DynamicSpatialHashedVoxelGridBase<
        CollisionCell, std::vector<CollisionCell>>>
DynamicSpatialHashedCollisionMap::DoClone() const
{
  return std::unique_ptr<DynamicSpatialHashedCollisionMap>(
      new DynamicSpatialHashedCollisionMap(*this));
}

/// We need to serialize the frame and locked flag.
uint64_t DynamicSpatialHashedCollisionMap::DerivedSerializeSelf(
    std::vector<uint8_t>& buffer,
    const CollisionCellSerializer& value_serializer) const
{
  CRU_UNUSED(value_serializer);
  const uint64_t start_size = buffer.size();
  common_robotics_utilities::serialization::SerializeString(frame_, buffer);
  const uint64_t bytes_written = buffer.size() - start_size;
  return bytes_written;
}

/// We need to deserialize the frame and locked flag.
uint64_t DynamicSpatialHashedCollisionMap::DerivedDeserializeSelf(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset,
    const CollisionCellDeserializer& value_deserializer)
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

bool DynamicSpatialHashedCollisionMap::OnMutableAccess(
    const Eigen::Vector4d& location)
{
  CRU_UNUSED(location);
  return true;
}

uint64_t DynamicSpatialHashedCollisionMap::Serialize(
    const DynamicSpatialHashedCollisionMap& map, std::vector<uint8_t>& buffer)
{
  return map.SerializeSelf(buffer, CollisionCell::Serialize);
}

DynamicSpatialHashedCollisionMap::DeserializedDynamicSpatialHashedCollisionMap
DynamicSpatialHashedCollisionMap::Deserialize(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  DynamicSpatialHashedCollisionMap temp_map;
  const uint64_t bytes_read
      = temp_map.DeserializeSelf(
          buffer, starting_offset, CollisionCell::Deserialize);
  return common_robotics_utilities::serialization::MakeDeserialized(
      temp_map, bytes_read);
}

void DynamicSpatialHashedCollisionMap::SaveToFile(
    const DynamicSpatialHashedCollisionMap& map,
    const std::string& filepath,
    const bool compress)
{
  std::vector<uint8_t> buffer;
  DynamicSpatialHashedCollisionMap::Serialize(map, buffer);
  std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
  if (compress)
  {
    output_file.write("DMGZ", 4);
    const std::vector<uint8_t> compressed
        = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
    const size_t serialized_size = compressed.size();
    output_file.write(
        reinterpret_cast<const char*>(compressed.data()),
        static_cast<std::streamsize>(serialized_size));
  }
  else
  {
    output_file.write("DMGR", 4);
    const size_t serialized_size = buffer.size();
    output_file.write(
        reinterpret_cast<const char*>(buffer.data()),
        static_cast<std::streamsize>(serialized_size));
  }
  output_file.close();
}

DynamicSpatialHashedCollisionMap
DynamicSpatialHashedCollisionMap::LoadFromFile(const std::string& filepath)
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
    if (header_string == "DMGZ")
    {
      const std::vector<uint8_t> decompressed
          = common_robotics_utilities::zlib_helpers
              ::DecompressBytes(file_buffer);
      return DynamicSpatialHashedCollisionMap::Deserialize(
          decompressed, 0).Value();
    }
    else if (header_string == "DMGR")
    {
      return DynamicSpatialHashedCollisionMap::Deserialize(
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
VGT_NAMESPACE_END
}  // namespace voxelized_geometry_tools
