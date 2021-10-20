#pragma once

#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <voxelized_geometry_tools/collision_map.hpp>

// TODO(calderpg) Factor this out to support different voxel grid types.
namespace voxelized_geometry_tools
{
/// This sets all voxels intersected by the provided triangle to filled. Note
/// that it sets voxels filled regardless of whether or not the voxel cell
/// center is "inside" or "outside" the triangle.
/// Note: this uses a conservative approximation of triangle-box collision that
/// will identify some extraneous voxels as colliding. For its use so far, this
/// has better than adding too few, since we generally want a watertight layer
/// of surface voxels.
void RasterizeTriangle(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    const size_t triangle_index,
    voxelized_geometry_tools::CollisionMap& collision_map,
    const bool enforce_collision_map_contains_triangle = true);

/// This sets all voxels intersected by the provided mesh to filled. Note
/// that it sets voxels filled regardless of whether or not the voxel cell
/// center is "inside" or "outside" the intersecting triangle(s) of the mesh.
void RasterizeMesh(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    voxelized_geometry_tools::CollisionMap& collision_map,
    const bool enforce_collision_map_contains_mesh = true);

/// This sets all voxels intersected by the provided mesh to filled. Note
/// that it sets voxels filled regardless of whether or not the voxel cell
/// center is "inside" or "outside" the intersecting triangle(s) of the mesh.
/// The generated CollisionMap will fully contain the provided mesh.
voxelized_geometry_tools::CollisionMap RasterizeMeshIntoCollisionMap(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    const double resolution);
}  // namespace voxelized_geometry_tools

