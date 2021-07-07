#pragma once

#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <common_robotics_utilities/voxel_grid.hpp>

namespace voxelized_geometry_tools
{
namespace topology_computation
{
class NumberOfHolesAndVoids
{
private:
  int32_t num_holes_ = 0;
  int32_t num_voids_ = 0;

public:
  NumberOfHolesAndVoids() {}

  NumberOfHolesAndVoids(const int32_t num_holes, const int32_t num_voids)
      : num_holes_(num_holes), num_voids_(num_voids)
  {
    if (num_holes_ < 0)
    {
      throw std::invalid_argument("num_holes_ < 0");
    }
    if (num_voids_ < 0)
    {
      throw std::invalid_argument("num_voids_ < 0");
    }
  }

  int32_t NumHoles() const { return num_holes_; }

  int32_t NumVoids() const { return num_voids_; }
};

/// Map of connected component id -> number of holes and voids in that connected
/// component. The first three topological invariants, also known as Betti
/// numbers, are (1) the number of connected components, (2) the number of holes
/// in each connected component, and (3) the number of voids in each connected
/// component.
using TopologicalInvariants = std::map<uint32_t, NumberOfHolesAndVoids>;

using common_robotics_utilities::voxel_grid::GridIndex;

template<typename T, typename BackingStore=std::vector<T>>
int64_t MarkConnectedComponent(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& source_grid,
    const std::function<bool(const GridIndex&,
                             const GridIndex&)>& are_connected_fn,
    const std::function<uint32_t(const GridIndex&)>& get_component_fn,
    const std::function<void(const GridIndex&,
                             const uint32_t)>& mark_component_fn,
    const GridIndex& start_index,
    const uint32_t connected_component)
{
  // Make the working queue
  std::deque<GridIndex> working_queue;
  // Make a hash table to store queued indices (so we don't repeat work)
  // Let's provide an hint at the size of hashmap we'll need, since this will
  // reduce the need to resize & rehash. We're going to assume that connected
  // components, in general, will take ~1/16 of the grid in size
  // which means, with 2 cells/hash bucket, we'll initialize to grid size/32
  const size_t queued_set_size_hint
      = source_grid.GetImmutableRawData().size() / 32;
  std::unordered_set<GridIndex> queued_set(queued_set_size_hint);
  // Enqueue the starting index
  working_queue.push_back(start_index);
  queued_set.insert(start_index);
  // Helper lambda
  const auto handle_neighbor_index = [&] (
      const GridIndex& current_index, const GridIndex& neighbor_index)
  {
    if (get_component_fn(neighbor_index) == 0)
    {
      if (are_connected_fn(current_index, neighbor_index))
      {
        // Use the return from set.insert(value) to avoid the additional work
        // of if (set.count(value) == 0) { set.insert(value) }.
        const bool inserted = queued_set.insert(neighbor_index).second;
        if (inserted)
        {
          working_queue.push_back(neighbor_index);
        }
      }
    }
  };
  // Work
  int64_t marked_cells = 0;
  while (working_queue.size() > 0)
  {
    // Get an item off the queue to work with
    const GridIndex current_index = working_queue.front();
    working_queue.pop_front();
    // Mark the connected component
    mark_component_fn(current_index, connected_component);
    // Go through the possible neighbors and enqueue as needed
    // Since there are only six cases
    // (voxels must share a face to be considered connected),
    // we handle each explicitly
    handle_neighbor_index(
        current_index,
        GridIndex(current_index.X() - 1, current_index.Y(), current_index.Z()));
    handle_neighbor_index(
        current_index,
        GridIndex(current_index.X() + 1, current_index.Y(), current_index.Z()));
    handle_neighbor_index(
        current_index,
        GridIndex(current_index.X(), current_index.Y() - 1, current_index.Z()));
    handle_neighbor_index(
        current_index,
        GridIndex(current_index.X(), current_index.Y() + 1, current_index.Z()));
    handle_neighbor_index(
        current_index,
        GridIndex(current_index.X(), current_index.Y(), current_index.Z() - 1));
    handle_neighbor_index(
        current_index,
        GridIndex(current_index.X(), current_index.Y(), current_index.Z() + 1));
  }
  return marked_cells;
}

template<typename T, typename BackingStore=std::vector<T>>
uint32_t ComputeConnectedComponents(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& source_grid,
    const std::function<bool(const GridIndex&,
                             const GridIndex&)>& are_connected_fn,
    const std::function<int64_t(const GridIndex&)>& get_component_fn,
    const std::function<void(const GridIndex&,
                             const uint32_t)>& mark_component_fn)
{
  // Reset components first
  for (int64_t x_index = 0; x_index < source_grid.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < source_grid.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < source_grid.GetNumZCells(); z_index++)
      {
        const GridIndex index(x_index, y_index, z_index);
        mark_component_fn(index, 0u);
      }
    }
  }
  // Mark the components
  int64_t total_cells = source_grid.GetNumXCells()
                        * source_grid.GetNumYCells()
                        * source_grid.GetNumZCells();
  int64_t marked_cells = 0;
  uint32_t connected_components = 0;
  // Sweep through the grid
  for (int64_t x_index = 0; x_index < source_grid.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < source_grid.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < source_grid.GetNumZCells(); z_index++)
      {
        const GridIndex index(x_index, y_index, z_index);
        // Check if the cell has already been marked, if so, ignore
        if (get_component_fn(index) == 0)
        {
          // Start marking a new connected component
          connected_components++;
          const int64_t cells_marked
              = MarkConnectedComponent(source_grid,
                                       are_connected_fn,
                                       get_component_fn,
                                       mark_component_fn,
                                       index, connected_components);
          marked_cells += cells_marked;
          // Short-circuit if we've marked everything
          if (marked_cells == total_cells)
          {
            return connected_components;
          }
        }
      }
    }
  }
  return connected_components;
}

inline int32_t ComputeConnectivityOfSurfaceVertices(
    const std::unordered_map<GridIndex, uint8_t>&
      surface_vertex_connectivity)
{
  int32_t connected_components = 0;
  int64_t processed_vertices = 0;
  // Compute a hint for initial vertex components hashmap size
  // real # of surface vertices
  // surface vertices
  size_t vertex_components_size_hint = surface_vertex_connectivity.size();
  std::unordered_map<GridIndex, int32_t>
      vertex_components(vertex_components_size_hint);
  // Iterate through the vertices
  for (const auto& vertex_and_connectivity : surface_vertex_connectivity)
  {
    const GridIndex& key = vertex_and_connectivity.first;
    // First, check if the vertex has already been marked
    if (vertex_components[key] > 0)
    {
      continue;
    }
    else
    {
      // If not, we start marking a new connected component
      connected_components++;
      // Make the working queue
      std::deque<GridIndex> working_queue;
      // Make a hash table to store queued indices (so we don't repeat work)
      // Compute a hint for initial queued hashtable hashmap size
      // If we assume that most object surfaces are, in fact, intact,
      // then the first (and only) queued_hashtable will need to store an entry
      // for every vertex on the surface.
      size_t queued_set_size_hint = surface_vertex_connectivity.size();
      std::unordered_set<GridIndex> queued_set(queued_set_size_hint);
      // Add the current point
      working_queue.push_back(key);
      queued_set.insert(key);
      // Keep track of the number of vertices we've processed
      int64_t component_processed_vertices = 0;
      // Helper lambda
      const auto handle_connected_vertex = [&] (
          const GridIndex& connected_vertex)
      {
        if (queued_set.insert(connected_vertex).second)
        {
          working_queue.push_back(connected_vertex);
        }
      };
      // Loop from the queue
      while (working_queue.size() > 0)
      {
        // Get the top of thw working queue
        const GridIndex current_vertex = working_queue.front();
        working_queue.pop_front();
        component_processed_vertices++;
        vertex_components[current_vertex] = connected_components;
        // Check the six possibly-connected vertices and
        // add them to the queue if they are connected
        // Get the connectivity of our index
        uint8_t connectivity = surface_vertex_connectivity.at(current_vertex);
        // Go through the neighbors
        if ((connectivity & 0b00000001) > 0)
        {
          handle_connected_vertex(GridIndex(
              current_vertex.X(), current_vertex.Y(), current_vertex.Z() - 1));
        }
        if ((connectivity & 0b00000010) > 0)
        {
          handle_connected_vertex(GridIndex(
              current_vertex.X(), current_vertex.Y(), current_vertex.Z() + 1));
        }
        if ((connectivity & 0b00000100) > 0)
        {
          handle_connected_vertex(GridIndex(
              current_vertex.X(), current_vertex.Y() - 1, current_vertex.Z()));
        }
        if ((connectivity & 0b00001000) > 0)
        {
          handle_connected_vertex(GridIndex(
              current_vertex.X(), current_vertex.Y() + 1, current_vertex.Z()));
        }
        if ((connectivity & 0b00010000) > 0)
        {
          handle_connected_vertex(GridIndex(
              current_vertex.X() - 1, current_vertex.Y(), current_vertex.Z()));
        }
        if ((connectivity & 0b00100000) > 0)
        {
          handle_connected_vertex(GridIndex(
              current_vertex.X() + 1, current_vertex.Y(), current_vertex.Z()));
        }
      }
      processed_vertices += component_processed_vertices;
      if (processed_vertices
          == static_cast<int64_t>(surface_vertex_connectivity.size()))
      {
        break;
      }
    }
  }
  return connected_components;
}

template<typename T, typename BackingStore=std::vector<T>>
std::map<uint32_t, std::unordered_map<GridIndex, uint8_t>>
ExtractComponentSurfaces(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& source_grid,
    const std::function<int64_t(const GridIndex&)>& get_component_fn,
    const std::function<bool(const GridIndex&)>& is_surface_index_fn)
{
  std::map<uint32_t, std::unordered_map<GridIndex, uint8_t>>
      component_surfaces;
  // Loop through the grid and extract surface cells for each component
  for (int64_t x_index = 0; x_index < source_grid.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < source_grid.GetNumYCells(); y_index++)
    {
      for (int64_t z_index = 0; z_index < source_grid.GetNumZCells(); z_index++)
      {
        const GridIndex current_index(x_index, y_index, z_index);
        if (is_surface_index_fn(current_index))
        {
          const uint32_t current_component =
              static_cast<uint32_t>(get_component_fn(current_index));
          component_surfaces[current_component][current_index] = 1;
        }
      }
    }
  }
  return component_surfaces;
}

inline NumberOfHolesAndVoids ComputeHolesAndVoidsInSurface(
    const uint32_t component,
    const std::unordered_map<GridIndex, uint8_t>& surface,
    const std::function<int64_t(const GridIndex&)>& get_component_fn,
    const bool verbose)
{
  // We have a list of all voxels with an exposed surface face
  // We loop through this list of voxels, and convert each voxel
  // into 8 vertices (the corners), which we individually check:
  //
  // First - we check to see if the vertex has already been
  // evaluated
  //
  // Second - we check if the vertex is actually on the surface
  // (make sure at least one of the three adjacent vertices is
  // exposed)
  //
  // Third - insert into hashtable of surface vertices
  //
  // Once we have completed this process, we loop back through
  // the hashtable of surface vertices and compute the number
  // of distance-1 neighboring surface vertices (we do this by
  // checking each of the six potential neighbor vertices) and
  // keep a running count of all vertices with 3, 5, and 6
  // neighbors.
  //
  // Once we have evaluated all the neighbors of all surface
  // vertices, we count the number of holes in the grid using
  // the formula from Chen and Rong, "Linear Time Recognition
  // Algorithms for Topological Invariants in 3D":
  //
  // #holes = 1 + (M5 + 2 * M6 - M3) / 8
  //
  // where M5 is the number of vertices with 5 neighbors,
  // M6 is the number of vertices with 6 neighbors, and
  // M3 is the number of vertices with 3 neighbors
  //
  // Storage for surface vertices
  // Compute a hint for initial surface vertex hashmap size
  // expected # of surface vertices
  // surface cells * 8
  const size_t surface_vertices_size_hint = surface.size() * 8;
  std::unordered_set<GridIndex> surface_vertices(surface_vertices_size_hint);
  // Loop through all the surface voxels and extract surface vertices
  for (const auto& index_and_value : surface)
  {
    if (index_and_value.second == 0)
    {
      continue;
    }

    const GridIndex& current_index = index_and_value.first;
    // First, grab all six neighbors from the grid
    const int64_t xyzm1_component
        = get_component_fn(GridIndex(current_index.X(),
                                     current_index.Y(),
                                     current_index.Z() - 1));
    const int64_t xyzp1_component
        = get_component_fn(GridIndex(current_index.X(),
                                     current_index.Y(),
                                     current_index.Z() - 1));
    const int64_t xym1z_component
        = get_component_fn(GridIndex(current_index.X(),
                                     current_index.Y() - 1,
                                     current_index.Z()));
    const int64_t xyp1z_component
        = get_component_fn(GridIndex(current_index.X(),
                                     current_index.Y() + 1,
                                     current_index.Z()));
    const int64_t xm1yz_component
        = get_component_fn(GridIndex(current_index.X() - 1,
                                     current_index.Y(),
                                     current_index.Z()));
    const int64_t xp1yz_component
        = get_component_fn(GridIndex(current_index.X() + 1,
                                     current_index.Y(),
                                     current_index.Z()));
    // Generate all 8 vertices for the current voxel, check if an adjacent
    // vertex is on the surface, and insert it if so
    // First, check the (-,-,-) vertex
    if (component != xyzm1_component
        || component != xym1z_component
        || component != xm1yz_component)
    {
      const GridIndex vertex1(current_index.X(),
                              current_index.Y(),
                              current_index.Z());
      surface_vertices.insert(vertex1);
    }
    // Second, check the (-,-,+) vertex
    if (component != xyzp1_component
        || component != xym1z_component
        || component != xm1yz_component)
    {
      const GridIndex vertex2(current_index.X(),
                              current_index.Y(),
                              current_index.Z() + 1);
      surface_vertices.insert(vertex2);
    }
    // Third, check the (-,+,-) vertex
    if (component != xyzm1_component
        || component != xyp1z_component
        || component != xm1yz_component)
    {
      const GridIndex vertex3(current_index.X(),
                              current_index.Y() + 1,
                              current_index.Z());
      surface_vertices.insert(vertex3);
    }
    // Fourth, check the (-,+,+) vertex
    if (component != xyzp1_component
        || component != xyp1z_component
        || component != xm1yz_component)
    {
      const GridIndex vertex4(current_index.X(),
                              current_index.Y() + 1,
                              current_index.Z() + 1);
      surface_vertices.insert(vertex4);
    }
    // Fifth, check the (+,-,-) vertex
    if (component != xyzm1_component
        || component != xym1z_component
        || component != xp1yz_component)
    {
      const GridIndex vertex5(current_index.X() + 1,
                              current_index.Y(),
                              current_index.Z());
      surface_vertices.insert(vertex5);
    }
    // Sixth, check the (+,-,+) vertex
    if (component != xyzp1_component
        || component != xym1z_component
        || component != xp1yz_component)
    {
      const GridIndex vertex6(current_index.X() + 1,
                              current_index.Y(),
                              current_index.Z() + 1);
      surface_vertices.insert(vertex6);
    }
    // Seventh, check the (+,+,-) vertex
    if (component != xyzm1_component
        || component != xyp1z_component
        || component != xp1yz_component)
    {
      const GridIndex vertex7(current_index.X() + 1,
                              current_index.Y() + 1,
                              current_index.Z());
      surface_vertices.insert(vertex7);
    }
    // Eighth, check the (+,+,+) vertex
    if (component != xyzp1_component
        || component != xyp1z_component
        || component != xp1yz_component)
    {
      const GridIndex vertex8(current_index.X() + 1,
                              current_index.Y() + 1,
                              current_index.Z() + 1);
      surface_vertices.insert(vertex8);
    }
  }
  if (verbose)
  {
    std::cerr << "Surface with " << surface.size() << " voxels has "
              << surface_vertices.size() << " surface vertices" << std::endl;
  }
  // Iterate through the surface vertices and count the neighbors of each vertex
  int32_t M3 = 0;
  int32_t M5 = 0;
  int32_t M6 = 0;
  // Store the connectivity of each vertex
  // Compute a hint for initial vertex connectivity hashmap size
  // real # of surface vertices
  // surface vertices
  const size_t vertex_connectivity_size_hint = surface_vertices.size();
  std::unordered_map<GridIndex, uint8_t> vertex_connectivity(
        vertex_connectivity_size_hint);
  for (const GridIndex& key : surface_vertices)
  {
    // Insert into the connectivity map
    vertex_connectivity[key] = 0b00000000;
    // Check the six edges from the current vertex and count the number of
    // exposed edges (an edge is exposed if the at least one of the four
    // surrounding voxels is not part of the current component)
    int32_t edge_count = 0;
    // First, get the 8 voxels that surround the current vertex
    const int64_t xm1ym1zm1_component
        = get_component_fn(GridIndex(key.X() - 1, key.Y() - 1, key.Z() - 1));
    const int64_t xm1ym1zp1_component
        = get_component_fn(GridIndex(key.X() - 1, key.Y() - 1, key.Z() + 0));
    const int64_t xm1yp1zm1_component
        = get_component_fn(GridIndex(key.X() - 1, key.Y() + 0, key.Z() - 1));
    const int64_t xm1yp1zp1_component
        = get_component_fn(GridIndex(key.X() - 1, key.Y() + 0, key.Z() + 0));
    const int64_t xp1ym1zm1_component
        = get_component_fn(GridIndex(key.X() + 0, key.Y() - 1, key.Z() - 1));
    const int64_t xp1ym1zp1_component
        = get_component_fn(GridIndex(key.X() + 0, key.Y() - 1, key.Z() + 0));
    const int64_t xp1yp1zm1_component
        = get_component_fn(GridIndex(key.X() + 0, key.Y() + 0, key.Z() - 1));
    const int64_t xp1yp1zp1_component
        = get_component_fn(GridIndex(key.X() + 0, key.Y() + 0, key.Z() + 0));
    // Check the "z- down" edge
    if (component != xm1ym1zm1_component || component != xm1yp1zm1_component
        || component != xp1ym1zm1_component || component != xp1yp1zm1_component)
    {
      if (!(component != xm1ym1zm1_component
            && component != xm1yp1zm1_component
            && component != xp1ym1zm1_component
            && component != xp1yp1zm1_component))
      {
        edge_count++;
        vertex_connectivity[key] |= 0b00000001;
      }
    }
    // Check the "z+ up" edge
    if (component != xm1ym1zp1_component || component != xm1yp1zp1_component
        || component != xp1ym1zp1_component || component != xp1yp1zp1_component)
    {
      if (!(component != xm1ym1zp1_component
            && component != xm1yp1zp1_component
            && component != xp1ym1zp1_component
            && component != xp1yp1zp1_component))
      {
        edge_count++;
        vertex_connectivity[key] |= 0b00000010;
      }
    }
    // Check the "y- right" edge
    if (component != xm1ym1zm1_component || component != xm1ym1zp1_component
        || component != xp1ym1zm1_component || component != xp1ym1zp1_component)
    {
      if (!(component != xm1ym1zm1_component
            && component != xm1ym1zp1_component
            && component != xp1ym1zm1_component
            && component != xp1ym1zp1_component))
      {
        edge_count++;
        vertex_connectivity[key] |= 0b00000100;
      }
    }
    // Check the "y+ left" edge
    if (component != xm1yp1zm1_component || component != xm1yp1zp1_component
        || component != xp1yp1zm1_component || component != xp1yp1zp1_component)
    {
      if (!(component != xm1yp1zm1_component
            && component != xm1yp1zp1_component
            && component != xp1yp1zm1_component
            && component != xp1yp1zp1_component))
      {
        edge_count++;
        vertex_connectivity[key] |= 0b00001000;
      }
    }
    // Check the "x- back" edge
    if (component != xm1ym1zm1_component || component != xm1ym1zp1_component
        || component != xm1yp1zm1_component || component != xm1yp1zp1_component)
    {
      if (!(component != xm1ym1zm1_component
            && component != xm1ym1zp1_component
            && component != xm1yp1zm1_component
            && component != xm1yp1zp1_component))
      {
        edge_count++;
        vertex_connectivity[key] |= 0b00010000;
      }
    }
    // Check the "x+ front" edge
    if (component != xp1ym1zm1_component || component != xp1ym1zp1_component
        || component != xp1yp1zm1_component || component != xp1yp1zp1_component)
    {
      if (!(component != xp1ym1zm1_component
            && component != xp1ym1zp1_component
            && component != xp1yp1zm1_component
            && component != xp1yp1zp1_component))
      {
        edge_count++;
        vertex_connectivity[key] |= 0b00100000;
      }
    }
    // Increment M counts
    if (edge_count == 3)
    {
      M3++;
    }
    else if (edge_count == 5)
    {
      M5++;
    }
    else if (edge_count == 6)
    {
      M6++;
    }
  }
  // Check to see if the set of vertices is connected.
  // If not, our object contains void(s)
  const int32_t number_of_surfaces
      = ComputeConnectivityOfSurfaceVertices(vertex_connectivity);
  const int32_t number_of_voids = number_of_surfaces - 1;
  // Compute the number of holes in the surface
  const int32_t raw_number_of_holes = 1 + ((M5 + (2 * M6) - M3) / 8);
  const int32_t number_of_holes = raw_number_of_holes + number_of_voids;
  if (verbose)
  {
    std::cout << "Processing surface with M3 = " << M3 << " M5 = " << M5
              << " M6 = " << M6 << " holes = " << number_of_holes
              << " surfaces = " << number_of_surfaces
              << " voids = " << number_of_voids << std::endl;
  }
  return NumberOfHolesAndVoids(number_of_holes, number_of_voids);
}

template<typename T, typename BackingStore=std::vector<T>>
TopologicalInvariants ComputeComponentTopology(
    const common_robotics_utilities::voxel_grid
        ::VoxelGridBase<T, BackingStore>& source_grid,
    const std::function<int64_t(const GridIndex&)>& get_component_fn,
    const std::function<bool(const GridIndex&)>& is_surface_index_fn,
    const bool verbose)
{
  // Extract the surfaces of each connected component
  const std::map<uint32_t, std::unordered_map<GridIndex, uint8_t>>
      component_surfaces = ExtractComponentSurfaces(source_grid,
                                                    get_component_fn,
                                                    is_surface_index_fn);
  // Compute the number of holes in each surface
  TopologicalInvariants component_holes_and_voids;
  for (const auto& component_and_surface : component_surfaces)
  {
    const uint32_t component_number = component_and_surface.first;
    const std::unordered_map<GridIndex, uint8_t>& component_surface
        = component_and_surface.second;
    const NumberOfHolesAndVoids number_of_holes_and_voids
        = ComputeHolesAndVoidsInSurface(
            component_number, component_surface, get_component_fn, verbose);
    component_holes_and_voids[component_number] = number_of_holes_and_voids;
  }
  return component_holes_and_voids;
}

// Extracts the active indices from a surface map as a vector, which is useful
// in contexts where a 1-dimensional index into the surface is needed
inline std::vector<GridIndex> ExtractStaticSurface(
    const std::unordered_map<GridIndex, uint8_t>& raw_surface)
{
  std::vector<GridIndex> static_surface;
  // This may be larger than the actual surface we'll extract
  static_surface.reserve(raw_surface.size());
  for (const auto& index_and_value : raw_surface)
  {
    if (index_and_value.second == 1)
    {
      static_surface.push_back(index_and_value.first);
    }
  }
  // Try to reclaim the unnecessary vector capacity
  static_surface.shrink_to_fit();
  return static_surface;
}

inline std::unordered_map<GridIndex, uint8_t> ConvertToDynamicSurface(
    const std::vector<GridIndex>& static_surface)
{
  std::unordered_map<GridIndex, uint8_t> dynamic_surface(
        static_surface.size());
  for (const GridIndex& grid_index : static_surface)
  {
    dynamic_surface[grid_index] = 1u;
  }
  return dynamic_surface;
}

inline std::unordered_map<GridIndex, size_t> BuildSurfaceIndexMap(
    const std::vector<GridIndex>& static_surface)
{
  std::unordered_map<GridIndex, size_t> dynamic_surface(static_surface.size());
  for (size_t idx = 0; idx < static_surface.size(); idx++)
  {
    const GridIndex& current_index = static_surface[idx];
    dynamic_surface[current_index] = idx;
  }
  return dynamic_surface;
}
}  // namespace topology_computation
}  // namespace voxelized_geometry_tools
