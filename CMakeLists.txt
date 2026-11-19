cmake_minimum_required(VERSION 3.28...4.2)
project(voxelized_geometry_tools VERSION 0.1.0 LANGUAGES C CXX)

include(CheckIPOSupported)
include(CheckLanguage)
include(CTest)
include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(VOXELIZED_GEOMETRY_TOOLS_BUILD_EXAMPLES "Build the examples" ON)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(common_robotics_utilities REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCL)
# NVIDIA's packages install CUDA in /usr/local/cuda without putting nvcc on
# PATH. FindCUDA looked there, so keep finding it there.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  find_program(VOXELIZED_GEOMETRY_TOOLS_NVCC nvcc PATHS /usr/local/cuda/bin)
  if(VOXELIZED_GEOMETRY_TOOLS_NVCC)
    set(CMAKE_CUDA_COMPILER ${VOXELIZED_GEOMETRY_TOOLS_NVCC})
  endif()
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

rosidl_generate_interfaces(${PROJECT_NAME}_interfaces
  msg/DynamicSpatialHashedOccupancyMapMessage.msg
  msg/OccupancyComponentMapMessage.msg
  msg/OccupancyMapMessage.msg
  msg/SignedDistanceFieldMessage.msg
  msg/TaggedObjectOccupancyComponentMapMessage.msg
  msg/TaggedObjectOccupancyMapMessage.msg
  DEPENDENCIES std_msgs)
rosidl_get_typesupport_target(interfaces_cpp
  ${PROJECT_NAME}_interfaces rosidl_typesupport_cpp)

# Defaults when this project is built on its own, as colcon does. A parent
# project or the command line can override them.
if(PROJECT_IS_TOP_LEVEL)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  endif()
  if(NOT DEFINED CMAKE_COMPILE_WARNING_AS_ERROR)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
  endif()
  if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    check_ipo_supported(RESULT ipo_supported LANGUAGES CXX)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ${ipo_supported})
  endif()
endif()

# Flags for this project's own targets only; consumers don't inherit them.
add_library(vgt_compile_options INTERFACE)
set(vgt_warning_flags
  -Wall -Wextra -Wconversion -Wshadow -Wold-style-cast -Wpessimizing-move
  -Wuninitialized -Wmissing-declarations -Werror=non-virtual-dtor)
target_compile_options(vgt_compile_options INTERFACE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:${vgt_warning_flags}>")

# We don't depend on Drake, but we do use different build flags if present.
find_package(drake QUIET)
if(drake_FOUND)
  message(STATUS "Drake found, disabling -march=native")
else()
  message(STATUS "Drake NOT found, enabling -march=native")
  target_compile_options(vgt_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

# Common settings for every library and executable built here.
function(vgt_configure target)
  target_link_libraries(${target} PRIVATE $<BUILD_INTERFACE:vgt_compile_options>)
endfunction()

# Add a library with an alias in the project namespace.
function(vgt_library name)
  add_library(${name} ${ARGN})
  add_library(${PROJECT_NAME}::${name} ALIAS ${name})
  target_compile_features(${name} PUBLIC cxx_std_17)
  vgt_configure(${name})
endfunction()

# The device helpers only need this project's own headers, which are installed
# with the main library.
function(vgt_device_library name)
  vgt_library(${name} ${ARGN})
  target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
endfunction()

set(src src/${PROJECT_NAME})

vgt_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
  PRIVATE
    ${src}/dynamic_spatial_hashed_occupancy_map.cpp
    ${src}/mesh_rasterizer.cpp
    ${src}/occupancy_component_map.cpp
    ${src}/occupancy_map.cpp
    ${src}/occupancy_map_conversions.cpp
    ${src}/signed_distance_field.cpp
    ${src}/signed_distance_field_generation.cpp
    ${src}/tagged_object_occupancy_component_map.cpp
    ${src}/tagged_object_occupancy_map.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
      include/voxelized_geometry_tools/cl.hpp
      include/voxelized_geometry_tools/cpu_pointcloud_voxelization.hpp
      include/voxelized_geometry_tools/cuda_voxelization_helpers.h
      include/voxelized_geometry_tools/device_pointcloud_voxelization.hpp
      include/voxelized_geometry_tools/device_voxelization_interface.hpp
      include/voxelized_geometry_tools/dynamic_spatial_hashed_occupancy_map.hpp
      include/voxelized_geometry_tools/mesh_rasterizer.hpp
      include/voxelized_geometry_tools/occupancy_component_map.hpp
      include/voxelized_geometry_tools/occupancy_map.hpp
      include/voxelized_geometry_tools/occupancy_map_conversions.hpp
      include/voxelized_geometry_tools/opencl_voxelization_helpers.h
      include/voxelized_geometry_tools/pointcloud_voxelization.hpp
      include/voxelized_geometry_tools/pointcloud_voxelization_interface.hpp
      include/voxelized_geometry_tools/pointcloud_voxelization_ros_interface.hpp
      include/voxelized_geometry_tools/ros_interface.hpp
      include/voxelized_geometry_tools/signed_distance_field.hpp
      include/voxelized_geometry_tools/signed_distance_field_generation.hpp
      include/voxelized_geometry_tools/tagged_object_occupancy_component_map.hpp
      include/voxelized_geometry_tools/tagged_object_occupancy_map.hpp
      include/voxelized_geometry_tools/topology_computation.hpp
      include/voxelized_geometry_tools/vgt_namespace.hpp)
target_link_libraries(${PROJECT_NAME} PUBLIC
  common_robotics_utilities::common_robotics_utilities
  Eigen3::Eigen)

# Device voxelization helpers: the real CUDA and OpenCL backends when their
# toolchains are found, stubs otherwise (the stubs are also built next to the
# real ones so that both keep compiling).
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA found. Building CUDA voxelization helpers")
  vgt_device_library(${PROJECT_NAME}_cuda_voxelization_helpers
    ${src}/cuda_voxelization_helpers.cu)
  target_link_libraries(${PROJECT_NAME}_cuda_voxelization_helpers
    PRIVATE CUDA::cuda_driver CUDA::cudart)
  target_compile_options(${PROJECT_NAME}_cuda_voxelization_helpers PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  set_target_properties(${PROJECT_NAME}_cuda_voxelization_helpers PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  vgt_device_library(${PROJECT_NAME}_dummy_cuda_voxelization_helpers
    ${src}/dummy_cuda_voxelization_helpers.cc)
else()
  message(WARNING "CUDA not found. Building stubs for CUDA helpers.")
  vgt_device_library(${PROJECT_NAME}_cuda_voxelization_helpers
    ${src}/dummy_cuda_voxelization_helpers.cc)
endif()

if(OpenCL_FOUND)
  message(STATUS "OpenCL found. Building OpenCL voxelization helpers")
  vgt_device_library(${PROJECT_NAME}_opencl_voxelization_helpers
    ${src}/opencl_voxelization_helpers.cc)
  target_link_libraries(${PROJECT_NAME}_opencl_voxelization_helpers
    PRIVATE Eigen3::Eigen OpenCL::OpenCL)
  vgt_device_library(${PROJECT_NAME}_dummy_opencl_voxelization_helpers
    ${src}/dummy_opencl_voxelization_helpers.cc)
else()
  message(WARNING "OpenCL not found. Building stubs for OpenCL helpers.")
  vgt_device_library(${PROJECT_NAME}_opencl_voxelization_helpers
    ${src}/dummy_opencl_voxelization_helpers.cc)
endif()

vgt_library(${PROJECT_NAME}_pointcloud_voxelization
  ${src}/cpu_pointcloud_voxelization.cpp
  ${src}/device_pointcloud_voxelization.cpp
  ${src}/pointcloud_voxelization.cpp)
target_link_libraries(${PROJECT_NAME}_pointcloud_voxelization
  PUBLIC ${PROJECT_NAME}
  PRIVATE
    ${PROJECT_NAME}_cuda_voxelization_helpers
    ${PROJECT_NAME}_opencl_voxelization_helpers)

vgt_library(${PROJECT_NAME}_ros_interface ${src}/ros_interface.cpp)
target_link_libraries(${PROJECT_NAME}_ros_interface
  PUBLIC
    ${PROJECT_NAME}
    rclcpp::rclcpp
    ${std_msgs_TARGETS}
    ${visualization_msgs_TARGETS}
    ${interfaces_cpp})

vgt_library(${PROJECT_NAME}_pointcloud_voxelization_ros_interface
  ${src}/pointcloud_voxelization_ros_interface.cpp)
target_link_libraries(${PROJECT_NAME}_pointcloud_voxelization_ros_interface
  PUBLIC
    ${PROJECT_NAME}_pointcloud_voxelization
    ${sensor_msgs_TARGETS})

if(VOXELIZED_GEOMETRY_TOOLS_BUILD_EXAMPLES)
  function(vgt_example name source)
    add_executable(${name} example/${source})
    vgt_configure(${name})
    target_link_libraries(${name} PRIVATE
      ${PROJECT_NAME}_ros_interface
      ${PROJECT_NAME}_pointcloud_voxelization)
  endfunction()
  vgt_example(estimate_distance_example estimate_distance.cpp)
  vgt_example(pointcloud_voxelization pointcloud_voxelization.cpp)
  vgt_example(spatial_segments spatial_segments.cpp)
  vgt_example(voxel_raycasting voxel_raycasting.cpp)
  vgt_example(voxelized_geometry_tools_tutorial tutorial.cpp)
  install(TARGETS
      estimate_distance_example
      pointcloud_voxelization
      spatial_segments
      voxelized_geometry_tools_tutorial
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
endif()

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  function(vgt_test name library)
    add_executable(${name} test/${name}.cpp)
    vgt_configure(${name})
    target_link_libraries(${name} PRIVATE ${library} GTest::gtest_main)
    gtest_discover_tests(${name})
  endfunction()
  vgt_test(mesh_rasterization_test ${PROJECT_NAME})
  vgt_test(occupancy_map_conversions_test ${PROJECT_NAME})
  vgt_test(pointcloud_voxelization_test ${PROJECT_NAME}_pointcloud_voxelization)
  vgt_test(sdf_generation_test ${PROJECT_NAME})
  vgt_test(voxel_raycasting_test ${PROJECT_NAME}_pointcloud_voxelization)
endif()

install(TARGETS
    ${PROJECT_NAME}
    ${PROJECT_NAME}_cuda_voxelization_helpers
    ${PROJECT_NAME}_opencl_voxelization_helpers
    ${PROJECT_NAME}_pointcloud_voxelization
    ${PROJECT_NAME}_pointcloud_voxelization_ros_interface
    ${PROJECT_NAME}_ros_interface
  EXPORT ${PROJECT_NAME}Targets
  FILE_SET HEADERS)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
# Static device helpers pass their CUDA and OpenCL links on to consumers.
set(device_dependencies)
if(NOT BUILD_SHARED_LIBS)
  if(CMAKE_CUDA_COMPILER)
    list(APPEND device_dependencies CUDAToolkit)
  endif()
  if(OpenCL_FOUND)
    list(APPEND device_dependencies OpenCL)
  endif()
endif()
ament_export_dependencies(
  common_robotics_utilities
  Eigen3
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
  ${device_dependencies})
ament_package()
