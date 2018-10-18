cmake_minimum_required(VERSION 2.8.3)
project(voxelized_geometry_tools)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp
                                        std_msgs
                                        geometry_msgs
                                        visualization_msgs
                                        message_generation
                                        common_robotics_utilities)
find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
find_package(OpenMP)

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg
                  FILES
                  CollisionMapMessage.msg
                  DynamicSpatialHashedCollisionMapMessage.msg
                  SignedDistanceFieldMessage.msg
                  TaggedObjectCollisionMapMessage.msg)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs std_msgs)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project
catkin_package(INCLUDE_DIRS
               include
               LIBRARIES
               ${PROJECT_NAME}
               CATKIN_DEPENDS
               roscpp
               std_msgs
               geometry_msgs
               visualization_msgs
               message_runtime
               common_robotics_utilities
               DEPENDS
               Eigen3)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include SYSTEM ${catkin_INCLUDE_DIRS}
                                   ${Eigen3_INCLUDE_DIRS})

## Build options
add_compile_options(-std=c++11)
add_compile_options(-Wall)
add_compile_options(-Wextra)
add_compile_options(-Werror)
add_compile_options(-Wconversion)
add_compile_options(-Wshadow)
add_compile_options(-O3)
add_compile_options(-g)
add_compile_options(-flto)

## It's not clear if add_compile_options does the right things for flags that
## may differ between languages and target type.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")

# Voxelized geometry tools library
add_library(${PROJECT_NAME}
            include/${PROJECT_NAME}/collision_map.hpp
            include/${PROJECT_NAME}/dynamic_spatial_hashed_collision_map.hpp
            include/${PROJECT_NAME}/signed_distance_field.hpp
            include/${PROJECT_NAME}/signed_distance_field_generation.hpp
            include/${PROJECT_NAME}/tagged_object_collision_map.hpp
            include/${PROJECT_NAME}/topology_computation.hpp
            src/${PROJECT_NAME}/collision_map.cpp
            src/${PROJECT_NAME}/dynamic_spatial_hashed_collision_map.cpp
            src/${PROJECT_NAME}/tagged_object_collision_map.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS}
                                 ${PROJECT_NAME}_gencpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

# ROS interface library
add_library(${PROJECT_NAME}_ros_interface
            include/${PROJECT_NAME}/ros_interface.hpp
            src/${PROJECT_NAME}/ros_interface.cpp)
add_dependencies(${PROJECT_NAME}_ros_interface ${catkin_EXPORTED_TARGETS}
                                               ${PROJECT_NAME}_gencpp)
target_link_libraries(${PROJECT_NAME}_ros_interface ${PROJECT_NAME}
                                                    ${catkin_LIBRARIES})

# Examples
add_executable(voxelized_geometry_tools_tutorial example/tutorial.cpp)
add_dependencies(voxelized_geometry_tools_tutorial
    ${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(voxelized_geometry_tools_tutorial
    ${PROJECT_NAME}_ros_interface ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(estimate_distance_example example/estimate_distance.cpp)
add_dependencies(estimate_distance_example
    ${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(estimate_distance_example
    ${PROJECT_NAME}_ros_interface ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(spatial_segments example/spatial_segments.cpp)
add_dependencies(spatial_segments
    ${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(spatial_segments
    ${PROJECT_NAME}_ros_interface ${PROJECT_NAME} ${catkin_LIBRARIES})

#############
## Install ##
#############

## Mark library for installation
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_ros_interface
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
