find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${Eigen3_INCLUDE_DIRS})
