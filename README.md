# voxelized_geometry_tools
Voxelized geometry tools (Voxel-based collision/occupancy maps, signed-distance fields, discrete geometry tools)

## Setup

`voxelized_geometry_tools` is a ROS package.

Thus, it is best to build it within a ROS workspace:

```sh
mkdir -p ~/ws/src
cd ~/ws/src
git clone https://github.com/calderpg/voxelized_geometry_tools.git
```

This package supports [ROS 1 Kinetic+](http://wiki.ros.org/ROS/Installation)
and [ROS 2 Dashing+](https://index.ros.org/doc/ros2/Installation/) distributions.
Make sure to symlink the corresponding `CMakeLists.txt` and `package.xml` files
for the ROS distribution of choice:

*For ROS 1 Kinetic+*
```sh
cd ~/ws/src/voxelized_geometry_tools
ln -sT CMakeLists.txt.ros1 CMakeLists.txt
ln -sT package.xml.ros1 package.xml
```

*For ROS 2 Dashing+*
```sh
cd ~/ws/src/voxelized_geometry_tools
ln -sT CMakeLists.txt.ros2 CMakeLists.txt
ln -sT package.xml.ros2 package.xml
```

Finally, use [`rosdep`](https://docs.ros.org/independent/api/rosdep/html/)
to ensure all dependencies in the `package.xml` are satisfied:

```sh
cd ~/ws
rosdep install -i -y --from-path src
```

## Building

Use [`catkin_make`](http://wiki.ros.org/catkin/commands/catkin_make) or
[`colcon`](https://colcon.readthedocs.io/en/released/) accordingly.

*For ROS 1 Kinetic+*
```sh
cd ~/ws
catkin_make  # the entire workspace
catkin_make --pkg voxelized_geometry_tools  # the package only
```

*For ROS 2 Dashing +*
```sh
cd ~/ws
colcon build  # the entire workspace
colcon build --packages-select voxelized_geometry_tools  # the package only
```

## Running examples

Since examples are not installed, these end up build tree.
In ROS 1, `catkin_make` devel spaces aid execution. 
In ROS 2, there is no such aid.

*For ROS 1 Kinetic+*
```sh
cd ~/ws
source ./devel/setup.bash
rosrun voxelized_geometry_tools <example>
```

*For ROS 2 Dashing +*
```sh
cd ~/ws
source ./install/setup.bash
./build/voxelized_geometry_tools/<example>
```
