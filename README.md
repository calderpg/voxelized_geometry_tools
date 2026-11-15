# voxelized_geometry_tools
Voxelized geometry tools (Voxel-based collision/occupancy maps, signed-distance fields, discrete geometry tools).

## Setup

For convenience of building, `voxelized_geometry_tools` is a ROS package which
supports [ROS 2 Dashing](https://index.ros.org/doc/ros2/Installation/) and
newer ROS 2 distributions. Much of this project can be built entirely without
ROS support, see the example of the
[Drake](https://github.com/RobotLocomotion/drake) project which builds this
subset using Bazel for both Linux and macOS platforms.

As provided, `voxelized_geometry_tools` should be built in a ROS 2 workspace:

```sh
mkdir -p ~/ws/src
cd ~/ws/src
git clone https://github.com/calderpg/voxelized_geometry_tools.git
```

Finally, use [`rosdep`](https://docs.ros.org/independent/api/rosdep/html/)
to ensure all dependencies in the `package.xml` are satisfied:

```sh
cd ~/ws
rosdep install -i -y --from-path src
```

## Building

Use [`colcon`](https://colcon.readthedocs.io/en/released/):

*For ROS 2 Dashing+*
```sh
cd ~/ws
colcon build  # the entire workspace
colcon build --packages-select voxelized_geometry_tools  # the package only
```

## Testing

Use [`colcon`](https://colcon.readthedocs.io/en/released/):

*For ROS 2 Dashing+*
```sh
cd ~/ws
colcon test --event-handlers=console_direct+  # the entire workspace
colcon test --event-handlers=console_direct+ --packages-select voxelized_geometry_tools  # the package only
```

## Running examples

Use [`ros2 run`](https://index.ros.org/doc/ros2/Tutorials/Understanding-ROS2-Nodes/#ros2-run):

*For ROS 2 Dashing+*
```sh
cd ~/ws
source ./install/setup.bash
ros2 run voxelized_geometry_tools <example>
```
