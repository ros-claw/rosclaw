#!/bin/bash
set -e

# Source ROS 2 environment.
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# Run Qt applications (e.g. turtlesim) without a real X server.
export QT_QPA_PLATFORM=offscreen

# If the first argument is 'turtlesim', launch turtlesim + rosbridge.
if [ "$1" = "turtlesim" ]; then
    shift
    # Launch a single turtlesim node on the root namespace so tests see
    # /turtle1/cmd_vel and /turtle1/pose directly.
    xvfb-run -a ros2 run turtlesim turtlesim_node &
    sleep 2
    exec ros2 launch rosbridge_server rosbridge_websocket_launch.xml "$@"
fi

exec "$@"
