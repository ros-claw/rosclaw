# ROS 2 Humble + rosbridge + turtlesim test environment
#
# Usage:
#   docker compose -f docker-compose.ros-test.yml up --build
#
# This image deliberately installs rosbridge_server and turtlesim only;
# no ROSClaw code is needed inside the container.

FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
# Run Qt applications (e.g. turtlesim) without a real X server.
ENV QT_QPA_PLATFORM=offscreen

# Install rosbridge_server and turtlesim packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-rosbridge-server \
        ros-${ROS_DISTRO}-turtlesim \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# Source ROS setup automatically for interactive shells and entrypoint.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

COPY docker/ros-entrypoint.sh /ros-entrypoint.sh
RUN chmod +x /ros-entrypoint.sh

ENTRYPOINT ["/ros-entrypoint.sh"]
CMD ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"]
