# ROS 1 Noetic + rosbridge + turtlesim test environment
#
# Usage:
#   docker compose -f docker-compose.ros1-test.yml up --build -d
#
# This image deliberately installs rosbridge_server and turtlesim only;
# no ROSClaw code is needed inside the container.

FROM ros:noetic-ros-base

ENV ROS_DISTRO=noetic
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_QPA_PLATFORM=offscreen
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rosbridge-suite \
    ros-noetic-turtlesim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros_ws

COPY docker/ros1-noetic-launch.launch /ros_ws/ros1-noetic-launch.launch

EXPOSE 9090

CMD ["roslaunch", "/ros_ws/ros1-noetic-launch.launch"]
