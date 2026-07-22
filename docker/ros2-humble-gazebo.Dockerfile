# ROS 2 Humble + Gazebo Fortress + ros_gz bridge verification environment.
#
# Humble's officially recommended Gazebo pairing is Fortress.  The ros_gz
# metapackage provides the simulator integration and bridge packages without
# requiring ROS to be installed on the host.

ARG BASE_IMAGE=osrf/ros:humble-desktop
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-gz \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
CMD ["bash"]
