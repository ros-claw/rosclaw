ARG BASE_IMAGE=ros:jazzy-ros-base-noble
FROM ${BASE_IMAGE}
ARG APT_PROXY=
ENV ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1
# Only this disposable container is configured; the workstation is untouched.
# A proxy may forward downloads on restricted networks. APT retains signature
# and package hash verification using the image's original trusted keys.
RUN if [ -n "$APT_PROXY" ]; then \
      sed -i "s|http://ports.ubuntu.com/ubuntu-ports|${APT_PROXY}/ubuntu-ports|g" /etc/apt/sources.list.d/ubuntu.sources && \
      sed -i "s|http://packages.ros.org/ros2/ubuntu|${APT_PROXY}/ros2|g" /etc/apt/sources.list.d/ros2.sources; \
    fi && \
    sed -i 's/Types: deb deb-src/Types: deb/' /etc/apt/sources.list.d/ros2.sources && \
    apt-get -o Acquire::Retries=3 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-minimal-tb3-sim \
      ros-jazzy-fields2cover ros-jazzy-rosbridge-server ros-jazzy-rosbridge-library \
      ros-jazzy-tf-transformations ros-jazzy-diagnostic-msgs libgdal-dev \
      python3-colcon-common-extensions python3-venv python3-pip && \
    dpkg-query -W > /opt/ros-expert-package-lock.txt
COPY opennav_coverage /ws/src/opennav_coverage
RUN . /opt/ros/jazzy/setup.sh && cd /ws && \
    colcon build --symlink-install --packages-up-to opennav_coverage_navigator \
      --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
WORKDIR /ws
