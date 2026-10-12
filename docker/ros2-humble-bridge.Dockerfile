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

# Keep the signed official repositories. Partial index refresh must fail here,
# rather than becoming a misleading "package not found" during installation.
# Transport retries are bounded; signature/package errors remain fatal.
# Preserve signed indexes and installed versions for deployment provenance.
# Install rosbridge_server and turtlesim packages.
RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 -o APT::Update::Error-Mode=any update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 install -y --no-install-recommends \
        ros-${ROS_DISTRO}-rosbridge-server \
        ros-${ROS_DISTRO}-turtlesim \
        xvfb \
    && mkdir -p /usr/share/rosclaw-build-evidence \
    && cp /var/lib/apt/lists/*InRelease /usr/share/rosclaw-build-evidence/ \
    && sha256sum /var/lib/apt/lists/*Release /var/lib/apt/lists/*Packages* \
        > /usr/share/rosclaw-build-evidence/apt-index-sha256.txt \
    && dpkg-query -W > /usr/share/rosclaw-build-evidence/packages.tsv \
    && cp -a /etc/apt/sources.list /etc/apt/sources.list.d \
        /usr/share/rosclaw-build-evidence/ \
    && rm -rf /var/lib/apt/lists/*

# Source ROS setup automatically for interactive shells and entrypoint.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

COPY docker/ros-entrypoint.sh /ros-entrypoint.sh
RUN chmod +x /ros-entrypoint.sh

ENTRYPOINT ["/ros-entrypoint.sh"]
CMD ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"]
