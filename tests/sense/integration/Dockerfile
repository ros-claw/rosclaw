FROM ros:jazzy-ros-base
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pytest python3-yaml \
    ros-jazzy-sensor-msgs ros-jazzy-rmw-cyclonedds-cpp \
    && dpkg-query -W -f='${Package}\t${Version}\n' > /ros-sense-fixture-packages.tsv \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml /candidate/pyproject.toml
COPY src /candidate/src
COPY tests/conftest.py /candidate/tests/conftest.py
COPY tests/sense/integration/test_ros2_live_collector.py /candidate/tests/sense/integration/test_ros2_live_collector.py
COPY tests/sense/integration/cyclone-loopback.xml /fixture/cyclone.xml
ENV PYTHONPATH=/candidate/src:/candidate \
    PYTHONDONTWRITEBYTECODE=1 \
    ROS_DOMAIN_ID=32 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=file:///fixture/cyclone.xml \
    HOME=/tmp
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["source /opt/ros/jazzy/setup.bash && exec python3 -m pytest -q -p no:cacheprovider -m integration /candidate/tests/sense/integration/test_ros2_live_collector.py"]
