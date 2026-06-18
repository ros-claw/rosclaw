# ROSClaw Core container — no ROS installation.
#
# Usage:
#   docker compose -f docker-compose.ros-test.yml up --build -d
#
# This image runs the ROSClaw Python codebase and talks to rosbridge over the
# docker network. It deliberately does NOT install rclpy or any ROS packages.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rosclaw

# Install the package in editable mode.
# The docker compose volume mount brings the live source code at runtime.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the ROS connector unit tests (integration tests are opt-in).
CMD ["pytest", "-q", "tests/connectors/ros", "-k", "not integration"]
