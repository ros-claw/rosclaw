#!/usr/bin/env bash
# Phase 1 — Install RealSense driver + ROS2 Jazzy RealSense wrapper
# Run with: sudo bash scripts/phase1_install_realsense.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[phase1] Updating apt (ignoring unreachable mirrors) ..."
apt-get update || true

echo "[phase1] Installing librealsense2 (driver + utils, no dkms on arm64) ..."
apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-udev-rules

echo "[phase1] Installing ROS2 Jazzy RealSense camera wrapper ..."
apt-get install -y \
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-camera-msgs \
    ros-jazzy-realsense2-description

echo "[phase1] Sourcing ROS2 Jazzy environment ..."
source /opt/ros/jazzy/setup.bash

echo "[phase1] Enumerating RealSense devices ..."
rs-enumerate-devices -s || true

echo "[phase1] Done."
