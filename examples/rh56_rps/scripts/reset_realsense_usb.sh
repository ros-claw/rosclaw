#!/usr/bin/env bash
# Software reset of the Intel RealSense camera without physically reseating it.
#
# This script uses the standard usbreset tool (usbutils) to send a USB reset to
# the RealSense device.  It is useful when the camera node reports
# "Frames didn't arrive within 5 seconds" or the device disappears from lsusb.
#
# Usage:
#   ./scripts/reset_realsense_usb.sh
#
# usbreset usually requires root privileges.  If the device is not currently
# enumerated, you can reset the whole USB bus it was on (e.g. usb4) with:
#   sudo bash -c 'echo 0 > /sys/bus/usb/devices/usb4/authorized; sleep 2; echo 1 > /sys/bus/usb/devices/usb4/authorized'

set -e

DEVICE=$(lsusb | awk '/Intel Corp.*RealSense/ {print $2"/"$4}' | sed 's/://' | head -1)

if [ -z "${DEVICE}" ]; then
    echo "No Intel RealSense device found in lsusb."
    echo "The camera is not enumerated by the kernel.  Options:"
    echo "  1. Reset the USB bus it is attached to (e.g. usb4):"
    echo "       sudo bash -c 'echo 0 > /sys/bus/usb/devices/usb4/authorized; sleep 2; echo 1 > /sys/bus/usb/devices/usb4/authorized'"
    echo "  2. Physically unplug and replug the camera USB cable."
    exit 1
fi

NODE="/dev/bus/usb/${DEVICE}"
if [ ! -c "${NODE}" ]; then
    echo "Expected USB device node does not exist: ${NODE}"
    exit 1
fi

echo "Resetting RealSense USB device: ${NODE}"
if command -v usbreset >/dev/null 2>&1; then
    sudo usbreset "${NODE}"
else
    echo "usbreset not found.  Install usbutils or use the sysfs bus reset command above."
    exit 1
fi

echo "Reset complete.  Wait a few seconds, then check with: rs-enumerate-devices"
