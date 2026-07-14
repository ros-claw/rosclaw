#!/usr/bin/env bash
# Disable system services/udev rules that commonly grab CH340 USB/RS485 adapters
# and cause "Input/output error" or kernel -110 on reopen.
set -e

if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo."
    exit 1
fi

# 1. Blacklist CH340 from ModemManager probing.
RULE_SRC="/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps/scripts/99-rh56-usb.rules"
RULE_DST="/etc/udev/rules.d/99-rh56-usb.rules"
if [ -f "$RULE_SRC" ]; then
    cp "$RULE_SRC" "$RULE_DST"
    chmod 644 "$RULE_DST"
    echo "Installed $RULE_DST"
fi

# 2. Disable brltty's udev activation so it cannot claim ttyUSB devices.
#    (The brltty package is installed on Ubuntu Desktop by default.)
BRLTTY_RULE="/usr/lib/udev/rules.d/85-brltty.rules"
if [ -f "$BRLTTY_RULE" ] && [ ! -L "/etc/udev/rules.d/85-brltty.rules" ]; then
    ln -s /dev/null /etc/udev/rules.d/85-brltty.rules
    echo "Disabled brltty udev rules for CH340 devices."
fi

# 3. Stop currently running interference for this session.
if systemctl is-active --quiet ModemManager 2>/dev/null; then
    systemctl stop ModemManager
    echo "Stopped ModemManager for this session."
fi
if systemctl is-active --quiet brltty-udev 2>/dev/null; then
    systemctl stop brltty-udev
    echo "Stopped brltty-udev for this session."
fi

# 4. Reload udev and reapply rules to already-connected adapters.
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=1a86 --attr-match=idProduct=7523

echo "Done. Re-plug the CH340 adapters, then retry the demo."
