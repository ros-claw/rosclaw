# Dell 7960 Deployment Guide — ROSClaw v1.0

> **Fast-track guide** for Dell Precision 7960 Tower / Rack workstations.
>
> For deeper coverage (BIOS tuning, iDRAC, systemd unit files, performance
> baselines, troubleshooting), see [`DEPLOY_DELL_7960.md`](DEPLOY_DELL_7960.md).

---

## TL;DR — 6 Commands

```bash
# 1. System prep (Ubuntu 22.04 / 24.04)
sudo apt update && sudo apt install -y \
    python3.10 python3.10-venv git \
    libgl1-mesa-glx libosmesa6 libglfw3 \
    linux-lowlatency

# 2. Reboot into low-latency kernel
sudo reboot

# 3. Virtual env
python3.10 -m venv ~/.venvs/rosclaw && source ~/.venvs/rosclaw/bin/activate

# 4. Install
pip install --upgrade pip wheel && pip install rosclaw==1.0.0

# 5. Health check
rosclaw --version          # rosclaw 1.0.0
rosclaw doctor             # all green

# 6. Run
rosclaw init ~/rosclaw-ws && cd ~/rosclaw-ws && rosclaw run --safety-level STRICT
```

---

## Recommended Hardware (Dell 7960 Tower / Rack)

| Component | Minimum | Recommended | Reason |
|-----------|---------|-------------|--------|
| CPU | Xeon w5-3425 (12C/24T) | Xeon w7-3465X (28C/56T) | MuJoCo + 1 kHz capture love cores |
| RAM | 32 GB DDR5 ECC | 128 GB DDR5 ECC | SeekDB + multi-robot scenes |
| GPU | iGPU | NVIDIA RTX 6000 Ada (48 GB) | Local LLM / vision |
| Storage | 1 TB NVMe | 2 × 4 TB NVMe (RAID-1) | ~50 GB/day/robot at 1 kHz |
| Network | 1 GbE | 2 × 10 GbE | Multi-robot DDS |

---

## Dell 7960-Specific BIOS Settings

Set these before installing ROSClaw:

| BIOS Setting | Value |
|--------------|-------|
| System Profile | **Performance** (not "Performance Per Watt") |
| C-States | Disabled |
| Hyper-Threading | Enabled |
| Intel SpeedStep | Enabled |
| Intel Turbo Boost | Enabled |
| Intel Virtualization (VT-x) | Enabled (if running Docker) |
| Memory Operating Mode | Optimizer Mode |

> Without `Performance` profile, the 1 kHz `practice` recorder will exhibit
> > 1 ms jitter under load.

---

## Optional: ROS 2 Bridge

If you also run ROS 2 on the same workstation:

```bash
# Source your ROS 2 install
source /opt/ros/humble/setup.bash      # or /opt/ros/jazzy/

# Install with the ros2 extra
pip install 'rosclaw[ros2]==1.0.0'

# Switch to Cyclone DDS for lower latency
sudo apt install ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Start ROSClaw with ROS 2 bridge
rosclaw run --enable-ros2-bridge --safety-level STRICT
```

Pin DDS to your wired interface (e.g. `eno1`) to avoid Wi-Fi drift — see the
long-form guide for the `~/.cyclonedds.xml` snippet.

---

## Optional: systemd Service

For unattended operation:

```bash
sudo tee /etc/systemd/system/rosclaw.service > /dev/null << 'EOF'
[Unit]
Description=ROSClaw Runtime
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=dell
Environment=PATH=/home/dell/.venvs/rosclaw/bin:/usr/bin
WorkingDirectory=/home/dell/rosclaw-ws
ExecStart=/home/dell/.venvs/rosclaw/bin/rosclaw run --safety-level STRICT
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now rosclaw
sudo systemctl status rosclaw
```

---

## Performance You Should See (Dell 7960 Tower, w7-3465X)

| Metric | Expected (Mock) | Expected (MuJoCo full) |
|--------|-----------------|-----------------------|
| Firewall validation p50 | 0.4 ms | 2.1 ms |
| Firewall validation p99 | 1.1 ms | 6.8 ms |
| EventBus throughput | 110 k msg/s | 95 k msg/s |
| 1 kHz capture jitter p99 | < 50 µs | < 80 µs |
| SeekDB top-3 vector query (100 k vectors) | 8 ms | 8 ms |
| Know-How end-to-end | 15 ms | 18 ms |

If your numbers are significantly worse:

1. Confirm BIOS `Performance` profile is active.
2. Confirm you booted into the `-lowlatency` kernel: `uname -r`.
3. Confirm GPU isn't thermally throttling: `nvidia-smi -q -d TEMPERATURE`.

---

## Troubleshooting Quickref

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `MuJoCo: failed to load OpenGL` | SSH without X11 | `export MUJOCO_GL=osmesa` |
| 1 kHz jitter > 1 ms | Default scheduler | Lowlatency kernel + `chrt` |
| `rosclaw run` → port 8088 busy | Old runtime still up | `lsof -i :8088` → kill PID |
| GPU temp > 85 °C | Tower fan direction wrong | Flip top fan to exhaust |
| iDRAC blocks systemd start | iDRAC10 state stuck | `sudo systemctl restart racadm` |
| SeekDB first start hangs 30 s | Embedding model download | Ensure outbound HTTPS works |

---

## What's Next

- **Day 1**: Run `examples/hello_robot.py` to verify the full pipeline.
- **Day 2**: Read [`MODULES.md`](MODULES.md) to understand the 16 sub-modules.
- **Day 3**: Plug in your robot via `mcp_drivers` and try a real skill.
- **Production**: Promote via the systemd unit above and add Grafana
  monitoring (see [`MONITORING.md`](MONITORING.md) — ships in v1.1).

---

*Quickstart for impatient operators. Comprehensive coverage in
[`DEPLOY_DELL_7960.md`](DEPLOY_DELL_7960.md).*
