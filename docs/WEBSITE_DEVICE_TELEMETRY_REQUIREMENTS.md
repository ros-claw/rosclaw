# ROSClaw 设备/环境遥测 — 网站侧开发需求文档

> 目标读者：`rosclaw-website` 开发团队  
> 配套文档：> - `docs/WEBSITE_VERCEL_FEEDBACK_TELEMETRY_HANDOFF.md`（整体实施手册）> - `rosclaw_feedback_v1.md`（产品/技术规范）> 版本：v1.1 / 2026-06-26

---

## 1. 背景

CLI 侧（`ros-claw/rosclaw`）已经在产品遥测事件的 `payload` 中自动附带了设备/环境信息，包括 ROS 版本、机器人型号、GPU/CUDA、操作系统版本、传感器类型等。

网站侧需要把这些数据**接进来、存下来、展示出来、聚合出来**，让运营/产品团队能在后台看到：

- 用户在用什么机器人型号
- ROS 版本分布
- CUDA/GPU 分布
- 操作系统版本分布
- 传感器 adoption

---

## 2. 数据在哪里？

所有设备信息都已经在 CLI 上报到 `/api/telemetry/event` 的 `payload` 字段里，例如：

```json
{
  "schema_version": "rosclaw.telemetry.event.v1",
  "event_type": "command_completed",
  "anonymous_installation_id": "rci_...",
  "command_name": "doctor",
  "command_status": "success",
  "payload": {
    "duration_ms": 250,
    "os_version": "6.8.0-101-generic",
    "ros_distro_present": "humble",
    "ros_distros": ["humble"],
    "cuda_available": true,
    "gpu_info": "NVIDIA GeForce RTX 3080",
    "robot_type": "sim_ur5e",
    "sensor_types": ["camera", "imu"]
  }
}
```

**注意**：
- 如果某项探测不到，CLI 会直接省略该字段，不会传 `null`。
- 这些字段**不包含** hostname、username、ip、robot_serial 等敏感信息。
- 设备信息会出现在几乎所有 telemetry 事件中（`command_completed`、`heartbeat`、`firstboot_started`、`firstboot_completed`、`doctor_started`、`doctor_completed` 等）。

---

## 3. 数据库存储：不需要改 Schema

继续使用现有的 Supabase 表：

```sql
public.telemetry_events (
  id uuid,
  anonymous_installation_id text,
  event_type text,
  command_name text,
  command_status text,
  rosclaw_version text,
  os_family text,
  arch text,
  python_major_minor text,
  created_at timestamptz,
  received_at timestamptz default now(),
  payload jsonb
)
```

设备信息整体存在 `payload`（`jsonb`）里即可。

### 可选优化（P2 不做也行）

如果后续数据量大、查询慢，可以加 generated columns + 索引：

```sql
alter table public.telemetry_events
  add column robot_type text generated always as (payload->>'robot_type') stored;

alter table public.telemetry_events
  add column ros_distro text generated always as (payload->>'ros_distro_present') stored;

create index idx_telemetry_events_robot_type on public.telemetry_events(robot_type);
create index idx_telemetry_events_ros_distro on public.telemetry_events(ros_distro);
```

P2 阶段建议先直接用 `payload` 查询，不够快再上 generated columns。

---

## 4. API 接收端：需要确认/放开 payload 校验

### 4.1 检查点

如果你之前对 `/api/telemetry/event` 的 `payload` 做了**严格字段白名单校验**，请改成：

- 只校验**顶层字段**的 schema_version / event_type / anonymous_installation_id / command_name / command_status / duration_bucket / error_class_bucket。
- `payload` 整体作为 `jsonb` 透传，不再校验内部字段名。
- 仍然递归检查 `payload` 内部是否包含 forbidden fields（hostname、username、ip、robot_serial 等）。

### 4.2 为什么不能对 payload 内部做严格白名单？

因为 CLI 会不断补充新的设备/环境字段，payload 是产品设计上的"扩展区"。服务端只需要保证没有敏感字段泄露即可。

---

## 5. Admin Dashboard 需要新增的内容

在 `/admin/telemetry` 页面增加一个 **"Environment & Devices"** 区块。

### 5.1 必须展示的卡片/图表

| 图表 | 指标 | SQL / 计算方式 |
|---|---|---|
| **Robot Type Distribution** | 各机器人型号占比 | `payload->>'robot_type'` |
| **ROS Distro Distribution** | ROS 版本分布 | `payload->>'ros_distro_present'` |
| **CUDA Availability** | CUDA 可用/不可用占比 | `(payload->>'cuda_available')::boolean` |
| **GPU Top List** | 常见 GPU 型号 Top N | `payload->>'gpu_info'` |
| **OS Version Distribution** | 操作系统版本分布 | `payload->>'os_version'` |
| **Sensor Adoption** | 传感器类型使用次数 | `jsonb_array_elements_text(payload->'sensor_types')` |

### 5.2 推荐的数据源事件

为了指标准确，建议按事件类型过滤：

- **Robot / ROS / CUDA / OS**：用 `event_type = 'firstboot_completed'`（每个安装一次）或 `heartbeat`（每个活跃安装一次）。
- **Command-level 分布**：用 `event_type = 'command_completed'`。
- **Sensor adoption**：用 `firstboot_completed` 或 `command_completed`，因为 sensor_types 来自 `body.yaml`，只有用户配置 body 后才有。

### 5.3 SQL 示例

```sql
-- 机器人型号分布（按安装去重）
select
  payload->>'robot_type' as robot_type,
  count(distinct anonymous_installation_id) as installs
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'robot_type' is not null
group by payload->>'robot_type'
order by installs desc;

-- ROS 版本分布
select
  payload->>'ros_distro_present' as ros_distro,
  count(distinct anonymous_installation_id) as installs
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'ros_distro_present' is not null
group by payload->>'ros_distro_present'
order by installs desc;

-- CUDA 可用率
select
  (payload->>'cuda_available')::boolean as cuda_available,
  count(distinct anonymous_installation_id) as installs
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'cuda_available' is not null
group by (payload->>'cuda_available')::boolean;

-- GPU 型号 Top 10
select
  payload->>'gpu_info' as gpu_info,
  count(distinct anonymous_installation_id) as installs
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'gpu_info' is not null
group by payload->>'gpu_info'
order by installs desc
limit 10;

-- 操作系统版本分布
select
  payload->>'os_version' as os_version,
  count(distinct anonymous_installation_id) as installs
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'os_version' is not null
group by payload->>'os_version'
order by installs desc;

-- 传感器类型 adoption（数组展开）
select
  sensor_type,
  count(*) as appearances
from public.telemetry_events,
     lateral jsonb_array_elements_text(payload->'sensor_types') as sensor_type
where payload->'sensor_types' is not null
group by sensor_type
order by appearances desc;
```

### 5.4 页面布局建议

```text
/admin/telemetry

[Overview Cards]
  Total Installs | DAU | WAU | MAU | Firstboot Completion | Doctor Success Rate

[Charts]
  Installs over time
  Active installs over time
  Version distribution
  OS distribution

[NEW: Environment & Devices]
  Robot Type Distribution (Pie/Bar)
  ROS Distro Distribution (Bar)
  CUDA Availability (Pie)
  GPU Top List (Table)
  OS Version Distribution (Bar)
  Sensor Adoption (Tag cloud / Table)

[Tables]
  Top Commands
  Command Failure Rate
  Module Adoption
  Recent Events
```

---

## 6. 聚合 Cron 任务需要补的指标

在 `/api/admin/telemetry/aggregate` 中，除了原有指标，新增以下 `telemetry_daily_aggregates` 写入逻辑：

### 6.1 新增 metric_name

```text
robot_type_distribution        # dimension: { robot_type: "sim_ur5e" }
ros_distro_distribution        # dimension: { ros_distro: "humble" }
cuda_available_count           # dimension: { cuda_available: true }
cuda_unavailable_count         # dimension: { cuda_available: false }
os_version_distribution        # dimension: { os_version: "6.8.0-101-generic" }
gpu_info_distribution          # dimension: { gpu_info: "NVIDIA GeForce RTX 3080" }
sensor_type_adoption           # dimension: { sensor_type: "camera" }
```

### 6.2 聚合 SQL 示例

```sql
-- robot_type_distribution
insert into public.telemetry_daily_aggregates (day, metric_name, dimension, value)
select
  date(received_at) as day,
  'robot_type_distribution' as metric_name,
  jsonb_build_object('robot_type', payload->>'robot_type') as dimension,
  count(distinct anonymous_installation_id) as value
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'robot_type' is not null
  and received_at >= current_date - interval '1 day'
group by day, payload->>'robot_type'
on conflict (day, metric_name, dimension) do update set
  value = excluded.value,
  updated_at = now();

-- ros_distro_distribution
insert into public.telemetry_daily_aggregates (day, metric_name, dimension, value)
select
  date(received_at) as day,
  'ros_distro_distribution' as metric_name,
  jsonb_build_object('ros_distro', payload->>'ros_distro_present') as dimension,
  count(distinct anonymous_installation_id) as value
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'ros_distro_present' is not null
  and received_at >= current_date - interval '1 day'
group by day, payload->>'ros_distro_present'
on conflict (day, metric_name, dimension) do update set
  value = excluded.value,
  updated_at = now();

-- cuda_available_count / cuda_unavailable_count
insert into public.telemetry_daily_aggregates (day, metric_name, dimension, value)
select
  date(received_at) as day,
  'cuda_available_count' as metric_name,
  jsonb_build_object('cuda_available', (payload->>'cuda_available')::boolean) as dimension,
  count(distinct anonymous_installation_id) as value
from public.telemetry_events
where event_type = 'firstboot_completed'
  and payload->>'cuda_available' is not null
  and received_at >= current_date - interval '1 day'
group by day, (payload->>'cuda_available')::boolean
on conflict (day, metric_name, dimension) do update set
  value = excluded.value,
  updated_at = now();

-- sensor_type_adoption（数组展开后聚合）
insert into public.telemetry_daily_aggregates (day, metric_name, dimension, value)
select
  date(received_at) as day,
  'sensor_type_adoption' as metric_name,
  jsonb_build_object('sensor_type', sensor_type) as dimension,
  count(*) as value
from public.telemetry_events,
     lateral jsonb_array_elements_text(payload->'sensor_types') as sensor_type
where received_at >= current_date - interval '1 day'
group by day, sensor_type
on conflict (day, metric_name, dimension) do update set
  value = excluded.value,
  updated_at = now();
```

### 6.3 聚合 Cron 配置

`vercel.json` 保持每小时跑一次：

```json
{
  "crons": [
    {
      "path": "/api/admin/telemetry/aggregate",
      "schedule": "0 * * * *"
    }
  ]
}
```

---

## 7. Admin API 可能需要新增的查询参数

为了让 dashboard 可以下钻，建议给 `/api/admin/telemetry/events` 增加可选过滤参数：

```text
GET /api/admin/telemetry/events?robot_type=sim_ur5e&ros_distro=humble&cuda_available=true&limit=50
```

实现时用 SQL `payload` 过滤：

```sql
select *
from public.telemetry_events
where payload->>'robot_type' = $1
  and payload->>'ros_distro_present' = $2
  and (payload->>'cuda_available')::boolean = $3
order by received_at desc
limit $4;
```

---

## 8. 环境变量

不需要新增环境变量。沿用现有的：

```bash
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
ROSCLAW_ADMIN_EMAILS=
```

---

## 9. 本地联调验证步骤

### 9.1 启动本地环境

```bash
cd rosclaw-website
npm install
supabase start
supabase db reset
npm run dev
```

### 9.2 让 CLI 指向本地

```bash
export ROSCLAW_HOME=$(mktemp -d)
export ROSCLAW_TELEMETRY_ENDPOINT=http://localhost:3000/api/telemetry/event

# 假装有 ROS，这样 ros_distro_present 才有值
export ROS_DISTRO=humble

rosclaw firstboot --yes --profile offline --robot sim_ur5e
```

### 9.3 验证 Supabase 数据

在 Supabase Studio 执行：

```sql
select event_type, payload
from public.telemetry_events
where anonymous_installation_id like 'rci_%'
limit 10;
```

应该能看到 `payload` 里有：

```json
{
  "os_version": "...",
  "ros_distro_present": "humble",
  "cuda_available": false,
  "robot_type": "sim_ur5e"
}
```

### 9.4 验证 Dashboard

打开 `http://localhost:3000/admin/telemetry`，确认：

- 能看到 Robot Type Distribution 里有 `sim_ur5e`
- ROS Distro Distribution 里有 `humble`
- CUDA Availability 饼图有数据

---

## 10. Definition of Done

- [ ] `/api/telemetry/event` 不对 `payload` 内部字段做严格白名单，只拒绝 forbidden fields。
- [ ] `/admin/telemetry` 页面新增 Environment & Devices 区块。
- [ ] 至少展示：Robot Type、ROS Distro、CUDA Availability、GPU Top、OS Version、Sensor Adoption。
- [ ] `/api/admin/telemetry/aggregate` 新增 robot/ros_distro/cuda/os/gpu/sensor 的日聚合。
- [ ] `/api/admin/telemetry/events` 支持按 `robot_type`、`ros_distro`、`cuda_available` 过滤（可选但推荐）。
- [ ] 本地用 CLI 跑一遍 firstboot，能在 dashboard 看到设备分布。
- [ ] Vercel build pass，API tests pass。

---

## 11. 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| payload 里看不到 `robot_type` | `rosclaw.yaml` 还没有 `runtime.robot_id` | 运行 `rosclaw firstboot` 生成配置 |
| payload 里看不到 `sensor_types` | `body.yaml` 未创建 | 这是正常的，只有用户配置 body 后才有 |
| `ros_distro_present` 为空 | 环境没有 `ROS_DISTRO` 且 `/opt/ros` 不存在 | 本地测试时 `export ROS_DISTRO=humble` |
| dashboard 加载慢 | 每次都 count 全表 | 接入 aggregate 表或加 generated columns |

---

**输出文件**：`docs/WEBSITE_DEVICE_TELEMETRY_REQUIREMENTS.md`
