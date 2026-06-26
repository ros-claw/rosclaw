# ROSClaw Feedback & Telemetry — Website / Vercel 侧实施与联调手册

> 目标读者：`rosclaw-website`（Next.js + Vercel + Supabase）开发团队  
> 配套文档：`rosclaw_feedback_v1.md`（完整产品/技术规范）  
> CLI 侧实现：`ros-claw/rosclaw` PR #42（已合并/已提交）  
> 版本：v1.0 / 2026-06-24

---

## 1. 你们要做什么？一句话版

在 `rosclaw-website` 仓库里实现 **Telemetry 接收端** + **Feedback 接收端** + **Admin Dashboard** + **定时聚合任务**，让 ROSClaw CLI 能把匿名产品遥测事件和手动上传的脱敏反馈包安全地落到 Supabase，并在后台展示核心产品指标。

CLI 侧已经做完：本地 identity、遥测采集、命令 hook、firstboot 集成、脱敏、配置、上传客户端。你们不需要改动 CLI，只需要按照下面的接口契约实现服务端即可。

---

## 2. 数据分层（必须按三层做，不要一刀切）

| 层级 | 名称 | 默认 | 触发方式 | 内容 | 当前优先级 |
|---|---|---|---|---|---|
| Tier 0 | Product Telemetry | **默认开启**，用户可关 | 自动 | install/firstboot/command/heartbeat/模块使用/版本/OS 等 | P2 必须做 |
| Tier 1 | Diagnostic Telemetry | 默认本地采集 | 用户 consent 后上传 | 崩溃摘要、failure stats、sandbox block、provider performance | P2 必须做 |
| Tier 2 | Rich Feedback Bundle | 默认不上传 | 用户手动 `rosclaw feedback upload --redact` | 脱敏 bundle、MCAP 摘要、human feedback | P2 必须做 |

**核心原则**：
- 产品遥测永远**不包含** prompt、日志、本地路径、视频/音频/MCAP/trace、API key、hostname、username、robot serial。
- Feedback 上传**必须**带 `--redact`；服务端也应校验 `redacted=true`，否则拒绝。
- 服务端要假设客户端可能出错/被篡改，所以服务端必须做 schema 校验 + forbidden field 校验 + 限流。

---

## 3. 部署形态建议

```text
┌─────────────────────────────────────┐
│  Vercel / Next.js App Router        │  ← API Routes + Admin Pages + Cron
│  /api/telemetry/*                   │
│  /api/feedback/*                    │
│  /admin/telemetry, /admin/feedback  │
└──────────────┬──────────────────────┘
               │ service role key (server-side only)
               ▼
┌─────────────────────────────────────┐
│  Supabase Postgres                  │  ← telemetry_* / feedback_* 表
│  Supabase Storage                   │  ← feedback-bundles / feedback-attachments
└─────────────────────────────────────┘
```

P2 不需要独立后端服务；未来数据量大了再拆 `rosclaw-cloud-ingest`。

---

## 4. 仓库目录结构建议

在 `rosclaw-website` 中新增/修改：

```text
app/
  api/
    telemetry/
      event/route.ts            # POST 产品遥测事件
      heartbeat/route.ts        # POST heartbeat（可复用 event 逻辑）
    feedback/
      upload/route.ts           # POST multipart 反馈包
      delete-request/route.ts   # POST 删除请求（P2 基础实现）
    admin/
      telemetry/
        summary/route.ts        # GET dashboard 聚合指标
        aggregate/route.ts      # POST/GET 手动触发聚合
        events/route.ts         # GET 最近事件列表
        installations/route.ts  # GET 安装列表
      feedback/
        summary/route.ts        # GET feedback 概览
        events/route.ts         # GET feedback 事件
        batches/route.ts        # GET 上传的 bundle 列表
  admin/
    telemetry/page.tsx          # 产品指标后台
    feedback/page.tsx           # 反馈后台
  privacy/
    telemetry/page.tsx          # 隐私/遥测说明页

lib/
  telemetry/
    schema.ts                   # Zod 校验遥测 schema
    validators.ts               # forbidden field / allowlist 校验
    ingest.ts                   # 写入 Supabase 的辅助函数
    aggregate.ts                # 聚合指标 SQL
    rate-limit.ts               # 简单限流（可用 Vercel KV / Upstash / 内存）
  feedback/
    schema.ts                   # Zod 校验 feedback schema
    storage.ts                  # Supabase Storage 上传/下载
    parser.ts                   # 解析 tar.gz bundle
  supabase/
    admin.ts                    # service role client
    server.ts                   # anon client
  admin/
    auth.ts                     # 管理员权限校验

supabase/migrations/
  004_telemetry_feedback_schema.sql   # 从 rosclaw 主仓库复制

vercel.json                         # Cron 配置
```

---

## 5. Supabase 数据库 Schema（从 rosclaw 主仓库复制）

完整迁移文件已存在于：

```text
rosclaw/supabase/migrations/004_telemetry_feedback_schema.sql
```

请把它**原样复制**到 `rosclaw-website/supabase/migrations/004_telemetry_feedback_schema.sql`，并用 `supabase db reset` / `supabase migration up` 应用。

### 5.1 表清单

| 表 | 用途 |
|---|---|
| `public.telemetry_installations` | 匿名安装去重、首次/最近活跃、版本/OS |
| `public.telemetry_events` | 原始事件（command、heartbeat、install、firstboot 等） |
| `public.telemetry_daily_aggregates` | 每日聚合指标（Cron 产出） |
| `public.feedback_batches` | 用户手动上传的反馈包元数据 |
| `public.feedback_events` | 反馈包里的结构化事件 |
| `public.feedback_attachments` | 反馈包附带文件（媒体、MCAP 摘要等） |
| `public.feedback_delete_requests` | 用户删除请求 |

### 5.2 RLS 策略

迁移文件已开启 RLS 并只给 `service_role` 所有权限：

```sql
alter table public.telemetry_installations enable row level security;
...
create policy "service_role_only_telemetry_events"
  on public.telemetry_events
  for all
  to service_role
  using (true)
  with check (true);
```

**禁止**：
- 在浏览器/客户端用 anon key 直接读写这些表。
- 把 `SUPABASE_SERVICE_ROLE_KEY` 以 `NEXT_PUBLIC_` 前缀暴露。
- 在 API 错误响应中返回任何 key。

所有写入/读取都通过 Next.js API Route（server-side）使用 service role key。

---

## 6. API 接口契约

### 6.1 `POST /api/telemetry/event`

CLI 默认 endpoint：

```text
https://www.rosclaw.io/api/telemetry/event
```

#### Request body (JSON)

```json
{
  "schema_version": "rosclaw.telemetry.event.v1",
  "event_type": "command_completed",
  "anonymous_installation_id": "rci_a7d84c9e...",
  "created_at": "2026-06-24T18:00:00Z",
  "rosclaw_version": "1.0.3",
  "cli_version": "1.0.3",
  "os_family": "linux",
  "arch": "x86_64",
  "python": "3.12",
  "install_channel": "curl",
  "deployment_mode": "local",
  "command_name": "doctor",
  "command_status": "success",
  "duration_bucket": "1s-5s",
  "module_name": "core",
  "error_class_bucket": null,
  "payload": {}
}
```

#### 服务端校验逻辑

1. 必须是 `POST`。
2. `Content-Type: application/json`。
3. Body 大小 ≤ `ROSCLAW_TELEMETRY_MAX_EVENT_BYTES`（默认 16KB）。
4. `schema_version` 必须是 `rosclaw.telemetry.event.v1`。
5. `event_type` 必须在 allowlist 内。
6. `anonymous_installation_id` 必须是 `rci_` 开头，总长度 36 字符（`rci_` + 32 位 hex）。
7. 检查并**拒绝**任何 forbidden field（见 6.4）。
8. 限流：同一个 `anonymous_installation_id` 每小时最多 100 条事件；同 IP 每小时最多 1000 条。
9. Upsert `telemetry_installations`：
   - `anonymous_installation_id` 唯一。
   - 首次插入时 `first_seen_at` 和 `first_rosclaw_version`。
   - 每次更新 `last_seen_at` 和 `latest_rosclaw_version`。
10. Insert `telemetry_events`。
11. 返回 `request_id`。

#### Response

```json
{ "ok": true, "request_id": "tel_<uuid>" }
```

错误示例：

```json
{ "ok": false, "error": "forbidden_field", "field": "prompt" }
{ "ok": false, "error": "invalid_event_type", "event_type": "foo" }
{ "ok": false, "error": "rate_limited" }
{ "ok": false, "error": "payload_too_large" }
```

#### Payload 中自动附带的设备/环境信息

CLI 会在 `payload` 里自动附带以下**允许的**设备信息，方便后台做版本、机器人、传感器、GPU 分布分析：

```json
{
  "payload": {
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

说明：

- `robot_type` 来自 `rosclaw.yaml` 的 `runtime.robot_id`。
- `sensor_types` 来自 `body.yaml` 的 `installed_components.sensors` 键名列表。
- 如果某项探测不到，则不会出现在 payload 中（不会传 `null`）。
- 这些字段**不会**包含 hostname、username、ip、robot_serial 等敏感信息。

服务端可以直接把 payload 整体存入 `telemetry_events.payload`（`jsonb`），dashboard 用 `payload->>'robot_type'` 等 SQL 查询。

### 6.2 `POST /api/telemetry/heartbeat`

CLI 默认 endpoint：

```text
https://www.rosclaw.io/api/telemetry/heartbeat
```

逻辑可内部转发到 `telemetry/event` 的 ingest，但 `event_type` 固定为 `heartbeat`。

#### Request body

```json
{
  "schema_version": "rosclaw.telemetry.heartbeat.v1",
  "event_type": "heartbeat",
  "anonymous_installation_id": "rci_...",
  "created_at": "2026-06-24T18:00:00Z",
  "rosclaw_version": "1.0.3",
  "os_family": "linux",
  "arch": "x86_64",
  "python": "3.12",
  "enabled_modules": ["provider", "sandbox", "dashboard"]
}
```

#### 限流

同一个 `anonymous_installation_id` 每天最多 3 次 heartbeat（CLI 本地 24h 节流，服务端做兜底）。

### 6.3 `event_type` / `command_name` / `command_status` / bucket 白名单

#### event_type 白名单

```text
install_started
install_completed
firstboot_started
firstboot_completed
doctor_started
doctor_completed
command_completed
module_enabled
provider_installed
provider_served
hub_asset_installed
dashboard_opened
practice_started
heartbeat
telemetry_ping
```

#### command_name 白名单

```text
doctor
firstboot
provider
hub
practice
dashboard
sandbox
memory
skill
body
mcp
feedback
version
help
```

#### command_status 白名单

```text
success
failure
cancelled
timeout
```

#### duration_bucket 白名单

```text
<100ms
100ms-1s
1s-5s
5s-30s
30s-5m
>5m
```

#### error_class_bucket 白名单

```text
ImportError
ConfigError
DockerUnavailable
ROSNotFound
ProviderTimeout
PermissionDenied
NetworkError
ValidationError
RuntimeError
Unknown
```

任何不在白名单内的值都应该被拒绝或归入 `Unknown`（建议 P2 直接拒绝以便发现异常）。

### 6.4 Forbidden fields（必须拒绝）

请求 JSON 中**出现以下任一顶层或嵌套字段**，服务端必须返回 `forbidden_field` 错误，且**不能写入数据库**。

```text
hostname
username
ip
local_path
cwd
full_command
full_args
prompt
system_prompt
tool_arguments
provider_response
stacktrace
log
video
image
audio
mcap
trace
api_key
secret
robot_serial
```

建议实现：递归遍历 JSON，一旦发现 key 在白名单外且属于 forbidden，立即拒绝。

### 6.5 `POST /api/feedback/upload`

CLI 默认 endpoint：

```text
https://www.rosclaw.io/api/feedback/upload
```

#### Content-Type

`multipart/form-data`

#### Form fields

| 字段 | 类型 | 说明 |
|---|---|---|
| `bundle` | File (required) | `.tar.gz` 反馈包，包含 `manifest.json`、`telemetry.jsonl`、`feedback.jsonl` |
| `schema_version` | string | 必须是 `rosclaw.feedback.upload.v1` |
| `anonymous_installation_id` | string | `rci_...` |
| `client_version` | string | ROSClaw CLI 版本 |
| `redacted` | string/boolean | 必须为 `true`；否则拒绝 |
| `media_count` | integer | 包内媒体文件数量 |
| `days` | integer | 包含最近多少天的事件 |

#### 服务端处理流程

1. 校验 `redacted=true`。
2. 校验 `anonymous_installation_id` 格式。
3. 校验 bundle 大小 ≤ `ROSCLAW_FEEDBACK_MAX_BUNDLE_MB`（默认 25MB）。
4. 计算 bundle `sha256`。
5. 上传到 Supabase Storage bucket：`feedback-bundles`。
   - 建议路径：`{anonymous_installation_id}/{batch_id}/bundle.tar.gz`
   - bucket 必须私有，仅服务端可读。
6. Insert `feedback_batches` 一行，状态 `received`。
7. 解压 bundle（内存中或临时文件），读取 `manifest.json` 和 `feedback.jsonl`。
8. 对 `feedback.jsonl` 中每一行：
   - 校验 `category` 白名单。
   - Insert `feedback_events`。
9. 如有附件（media、mcap 摘要等），insert `feedback_attachments` 并上传到 `feedback-attachments` bucket。
10. 更新 `feedback_batches.event_count` / `attachment_count`。
11. 返回 `request_id`（即 `batch_id`）。

#### Response

```json
{ "ok": true, "request_id": "fb_<uuid>", "batch_id": "<uuid>" }
```

#### feedback category 白名单

```text
failure_stats
skill_performance
crash_summary
human_feedback
sandbox_block
provider_performance
```

### 6.6 `POST /api/feedback/delete-request`（P2 基础）

用户可通过 CLI/网页请求删除与其匿名 ID 关联的数据。

```json
{
  "anonymous_installation_id": "rci_...",
  "reason": "I want to delete my data"
}
```

服务端 insert `feedback_delete_requests` 一行，状态 `pending`；后台人工或自动脚本处理。

---

## 7. Admin Dashboard API

### 7.1 `GET /api/admin/telemetry/summary`

返回核心指标：

```json
{
  "total_installs": 1234,
  "dau": 89,
  "wau": 345,
  "mau": 890,
  "firstboot_completion_rate": 0.72,
  "doctor_success_rate": 0.91,
  "top_commands": [
    { "command_name": "doctor", "count": 500 },
    { "command_name": "firstboot", "count": 300 }
  ],
  "version_distribution": [
    { "rosclaw_version": "1.0.3", "count": 800 }
  ],
  "os_distribution": [
    { "os_family": "linux", "count": 700 }
  ]
}
```

实现方式：
- P2 可直接从 `telemetry_events` 实时 count（数据量小时足够）。
- 数据量大后从 `telemetry_daily_aggregates` 读取。

### 7.2 `POST /api/admin/telemetry/aggregate`

Vercel Cron 每小时调用，也支持手动触发。

聚合当天（以及补跑前一天）指标并写入 `telemetry_daily_aggregates`。

必须包含的 metric_name（`dimension` 用 JSONB，空对象 `{}` 表示全局）：

```text
total_installs
daily_active_installs
weekly_active_installs
monthly_active_installs
install_completed_count
firstboot_completed_count
firstboot_completion_rate
doctor_total
doctor_success
doctor_success_rate
command_count_by_name          # dimension: { command_name: "doctor" }
command_failure_rate_by_name   # dimension: { command_name: "doctor" }
module_adoption_by_name        # dimension: { module_name: "provider" }
version_distribution           # dimension: { rosclaw_version: "1.0.3" }
os_distribution                # dimension: { os_family: "linux" }
python_distribution            # dimension: { python: "3.12" }
```

示例聚合逻辑（PostgreSQL）：

```sql
insert into public.telemetry_daily_aggregates (day, metric_name, dimension, value)
select
  date(received_at) as day,
  'command_count_by_name' as metric_name,
  jsonb_build_object('command_name', command_name) as dimension,
  count(*) as value
from public.telemetry_events
where event_type = 'command_completed'
  and received_at >= current_date - interval '1 day'
group by day, command_name
on conflict (day, metric_name, dimension) do update set
  value = excluded.value,
  updated_at = now();
```

### 7.3 `GET /api/admin/feedback/summary`

```json
{
  "total_batches": 12,
  "total_events": 150,
  "crashes": 3,
  "failure_stats": 80,
  "sandbox_blocks": 10,
  "provider_performance_reports": 20,
  "human_feedback": 5
}
```

### 7.4 权限

P2 推荐简单方案：

```bash
ROSCLAW_ADMIN_EMAILS=shaoxiang007@gmail.com,another@example.com
```

在 API route 中校验当前登录用户 email（或 Vercel Basic Auth）是否在 allowlist。不要公开 admin 页面。

---

## 8. Admin Dashboard 页面

### 8.1 `/admin/telemetry`

必须展示：

- Overview Cards：Total Installs / DAU / WAU / MAU / Firstboot Completion / Doctor Success Rate
- Charts：Installs over time / Active installs over time / Version distribution / OS distribution
- Tables：Top Commands / Command Failure Rate / Module Adoption / Recent Events

### 8.2 `/admin/feedback`

- Overview：Feedback batches / Crash summaries / Failure stats / Sandbox blocks / Provider performance
- Failure Trends：Top failure types / Top affected skills
- Provider Panel：Latency buckets / Timeout rate
- Sandbox Panel：Block reasons / Risk levels
- Human Feedback：Ratings / Labels / Redacted comments

### 8.3 `/privacy/telemetry`

面向用户的隐私说明页，文案可直接复用 CLI firstboot 中的：

```text
ROSClaw 默认开启轻量匿名产品遥测，用于了解安装成功率、活跃使用量、版本分布和命令可靠性。
诊断包、prompt、日志、媒体和 trace 默认绝不上传，必须由用户主动触发，并默认脱敏。
```

---

## 9. Vercel Cron 配置

在 `vercel.json` 中新增：

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

每小时整点跑一次聚合。

---

## 10. 环境变量

在 Vercel Project Settings 中配置：

```bash
# Supabase（SUPABASE_SERVICE_ROLE_KEY 绝对不能暴露给前端）
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<anon>
SUPABASE_SERVICE_ROLE_KEY=<service_role>

# 功能开关
NEXT_PUBLIC_TELEMETRY_NOTICE_ENABLED=true
ROSCLAW_TELEMETRY_MAX_EVENT_BYTES=16384
ROSCLAW_TELEMETRY_RATE_LIMIT_ENABLED=true
ROSCLAW_FEEDBACK_REQUIRE_REDACT=true
ROSCLAW_FEEDBACK_MAX_BUNDLE_MB=25

# 管理员
ROSCLAW_ADMIN_EMAILS=shaoxiang007@gmail.com
```

本地开发用 `.env.local`，不要提交到 Git。

---

## 11. Supabase Storage

创建两个私有 bucket：

- `feedback-bundles`：存放用户上传的 `tar.gz` bundle。
- `feedback-attachments`：存放媒体/MCAP 摘要等附件。

Bucket 权限：
- 不启用 public URL。
- 只允许 service role 读写。
- Admin dashboard 通过服务端 `createSignedUrl` 临时签名 URL 预览/下载。

---

## 12. 本地开发与联调

### 12.1 启动 website 本地环境

```bash
cd rosclaw-website
npm install

# 启动 Supabase 本地
supabase start
supabase db reset

# 启动 Next.js
npm run dev
```

本地地址：`http://localhost:3000`

### 12.2 让 CLI 指向本地 website

新开一个终端：

```bash
export ROSCLAW_HOME=$(mktemp -d)
export ROSCLAW_TELEMETRY_ENDPOINT=http://localhost:3000/api/telemetry/event
export ROSCLAW_FEEDBACK_ENDPOINT=http://localhost:3000/api/feedback/upload

rosclaw firstboot --yes --profile offline --robot sim_ur5e
rosclaw doctor
rosclaw feedback telemetry ping
```

执行后应能在 Supabase Studio (`http://localhost:54323`) 中看到 `telemetry_events` 新增记录。

### 12.3 测试 feedback upload

```bash
# 先导出+上传干跑
rosclaw feedback export --redact --days 7
rosclaw feedback upload --redact --dry-run

# 真实上传
rosclaw feedback upload --redact --days 7
```

成功后：
- `feedback-bundles` bucket 出现文件。
- `feedback_batches` 新增一行。
- `feedback_events` 按 category 解析入库。

### 12.4 测试关闭遥测

```bash
rosclaw feedback telemetry off
rosclaw doctor
```

再次查看 `telemetry_events`，应无新的 `command_completed`。

### 12.5 测试 forbidden field 防线

用 curl 直接 POST 脏数据：

```bash
curl -X POST http://localhost:3000/api/telemetry/event \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": "rosclaw.telemetry.event.v1",
    "event_type": "command_completed",
    "anonymous_installation_id": "rci_00000000000000000000000000000000",
    "created_at": "2026-06-24T18:00:00Z",
    "prompt": "this should be rejected"
  }'
```

期望返回：

```json
{ "ok": false, "error": "forbidden_field", "field": "prompt" }
```

---

## 13. 验收清单（Definition of Done）

### Telemetry API

- [ ] `POST /api/telemetry/event` 接受合法事件并返回 `request_id`。
- [ ] 未知 `event_type` 被拒绝。
- [ ] forbidden field 被拒绝且不写入数据库。
- [ ] 超大 payload 被拒绝。
- [ ] 限流生效（至少按 anon id / IP 有基本限制）。
- [ ] `telemetry_installations` upsert 正确更新 `last_seen_at`。
- [ ] `POST /api/telemetry/heartbeat` 复用 ingest 逻辑，且每天最多 3 次。

### Feedback API

- [ ] `POST /api/feedback/upload` 只接受 `multipart/form-data`。
- [ ] `redacted != true` 时被拒绝。
- [ ] bundle 文件写入 `feedback-bundles` bucket。
- [ ] `feedback_batches` / `feedback_events` 正确入库。
- [ ] 未知 `category` 被拒绝。

### Admin Dashboard

- [ ] `/admin/telemetry` 非管理员不可访问。
- [ ] 页面展示 Total Installs / DAU / WAU / MAU / Firstboot / Doctor success。
- [ ] `/admin/feedback` 展示 batch/event 概览。
- [ ] `/privacy/telemetry` 页面可公开访问。

### Cron & Aggregation

- [ ] `vercel.json` cron 配置正确。
- [ ] `/api/admin/telemetry/aggregate` 手动触发可写入 `telemetry_daily_aggregates`。
- [ ] 聚合指标覆盖验收列表中的 metric_name。

### E2E

- [ ] CLI `firstboot` → 服务端收到 `install_completed` + `firstboot_completed`。
- [ ] CLI `doctor` → 服务端收到 `command_completed`。
- [ ] CLI `feedback telemetry ping` → 服务端收到 `telemetry_ping`。
- [ ] CLI `feedback upload --redact` → `feedback_batches` 新增。
- [ ] CLI `feedback telemetry off` → 后续无事件上传。

---

## 14. 常见问题与排查

| 现象 | 可能原因 | 排查 |
|---|---|---|
| CLI ping 成功但 DB 无数据 | RLS 阻止写入 | 确认 API route 使用 service role key |
| 上传返回 `forbidden_field` | 请求里带了 hostname/username 等 | 检查 CLI 是否最新版本 |
| feedback upload 报 `bundle_too_large` | bundle 超过 25MB | 检查本地 `~/.rosclaw/feedback` 大小 |
| dashboard 看不到数据 | aggregate 未跑或 admin API 用了 anon key | 检查 cron/aggregate 是否写入 daily aggregates |
| 本地联调 CLI 发不到 localhost | 环境变量未生效 | 确认 `ROSCLAW_TELEMETRY_ENDPOINT` 等已 export |
| rate limit 误伤 | 限流计数未按 anon id 区分 | 检查限流 key 是否包含 anon id |

---

## 15. 与 CLI 侧的对应关系

| 服务端文件 | CLI 侧对应 |
|---|---|
| `app/api/telemetry/event/route.ts` | `src/rosclaw/feedback/telemetry_client.py` |
| `app/api/feedback/upload/route.ts` | `src/rosclaw/feedback/upload.py` |
| `supabase/migrations/004_telemetry_feedback_schema.sql` | `supabase/migrations/004_telemetry_feedback_schema.sql`（已提供） |
| `lib/telemetry/schema.ts` | `src/rosclaw/feedback/telemetry_client.py` 中的 `ALLOWED_EVENT_TYPES` / `FORBIDDEN_FIELDS` |
| `lib/feedback/schema.ts` | `src/rosclaw/feedback/config.py` 中的 `FeedbackConfig` |

---

## 16. 下一步建议

1. 在 `rosclaw-website` 创建 `feat/telemetry-ingest` 分支。
2. 先跑通 `POST /api/telemetry/event` + Supabase 写入 + 本地 CLI ping。
3. 再做 admin dashboard 基础版。
4. 最后做 feedback upload + storage + dashboard。
5. 每个里程碑都跑一遍上面的 E2E 清单。

---

**输出文件**：`docs/WEBSITE_VERCEL_FEEDBACK_TELEMETRY_HANDOFF.md`  
**更新时间**：2026-06-24
