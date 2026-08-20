# ROSClaw × OpenClaw Channel 安全基线

本文冻结 Channel 集成的信任边界（Channel 设计 §4/§39/§40）。任何配置变更
必须对照本文评审。

## 信任边界

```text
Feishu User → OpenClaw identity → ACP request → ROSClaw Agent intent → Proposal
                              到这里停止。

REAL：Proposal → Trusted Operator → Permit → rosclawd   （另一条 trust path）
```

- **飞书/Discord 永远不直接访问 rosclawd，也永远不获得 Permit。**
- ACP `request_permission` 不得代替 ROSClaw Operator Authorization。
- 不自动批准 REAL Action。
- 飞书不收 daemon challenge；模型读不到 daemon challenge。
- 以下测试必须永久存在（设计 §4）：

```python
assert acp_client_cannot_issue_permit()
assert channel_message_cannot_issue_permit()
assert openclaw_permission_cannot_issue_permit()
assert model_cannot_read_daemon_challenge()
```

现有回归：`tests/agentd/test_acp.py::TestAcpMapping::test_no_authority_via_acp`
（ACP 路径 `list_grants() == []` 且 `pending_approvals() == []`）。

## 必须冻结的配置

| 项 | 值 | 原因 |
|----|----|------|
| `gateway.bind` | `loopback` | 飞书走 outbound WebSocket，无需公网 Gateway（§13） |
| `gateway.auth.mode` | `token`（env 注入） | 禁止无认证；token 文件 chmod 600 |
| `acp.allowedAgents` | `["rosclaw"]` | Gateway 只把消息送入 ROSClaw（§11） |
| `acpx.permissionMode` | `deny-all` | ACP harness 权限 ≠ ROSClaw 物理权限（§10） |
| `pluginToolsMcpBridge` / `openClawToolsMcpBridge` | `false` | 不注入第二套 tool surface（§10） |
| `channels.feishu.dmPolicy` | `pairing` | 陌生人先配对（§16/§18） |
| `channels.feishu.groupPolicy` | `allowlist` + `requireMention` | 群聊白名单 + @门槛 |
| `channels.feishu.tools.*` | 全 `false` | Channel 只是 transport（§17） |
| `dynamicAgentCreation.enabled` | `false` | 不创建第二套 Agent ownership（§32） |
| `session.dmScope` | `per-channel-peer` | 用户间 Mission 隔离（§31） |

**禁止**：`gateway.bind = lan` / `0.0.0.0`、公网暴露 AgentService、公网暴露
rosclawd、`dmPolicy = open`、`groupPolicy = open`。

## 身份模型（§39/§40）

`RuntimePrincipal`（`user:local:<uid>`，与本机 Unix trust model 关联）与
`ChannelIdentity`（`feishu:default:ou_xxx`）是两个概念：

- 禁止把 `owner_principal` 设成飞书身份。
- ChannelIdentity 只用于 audit / routing / provenance，不能用于 daemon
  authorization。
- 在专门的 Operator identity bridge 完成之前，外部 sender metadata 一律
  标记为 `untrusted presentation identity`；不要发明 `X-Feishu-User`
  并误当可信身份。

## OS 用户（§41）

- `openclaw` + `rosclaw-agentd`：非特权用户。
- `rosclawd`：另一个特权服务身份。
- 首版 OpenClaw 与 `rosclaw acp serve` 可同一普通用户（共享
  ROSCLAW_HOME）；注意这意味着 OpenClaw 进程被攻破时可读该用户可读的
  Agent credentials。P3 增强：OpenClaw → ACP thin proxy → Unix
  AgentService socket，让 OpenClaw 不直接持有模型密钥。

## ACP stdio 纯净（§36）

`rosclaw acp serve` 的 stdout 只承载 JSON-RPC 帧；一切日志走 stderr。
回归测试：`tests/agentd/test_acp.py::TestStdioProtocol::
test_stdout_contains_only_jsonrpc`。

## Secret 边界（§64）

Channel transcript / OpenClaw logs / ROSClaw logs / ACP events / Mission
journal 中搜索以下项必须为 0：

```text
OPENCLAW_GATEWAY_TOKEN
FEISHU_APP_SECRET
ROSCLAW_KIMI_API_KEY
MOONSHOT_API_KEY
daemon challenge
Permit HMAC
```

## Release Gate（§68/§76）

```text
session_misroute_count = 0
unauthorized_physical_decision_count = 0
permit_material_leak_count = 0
lost_final_message_count = 0
```
