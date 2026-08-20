# ROSClaw × OpenClaw Channel 集成

> **ROSClaw owns the Agent. OpenClaw owns the Channel. ACP owns the bridge.
> rosclawd owns physical authority.**

本目录是 ROSClaw 接入 OpenClaw Channel（飞书/Discord/…）的**配置与验证
脚手架**。ROSClaw 不实现任何 Channel adapter——消息平台连接、pairing、
allowlist、去重、重试、线程路由、流式卡片全部复用 OpenClaw 官方能力。
**不要 vendor OpenClaw，也不要在本仓库复制其 Channel 代码。**

```text
Feishu / Discord
      │
   OpenClaw Gateway（连接、鉴权、pairing、去重、流式卡片）
      │ ACP（stdio JSON-RPC）
      ▼
rosclaw acp serve        ← 本仓库的 ACP adapter（src/rosclaw/adapters/acp/）
      ▼
AgentService → Mission → Native Agent → Tools/Workers
      ▼
Operator Proposal → Operator Broker → rosclawd → Robot
```

## 文件

| 文件 | 说明 |
|------|------|
| `version.lock` | E2E 验证通过的组件版本锁（OpenClaw/acpx/feishu/Node/ACP SDK） |
| `install-openclaw.sh` | 幂等安装：版本窗口检查 + 锁版本装 OpenClaw 与插件 |
| `configure-openclaw.sh` | 幂等配置：安全基线 + rosclaw harness + feishu 策略（不碰密钥） |
| `openclaw.rosclaw.example.json5` | 单机器人完整逻辑配置样例（设计 §33） |
| `SECURITY.md` | 信任边界与安全配置基线 |
| `e2e/acpx_direct_probe.py` | Stage 2 直连 E2E（§51）：ACPX → rosclaw harness |
| `e2e/acp_continuity_probe.py` | kill ACP 子进程 → resume 的连续性 E2E（§49/§52） |
| `README.md` | 本文件 |

## 新机器部署（推荐顺序）

```bash
integrations/openclaw/install-openclaw.sh          # 1. 锁版本安装
openclaw channels login --channel feishu           # 2. 配飞书凭证（唯一需要密钥的步骤）
integrations/openclaw/configure-openclaw.sh \      # 3. 应用安全基线 + harness
    --rosclaw-bin /abs/path/.venv/bin/rosclaw \
    --rosclaw-home /home/ubuntu/.rosclaw \
    --dm-user ou_<USER_OPEN_ID>                    # 可选：DM 直接绑定该用户
openclaw gateway restart                           # 4. 生效
rosclaw channel doctor --require-openclaw          # 5. 验收
# 6. 群聊接入：把群 oc_xxx 加 groupAllowFrom 后，在群里发
#    @机器人 /acp spawn rosclaw --bind here
```

以下为手动分步说明（脚本已自动化其中的大部分）：

Doctor 实现不在本目录——它是 ROSClaw 产品化 CLI 的一部分：

```bash
rosclaw channel doctor                  # 只读检查，不改任何配置
rosclaw channel doctor --require-openclaw
rosclaw channel setup feishu            # 打印配置指引
```

（设计 §45 的文件清单中 `doctor.py` 落实为
`src/rosclaw/integrations/openclaw/doctor.py`，以便 `rosclaw channel
doctor` 直接复用；本目录只保留配置与文档，避免双份实现漂移。）

## Bring-up 步骤

### 0. 前置

```bash
# ROSClaw 侧：ACP 依赖版本已按验证锁定（pyproject.toml, >=0.12,<0.13）
rosclaw --help
rosclaw agent doctor
pytest tests/agentd/test_acp.py -q     # ACP adapter gate，必须 100% PASS
```

### 1. 安装 OpenClaw（Node 22.22.3+ / 24.15+ / 25.9+）

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
# 或严格版本锁：npm install -g openclaw@<TESTED_VERSION>
openclaw --version
openclaw doctor
```

不要在 CI 里用 `openclaw@latest`：第一次 bring-up 用当前 stable，E2E
全绿后记录 exact tested version 并锁进 CI（设计 §7）。飞书 Channel 要求
OpenClaw ≥ 2026.5.29。

### 2. 安装 ACPX 并注册 ROSClaw harness

```bash
openclaw plugins install @openclaw/acpx
openclaw config set plugins.entries.acpx.enabled true
```

把 `openclaw.rosclaw.example.json5` 中 `plugins.entries.acpx.config.
agents.rosclaw` 的 `command`/`args` 改成**本机绝对路径**后合入 OpenClaw
配置。不要依赖 PATH / cwd / shell alias / conda activate。

### 3. 冻结安全配置

逐项对照 `SECURITY.md`：Gateway loopback + token 认证、
`permissionMode: deny-all`、两个 MCP bridge 关闭、
`allowedAgents: ["rosclaw"]`、飞书 `pairing + allowlist + mention`、
飞书 workspace tools 全关、`dynamicAgentCreation.enabled: false`。

> OpenClaw 配置 schema 仍在演进：合入配置后必须用目标版本的
> `openclaw doctor` / `openclaw config` 验证，不允许只复制样例跳过
> schema validation（设计 §33 注）。

### 4. 飞书

```bash
openclaw channels login --channel feishu   # 引导配置 App ID / App Secret
openclaw gateway restart
openclaw gateway status
```

飞书开放平台：自建应用 → 启用机器人 → 发布 → 事件订阅选 **WebSocket 长
连接** → 订阅 `im.message.receive_v1`（设计 §15）。

### 5. 验证

```bash
rosclaw channel doctor --require-openclaw
```

随后按设计文档 §49–§67 执行 E2E：LIVE-ROSCLAW-ACP-OK、session
continuity、pairing、20-turn、多用户隔离、群聊 mention、duplicate
event、Gateway restart、reasoning、worker 可观测性、物理安全红队。

## Rollback（设计 §74）

```bash
openclaw config set acp.dispatch.enabled false     # 或 channels.feishu.enabled false
openclaw gateway restart
```

Channel 关闭后 `rosclaw chat` / `rosclaw-agentd` / `rosclawd` 必须继续正常
——Channel 是 external integration，故障不能拖垮 ROSClaw Core。

## Multi-Robot（设计 §34）

每台机器人 = 独立 `ROSCLAW_HOME` + 独立 ACP harness + 独立 OpenClaw ACP
agent，用 OpenClaw persistent binding 做静态路由。不要靠聊天里一句
"切到 G1" 让 LLM 自己换 body。

## 参考资料（只读，不 fork）

```bash
mkdir -p /tmp/rosclaw-channel-reference && cd /tmp/rosclaw-channel-reference
git clone --depth 1 https://github.com/openclaw/openclaw.git
git clone --depth 1 https://github.com/openclaw/acpx.git
git clone --depth 1 https://github.com/agentclientprotocol/python-sdk.git
```

## 实测要点（OpenClaw 2026.7.1-2 验证）

设计文档与实测之间的差异，按版本可能继续演进，以 `openclaw config schema` 为准：

1. **channel 消息进 ACP runtime 需要 binding，不是 `agents.list[].runtime.type=acp` 就够**。
   `runtime.type=acp` 只声明"该 agent 是 ACP harness 后端"；路由要：
   - DM：configured binding，且 `match.peer.id` 是**用户 open_id**（`ou_...`），不是 p2p chat id（`oc_...`）：
     ```bash
     openclaw config set bindings '[{"type":"acp","agentId":"rosclaw","match":{"channel":"feishu","peer":{"kind":"direct","id":"ou_<USER_OPEN_ID>"}},"acp":{"mode":"persistent","backend":"acpx","cwd":"/home/nvidia","label":"rosclaw"}}]'
     ```
   - 群/topic 会话：configured binding 不命中 topic 作用域会话，在群里发
     `@机器人 /acp spawn rosclaw --bind here`（gateway 斜杠命令）建立 persistent binding。
   - ACP binding 必须带具体 `match.peer`（不支持整 channel 通配）——新用户/新群各加一条，符合设计 §34 静态路由意图。
2. **`openclaw agent --agent rosclaw -m ...` CLI 不走 ACP runtime**（走 embedded 模型路径）；E2E 用飞书消息或 `integrations/openclaw/e2e/acpx_direct_probe.py`。
3. `channels.feishu.streaming` 在此版本是**布尔**（`true`），不是设计 §30 的 object。
4. 已有 embedded 会话记录会"粘住"路由——新 binding 生效前用
   `openclaw gateway call sessions.delete --params '{"key":"<sessionKey>"}'` 清掉旧会话（transcript 自动归档为 `.deleted` 备份）。
5. `openclaw doctor --fix` 可能把配置回滚到 last-known-good——任何 config 变更后复查关键项（`gateway.bind`、`bindings`、`acp.*`）。
