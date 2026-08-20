#!/usr/bin/env bash
# ROSClaw × OpenClaw 配置脚本（设计 §9-§16/§33）。
#
# 幂等应用安全基线与 rosclaw ACP harness 配置。所有配置经
# `openclaw config set` + schema validate，不手写 openclaw.json。
#
# 本脚本**不触碰**任何密钥：飞书 App ID/Secret 走
# `openclaw channels login --channel feishu`；Gateway token 走环境变量。
#
# 用法：
#   integrations/openclaw/configure-openclaw.sh \
#       --rosclaw-bin /abs/path/.venv/bin/rosclaw \
#       --rosclaw-home /home/ubuntu/.rosclaw \
#       [--dm-user ou_xxx]          # 可选：为指定飞书用户建 DM ACP binding
#       [--group oc_xxx]            # 可选：群加入 allowlist
set -euo pipefail

ROSCLAW_BIN=""
ROSCLAW_HOME=""
DM_USER=""
GROUP_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --rosclaw-bin)  ROSCLAW_BIN="$2"; shift 2 ;;
    --rosclaw-home) ROSCLAW_HOME="$2"; shift 2 ;;
    --dm-user)      DM_USER="$2"; shift 2 ;;
    --group)        GROUP_ID="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$ROSCLAW_BIN" ] || [ -z "$ROSCLAW_HOME" ]; then
  echo "必须提供 --rosclaw-bin 和 --rosclaw-home（绝对路径）" >&2
  exit 2
fi
case "$ROSCLAW_BIN" in /*) ;; *) echo "--rosclaw-bin 必须是绝对路径（设计 §9）" >&2; exit 2 ;; esac
case "$ROSCLAW_HOME" in /*) ;; *) echo "--rosclaw-home 必须是绝对路径" >&2; exit 2 ;; esac
[ -x "$ROSCLAW_BIN" ] || { echo "FAIL: $ROSCLAW_BIN 不可执行" >&2; exit 1; }

set_cfg() { openclaw config set "$1" "$2" >/dev/null; echo "  $1 = $2"; }

echo "==> Gateway 安全（§13：loopback + token；token 用环境变量注入）"
set_cfg gateway.bind loopback
set_cfg gateway.auth.mode token

echo "==> acpx harness（§9/§10）"
set_cfg plugins.entries.acpx.enabled true
set_cfg plugins.entries.acpx.config.agents.rosclaw.command "$ROSCLAW_BIN"
set_cfg plugins.entries.acpx.config.agents.rosclaw.args "[\"acp\",\"serve\",\"--home\",\"$ROSCLAW_HOME\"]"
set_cfg plugins.entries.acpx.config.permissionMode deny-all
set_cfg plugins.entries.acpx.config.nonInteractivePermissions deny
set_cfg plugins.entries.acpx.config.pluginToolsMcpBridge false
set_cfg plugins.entries.acpx.config.openClawToolsMcpBridge false
set_cfg plugins.entries.acpx.config.probeAgent rosclaw

echo "==> ACP 核心（§11）"
set_cfg acp.enabled true
set_cfg acp.dispatch.enabled true
set_cfg acp.backend acpx
set_cfg acp.defaultAgent rosclaw
set_cfg acp.allowedAgents '["rosclaw"]'
set_cfg acp.stream.deliveryMode live
set_cfg session.dmScope per-channel-peer

echo "==> rosclaw agent（§12；非默认，不抢占既有 main agent）"
set_cfg agents.list '[{"id":"rosclaw","runtime":{"type":"acp","acp":{"agent":"rosclaw","backend":"acpx","mode":"persistent","cwd":"/home/nvidia"}}}]'

echo "==> 飞书安全基线（§16/§17/§30/§32）"
set_cfg channels.feishu.connectionMode websocket
set_cfg channels.feishu.dmPolicy pairing
set_cfg channels.feishu.groupPolicy allowlist
set_cfg channels.feishu.requireMention true
set_cfg channels.feishu.groupSessionScope group_topic_sender
set_cfg channels.feishu.replyInThread disabled
set_cfg channels.feishu.streaming true
set_cfg channels.feishu.dynamicAgentCreation '{"enabled":false}'
set_cfg channels.feishu.tools '{"doc":false,"chat":false,"wiki":false,"drive":false,"perm":false,"scopes":false,"bitable":false}'

echo "==> ACP bindings（§19/§34：每会话一条，静态路由）"
BINDINGS="[]"
if [ -n "$DM_USER" ]; then
  BINDINGS="[{\"type\":\"acp\",\"agentId\":\"rosclaw\",\"match\":{\"channel\":\"feishu\",\"peer\":{\"kind\":\"direct\",\"id\":\"$DM_USER\"}},\"acp\":{\"mode\":\"persistent\",\"backend\":\"acpx\",\"cwd\":\"/home/nvidia\",\"label\":\"rosclaw\"},\"comment\":\"ROSClaw ACP DM binding\"}]"
  set_cfg channels.feishu.allowFrom "[\"$DM_USER\"]"
  echo "  （dmPolicy 改用 allowlist + allowFrom；若坚持 pairing 流程请手动改回并不设 allowFrom）"
fi
if [ -n "$GROUP_ID" ]; then
  set_cfg channels.feishu.groupAllowFrom "[\"$GROUP_ID\"]"
  echo "  群 $GROUP_ID 已入 allowlist。群会话绑定请在群内执行：@机器人 /acp spawn rosclaw --bind here"
  echo "  （topic 作用域群会话的 configured binding 不生效——实测结论，见 README 实测要点）"
fi
if [ "$BINDINGS" != "[]" ]; then
  set_cfg bindings "$BINDINGS"
fi

echo
echo "==> schema 校验"
openclaw config validate

echo
echo "完成。重启生效："
echo "  openclaw gateway restart"
echo "验收："
echo "  rosclaw channel doctor --require-openclaw"
if [ -z "$DM_USER" ]; then
  echo
  echo "提示：未提供 --dm-user。新用户 DM 走 pairing 审批："
  echo "  openclaw pairing list feishu && openclaw pairing approve feishu <CODE>"
  echo "  审批后为该用户加 DM binding：重跑本脚本并加 --dm-user ou_xxx"
fi
