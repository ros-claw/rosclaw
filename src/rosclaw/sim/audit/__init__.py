"""Physical Honesty Audit（PR-MH4，ADR-0014，规格 §17-§19）。

Maturity: experimental（ADR-0000 §4）。

不相信 Agent 说它完成了，也不只相信最终帧——相信物理过程和
可复现证据。A01-A08 吸收 Text2Mujoco（MIT）八项模型检查的思想
（showcase/model_audit.py），A15-A20 为 ROSClaw 基础扩展。
阈值全部收拢为 named policy（规格 §17.3：不散落 magic number）。
"""
