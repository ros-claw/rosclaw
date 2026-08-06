"""rosclaw-agentd — the ROSClaw-owned Native Agent process.

Maturity: **experimental** (ADR-0001). Owns missions, task graphs, embodied
context, model policy, cognitive workers and the team client. Holds **no**
hardware authority: every physical action is a request to rosclawd.
"""

MATURITY = "experimental"
