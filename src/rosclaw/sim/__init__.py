"""ROSClaw Simulation Harness（ADR-0014）。

Maturity: experimental（ADR-0000 §4）。

MuJoCo 是物理仿真引擎，不是 Agent Harness Backend（ADR-0014）。
本包收束 sim.api / sandbox / SimForge 共享的仿真原语；Model /
State / Trace 不可变；仿真证据永远 SIMULATED，永不生成 REAL permit。
"""

MATURITY = "experimental"
