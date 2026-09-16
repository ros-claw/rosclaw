"""WorldSpec（PR-MH7，ADR-0014，规格 §22-§24）。

Maturity: experimental（ADR-0000 §4）。

WorldSpec 是 Physical AI 的语义，MJCF 是其中一个物理实现：
Body 由 e-URDF 管，WorldSpec 管世界与任务。吸收 Text2Mujoco
scene_spec 思想（NL 理解与物理实现分离、typed interaction、
success/failure 机器谓词），但不复制 robot schema。
"""
