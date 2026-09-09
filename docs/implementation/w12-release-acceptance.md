# W12 发布验收矩阵与批准边界（规格 §16）

**日期**：2026-09-09　**基线**：main（W00–W11(1) 合入后）
**当前发布级别主张**：内部候选 → SIM Alpha（范围见下，冻结于测评前）

## 版本与发布物决议

- PyPI 现状：`rosclaw` 仅 0.0.1（占位）——1.2.0 不冲突。
- 发布物：PyPI wheel（Alpha 形式：含预构建 JS，需宿主
  Node ≥22.19，README 已明示）+ 离线 tar（自包含 Node）。
- 流程：TestPyPI 演练 → 用户批准 → tag `v1.2.0` → OIDC
  Trusted Publishing（release 环境，仅 tag 触发，普通 PR 无
  上传权限；上传 Build Package 的同一产物，不重建）。

## 硬 Gate 矩阵（§16.2 逐条对证）

| # | Gate | 证据 | 状态 |
|---|---|---|---|
| G1 | 0907 unsafe 成功表述回归 | 大道至简 R0-2 kernel 只报客观事实 + 语义验收器（WP-6/0827-P0-5）；真实主模型复跑 | **部分**：合成层绿；真实主模型复跑 NOT_RUN（无 key） |
| G2 | 完整 qpos/nv/nu、模型身份、坐标与时间 | W03 完整状态记录/回放 + LEGACY_PARTIAL_STATE；W02 LEGACY_MODEL_IDENTITY_MISSING；W04 时间驱动选帧 | ✅ 绿（tests/eval + w03/w04 套件） |
| G3 | 允许/禁止接触分离，未评估≠PASS | rollout collision_pairs vs contacts 分离（W03 保留）；verifier NOT_EVALUATED 语义（R0-5） | ✅ 绿 |
| G4 | 并发/取消/重启/重复不重复执行；旧 revision 不改新工作 | W04 渲染身份键双视角共存；W05 追加交付不复活/迟到请求不激活新 revision；idempotency 重放 | ✅ 绿 |
| G5 | 交付可解码/打开/导出；无旧 Artifact 冒充 | W07 CLI 契约 6 例 + W04 GIF 可解码 + 内容寻址幂等 | ✅ 绿 |
| G6 | 隔离负向有效；fake REAL 未授权命令数=0 | W06 探针（ready⇒smoke、evidence kernel-only、REAL 永不 POLICY_AUTO）+ consent 链 | ✅ 绿（本机 bwrap broken 声明已降级为进程内 provenance） |
| G7 | 主模型协议续接/恢复/压缩/工具语义 | W01 补丁退役恢复官方 thinkingSignature 回放 + journey A（含 compact/resume） | 部分：合成 provider 绿；真实 K3 续接 NOT_RUN |
| G8 | 新装/升级通过；缺 runtime 启动即诊断 | W11(1) wheel 干净 venv 安装+入口解析；doctor 预检 | ✅ 绿（CI Build Package 实开包校验） |
| G9 | release 测试含 integration/deployment | CI gate 聚合 13 必检（无 marker 排除假绿——十三审起既有） | ✅ 绿 |
| G10 | 缺 key/429/超时/未采集写 NOT_RUN 等，不计 PASS | 本轨道全部真实模型项标 NOT_RUN（每个 W 文档） | ✅ 纪律执行 |
| G11 | 真实 A/B 执行或不发性能声明 | **W10 未执行**（无 key）——不发布优于原生 Pi 的声明 | **NOT_RUN**（阻断性能声明，不阻断 SIM Alpha 范围声明） |
| G12 | 产物可追溯具体构建；秘密不进日志/产物 | build-stamp + js_stage 同源 staging；密钥全环境变量（本仓库 0 处落盘） | ✅ 绿 |

## SIM Alpha 范围冻结（§16.1，测评前）

- **核心**：L01、L02、L07、L09、L10；**强制行为 Gate**：L08；
  **拓展（必须运行并披露）**：L03–L06。
- 当前实测：L01/L02/L07/L10 物理层绿；L09 平台语义绿（W05）；
  L08 fake REAL 侧契约绿（W06）；**agent 层全部 NOT_RUN**
  （无 key——W10 待执行）。
- README 未宣传避障/视觉/推物/控制的达成（无对应承诺文本
  被新增）。

## 阻断与限制（发布前必须关闭或明确降级）

1. **真实模型验收全部 NOT_RUN**（ROSCLAW_KIMI_API_KEY 不在
   环境）——G1/G7/G11 与 agent 层 L03–L06/L09 依赖。W10 需
   带 key 执行后方可把级别主张升到 SIM Alpha；在此之前发布物
   只能称内部候选。
2. x86_64 认证矩阵：CI Build Package 在 x86_64 跑 wheel 校验；
   本机实测为 aarch64。
3. 发布动作边界：本工作只准备材料；**tag + 正式发布需维护者
   显式批准**（本文件即呈批件）。

## 回退（§16.5）

- 发布后 smoke 从 PyPI 全新下载核对 wheel hash；
- 严重问题：yank + 修复版本（不覆盖已发布文件）；
- 迁移失败保留原副本并停止；卸载脚本不碰用户产物/凭据。
