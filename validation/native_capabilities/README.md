# Native portable saved-capability replay

本套件由本轮 ROSClaw Native 编写，仅复核公开的、已完成的有限软件证据。范围为 `SOURCE_ONLY_PORTABLE_SAVED_CAPABILITY_EVIDENCE_NOT_PHYSICAL`。不会新跑模型、MoveIt、ROS、FK、物理 STEP、网络或硬件。它不是路径预演或动力学仿真，也不是机器人任务成功证明。

## 可移植接口与依赖

公开入口是 `replay(evidence_root)`，定义于 `replay_saved.py`。传入保存 bundle 的相对/绝对目录均可；复制完整 `protected_capability_evidence/` 到任意位置后，可用同一入口验证。无需原 workspace 路径、历史私密会话、注册数据库、模型权重、ROS 安装或原 C++ 编译器。bundle 中原路径只是历史注册关联字符串，不会打开那些路径。

运行依赖：Python >=3.10 标准库和**原样完整**的受保护 bundle。测试另需 pytest；选定源检查另需 Ruff、mypy。发布时应一并保留 bundle 的 manifest、七个 genuine Native 源、请求、数值输出、保存 receipts、注册快照及 fixturelib。不是允许替换任意模型证据的通用验证器；manifest SHA 和 genuine 源 SHA 明确钉住这一个公开证据版本。

可在有该 bundle 的独立 Python 环境导入：

```python
import importlib.util
from pathlib import Path

source = Path("validation/native_capabilities/replay_saved.py")
spec = importlib.util.spec_from_file_location("saved_replay", source)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.replay(Path("protected_capability_evidence"))
```

入口不写文件、不注册 Ref、不缓存 bundle 路径。每次读取 manifest、所有列举文件（SHA/size）、全部七源和保存数值证据。路径逃逸被拒绝。七源与历史注册快照的 hash、size、task、producer、role 关联逐项校验。fixturelib 字节先独立核对 SHA，然后在新 namespace 加载，避免 sys.path 或旧模块缓存充当证据。最终核对 manifest 的公开信任锚，连同输入/答案一起重新编造的 bundle 也不得通过。该信任锚来自公开 `prepared_packet_index_portable_v1.json` 中当前 manifest 的 SHA `034b1f8c6542af84925e9fb9b9c198869be1cf940755db167ce9030401904101`，不是固定 PASS。compact brief 中旧 SHA 与当前受保护 manifest 不一致：首次 RUN 因该锚不匹配失败，随后仅在 mutable runner 修正为公开 prepared index 的当前值，保护输入未改；首次格式检查失败也保留在 generated reports 中。

## 计算与证据归属

`fixturelib/moveit_case_oracle.py`、`fixturelib/vla_four_case_oracle.py` **由 operator 编写**，是有限生成输入的验收 oracle，不是 Native 机器人算法。Native 编写的是本轮 runner、测试、文档及 matrix；原七源保持受保护字节不变，不重新导入或执行其中的模型/ROS代码。

runner 实际调用 operator 数值/输入 oracle 重算：

- MoveIt10：请求逐例与保存接口 trace 相同；生成模型变量/索引、sphere 身份/位置/碰撞、baseline/replan 时序、路径端点/界限与 swept tool-sphere 间隙、contact frame 几何、未知变量/帧拒绝、Cartesian fraction/jumps 与越界不能报全路径。
- VLA8：四个结构化 case 的实际 solve 输入/输出/期望关联；两个 `[1,50,6]` chunk 的每个数值、raw 与 postprocessing 张量、归一化一致性、saved mean/std 后处理、first action 到第二次 preprocessing state 的数值因果关联、camera 张量与保存加载/forward 元数据。

这里重算保存数字，不调用原生成系统，也不再次证明 checkpoint 在当前环境被执行。旧模型执行的来源来自 hash-绑定的保存 observer/receipt。Native self-report 与 operator oracle 的角色明确分开；fixture trace 也不是独立硬件传感器。

全部 oracle 验证和信任关联成功后，返回 `PASS_REPLAYED_SAVED_CAPABILITY_EVIDENCE`、18 个独立 `{case_id, passed:true}`、七源实测 SHA、保存注册快照中的准确 Ref/hash/role。坏文件 hash/size、shape、role、case/input、源变化以及 malformed bundle 会抛 `AssertionError` 或 `ValueError`，不返回 PASS。返回值里的 `artifact_refs` 仅指**输入 bundle 的历史注册**，不可当成本轮 Task 的新登记。

## 测试与本轮验收

Native pytest 文件包含 11 个实质测试：有效18例/七源/准确保存Ref；完整搬迁；坏hash；更新 manifest 后的坏shape、role、case input、genuine source、profile数字、feedback数字；同一路径先有效后更改不缓存；57-row/51授权边界。修改只发生在 pytest 的临时 bundle 副本，原保护文件不变。重索引负例专门让语义验证实际运行，而非全部仅靠文件hash失败。

本轮只使用 `inputs/SOURCE_CONTRACT.json` 声明的公开 `check`/`run` helper。helper 的 stdout 为短JSON；完整命令、退出码、stdout/stderr、source hashes、replay/read proof、五个独立负控和实际 pytest counts 写入当前 `reports/operator_checks/`。`check` 对两个 Python 文件执行选定 Ruff check/format `--no-cache` 和 strict mypy `--no-incremental`；类型 cache 隔离在当前检查结果目录。这些检查不是项目全量兼容性声明。`run` 检查原址、搬迁副本、相同路径五个破坏副本，以及实际 pytest 至少六 PASS。成败以最新源字节的 helper receipt 为准，不由本 README 宣告固定成功。

交付时四个当前源文件均新登记：runner/test 为 `diagnostic_source`，README/matrix 为 `diagnostic_report`；六键 `delivery.json` 为 `progress_report`。这些登记不关闭 SOURCE_ONLY 任务或升级物理证据。若最新检查失败，最终 `SOURCE_FAILED`，无成功 credit。允许登记本轮 helper 生成的 `reports/operator_checks/` 诊断，禁止将输入历史 Ref 冒充当前新 Ref。

## 57-row matrix 与不可升级的边界

`evidence_matrix_57.json` 新写57个精确原 ID/name。`VERIFIED_BOUNDED_SCOPE` 的51项只继承公开 `CURRENT_ROOT_51_SCOPE_AUTHORITY.json` 的独立限定域授权；并非本轮重跑51项，更不是51项物理能力。只有 F38–F46 的18个已保存有限 case 在本轮重算。其他已接受项明确只有 authority 指针及原 history/dependencies 指针，不声称可用此 bundle 重跑。未接受的 F12/F19/F20/F33/X55/X56 为 BLOCKED 或 UNVERIFIED。物理资格、合法回球、合格交付视频均为0。

历史失败保留：原 operator-invalid、contract/UUID/budget failures、provider/whole failures、MoveIt LINK_BLOCKED 与 dirty-transform abort wholeFAIL、X56 已记录物理失败都不回填为成功。MoveIt header 是 operator interface carrier 经已完成 Native freshness repair，不是新作者运动算法。F42 越界 partial fraction 是正确拒绝结果，不能称完整路径。

VLA raw/model 与 so100 saved profile 的**物理单位 UNKNOWN**；无 physical calibration、无 G1/RH56 mapping。F44 是 synthetic action-as-state benchmark，不是环境反馈、闭环物理控制、抓取或任务完成。F45 是 deterministic structured-label grounding，不是 neural VLN/视觉验证/导航；F46 是有限离散 reservation，不是实体多机器人协调。无任意场景保证、部署安全、控制稳定性、接触/力/抓取成绩。本套件不包含私密 raw sessions、wire、auth、thinking 内容。
