# Frozen clock diagnostic RPC v1 — source-only

## 状态和边界

本轮仅窄修三个 source/test/doc 文件；未执行源码、pure tests、SDK、DDS、容器或 qualification。历史：原作者 62962 已 CLOSED0；Root 报告原版 pure 20 methods / 18 subtests / 78 reports PASS，仅为原版纯软件事实。原版 terminal 在 fsync 前采 row timestamp，fsync 后仍检查该旧值，无法证明实际持久化完成在 deadline 内；Root 因此阻止 actualDDS。此次不改写历史结果，也不把旧 PASS 当新版本通过。没有 actualDDS PASS，也没有机器人能力证明。纯 fixture 即使未来通过，也只是 oracle 的软件检查，不是 ROS clock 行为证据。

未来唯一目标环境：image `sha256:81b41b619cc44a3c4915cd7bf42a7faf5219ddf7d84e2506e890a2457adaaaf5`，`/usr/bin/python3.12`，固定 Jazzy API。不得安装、pull、host fallback 或外部 ROS 辅助进程。真实运行必须另经 Root 审核与授权，使用唯一 exact wrapper、原始 model-clock admission、cleanup reserve、实际进程 terminal status/output、真正 Refs 和独立 owned CID absence。本文不提供本轮执行许可。

保留 Atomic12 的历史结果及 outer FAIL；不复用其 PASS 算法。Credit 最多是一组 generated-DDS 软件组件，不能外推 ROS 服务取消、action 取消、硬件、传感器、QoS 保障、全部 CAP80 或机器人任务。

## 入口与调度

`run_frozen_clock_rpc_scenario(case_id, nonce, domain_id, evidence_path)` 四个参数必填。CLI 仅 `--case`、`--nonce`、`--domain-id`、`--evidence`（以及 argparse 的标准 help）。case/nonce 为 1–128 个 ASCII 字母、数字、下划线或连字符；domain 为非 bool 整数 0..232。每次真实运行的身份与 domain 独有由外层 admission 保证；源码不能证明 domain 在整个宿主机没有其他参与者。

模块导入仅标准库。唯一 lazy SDK 边界为 `_runtime`，只能由显式入口调用；pure validator、fixture、测试收集不创建 Context 或线程，不导入 SDK。测试以普通 unittest 和内存 BytesIO/mock 构造证据，不调用 runtime。

一个显式 Context、两个 owned Nodes、一个显式 `MultiThreadedExecutor(num_threads=2)`；主线程只调用 `spin_once` 和观察逻辑状态，不作为第三个 executor worker。服务使用 `ReentrantCallbackGroup`：LATE callback 在一个 worker 中按 monotonic 有界 sleep，第二个 worker 仍能处理 RECOVERY 与 client response。不是 ROS timer 延迟；不依赖冻结的默认 timer。没有自行创建 timer/guard、第二 Context 或外部进程。

两个 Node 显式 BOOL `use_sim_time=True`，禁用 rosout、logger service、内建 parameter services，参数 override 关闭 type-description service。节点默认参数事件 publisher 与 `/clock` subscriptions 如实列入 inventory，不把它们误报为不存在。没有 `/clock` publisher。每次 CLOCK_SAMPLE 实读两个 `get_clock().now().nanoseconds`、`ros_time_is_active` 和 SDK `get_parameter('use_sim_time')` 的类型/值。初始、每次提交前、LATE 等待中约每 100ms 和最终 readback 采样；所有采样必须为 active 且 0ns。离散采样不声称连续物理测量。

只创建一个自定义 `GetParameters` diagnostic 服务和一个 client。服务名 `/frozen_rpc_{case}_{nonce}`；这是明确的 typed diagnostic RPC，不是 SDK 内建参数服务延期语义，不修改参数。

三个且仅三个 `call_async`：FAST、LATE、RECOVERY，request `names` 分别为单元素 `[nonce + '/' + request_id]`。SDK callback 实读 names，真实 response 使用 `ParameterValue(type=4, string_value=实际请求 names[0])`。投影只保留实际 STRING discriminator 和 string_value，不投影无关的默认 union 字段。future 全部保留至 cleanup；`str(id(future))` 是进程内局部身份，不是 RMW sequence 或 GID。

## 时序、持久化与失败

全部时间为同一进程 `time.monotonic_ns()`，不使用日历或 ROS time 计 deadline。入口第一次取钟后冻结 `work_end=start+60s`、`whole_end=start+90s`；阶段间不重置。setup、SDK import、discovery 和正常事件均须在原 work cutoff 内；cleanup/readback/final flush/fsync/close 及返回受 whole cutoff。cleanup 最多预留 30s。Root 运行前冻结这些源码常量；运行中不放宽期限。外层 240s/270s 上限不能替代内部 cutoff 或原 model-clock admission。不可抢占的 SDK destroy、fsync、解释器退出仍需外层监督；超时绝不凭 raw 中的 PASS 单独判定成功。

- FAST：submit 前取 monotonic；deadline 精确为 submit+2s。必须服务收到、ready、client 实际 future result、SUCCESS；观察和 terminal 均不晚于 deadline。
- LATE：deadline=submit+0.5s。服务收到的事件取钟后至少再等 2s 才构造 ready response。主线程在 deadline 后、submit+1.5s 前观察 pending future，写 terminal TIMED_OUT 并 flush/fsync，然后才允许提交 RECOVERY。逻辑超时不调用 cancel、不移除 pending request、不伪称服务器取消。
- RECOVERY：同一组 owned 对象上提交，deadline=submit+1s；真实 client response 和 SUCCESS 必须在 LATE 的 CLIENT_RESPONSE_OBSERVED 之前。两 pending futures 同时保留。随后继续 spin 直到 LATE client 实际 result 被观察，最迟 LATE submit+5s。晚到 payload 只能进入 LATE_RESPONSE_QUARANTINED，不复活 TIMED_OUT。

服务 ready 只表示 callback 已构造返回值，不表示 DDS 已发送或 client 已收到。只有对保留 SDK future 的 `done()/result()` 读取才记录 CLIENT_RESPONSE_OBSERVED。success 时间是本地观察时间，不伪造 SDK 底层完成时间。

所有事件使用可重入锁串行写入，避免 worker/main 顺序混淆；提交时持锁直到 future 与提交事件登记。每行 flush，terminal 与 FINAL 另 fsync。terminal 原行写入/flush 后先保留到内存证据与字节计数，再调用 `terminal_sync(stream, time.monotonic_ns, os.fsync)`：原 monotonic 在实际 fsync 调用前后分别实采 `before_ns`、`completed_ns`。后者不是事件时间副本，也不宣称精确内核落盘瞬间，而是 fsync 成功返回后的保守完成上界。仍在同一锁内紧接写入 `TERMINAL_SYNC_COMPLETED`，关联原 terminal seq/request/future/owners，之后立即执行与 pure oracle 共用的完成检查，检查通过前不进入下一阶段。

要求 `terminal.mono_ns <= before_ns <= completed_ns <= receipt.mono_ns`，均为非 bool 整数；clock 标识必须为 `time.monotonic_ns`。实际 runtime 的采样函数与 BOOT、submit、event 是同一个 monotonic 来源，不允许 ROS/日历时钟替代。FAST 完成上界 ≤原 submit+2s；RECOVERY ≤原 submit+1s；LATE 原事件不得早于 submit+0.5s，实际完成上界 ≤原 submit+1.5s。完成还须早于原 work cutoff，原行时间门仍保留。receipt 自身的写入时间不是 terminal 完成时间，不重写原行，不重启 submit/deadline。receipt 行每行 flush，并由最后 FINAL 的真实 fsync 覆盖；不对 receipt 再递归要求 deadline 内持久化。

实际 fsync 异常或完成取钟异常时没有完成 receipt，保留已经写出的 terminal；非法/倒退/失配/过迟的可编码采样先留下 receipt，再报错，不能推进或返回成功。terminal 表示逻辑终态而非持久化成功凭据；即使逻辑 SUCCESS 已记录，缺 receipt、非法 receipt 或 ERROR 都不能取得 PASS。失败走原 owned cleanup、ERROR/FAIL 路径；证据写入/关闭失败可直接传播，不能保证损坏存储还能记录 ERROR，不覆盖或擦除既有原始证据。

raw 以 UTF-8 exclusive `x` 创建，不覆盖旧证据；最多 256 行/1MiB。没有错误且候选完整 transcript 经纯 oracle 通过、实际磁盘 prefix readback 一致才允许写 PASS FINAL。FINAL 最后写，flush/fsync/close 后才返回；返回前再次检查 whole cutoff。外部验收还须核对 exit status 和原始 cutoff，不能只信 FINAL 或 clock 标签。纯 oracle 只验记录自洽，来源真实性仍须真实执行 provenance。

失败记录 ERROR，保留已知 terminal，未达 terminal 的阶段为 UNKNOWN；不重试、不重建节点、不改期限。先有界 shutdown executor，再释放 futures（仅失败清理时可本地 cancel pending），逆序 destroy owned nodes（递归释放 node entities），最后 shutdown Context。shutdown 未完成时不在活 callback 下销毁 nodes；报告 cleanup 不完整，需外层 owned-container 清理。每项 cleanup 异常独立记录，不吞异常。没有记录未观测的自动 publish 次数，更不会把它写为 0。

## 冻结 JSONL schema

每行闭集公共键：`schema,case,nonce,domain,seq,mono_ns,event,data`。schema 演进为 `ROSClaw.FrozenClockRPC.v1.1`（旧版无完成证据，不能用于新验收）；case/nonce/domain 恒定；seq 是从 0 连续的非 bool 整数；mono_ns 为非负非 bool 整数且不倒退。JSON object 禁重复 key，拒绝非有限数（包括指数溢出）和未知事件/字段。

以下为 data 的精确键集；`L` 代表展开 `request_id,future_id,owners` 三键，不是额外嵌套字段。

| event | data keys |
|---|---|
| BOOT | start_ns, work_end_ns, whole_end_ns |
| ENTITY_INVENTORY | owners, service, entities |
| CLOCK_SAMPLE | owners, ros_ns, active, use_sim_time |
| REQUEST_SUBMITTED | L, submit_ns, deadline_ns, request |
| SERVER_REQUEST_RECEIVED | L, request |
| SERVER_RESPONSE_READY | L, response |
| CLIENT_RESPONSE_OBSERVED | L, response |
| REQUEST_TERMINAL | L, status, future_pending |
| TERMINAL_SYNC_COMPLETED | L, terminal_seq, clock, before_ns, completed_ns |
| LATE_RESPONSE_QUARANTINED | L, response |
| ERROR | class, message |
| CLEANUP | owners, executor_shutdown, released_futures, destroyed_nodes, context_shutdown, entities_released |
| FINAL | status, terminals, errors |

Nested shapes:

- `owners`: exact keys `context,server,client_node,executor,client`，值为不同的正十进制 local id 字符串；全过程一致。
- `entities`: exact keys `server,client_node`；每节点 exact categories `publishers,subscriptions,services,clients,timers,guards`。每类 list 的元素为 `{id:十进制字符串,name:实际解析名称字符串}`；无名称的 timer/guard 用空串。总数严格 2 publishers、2 subscriptions、1 service、1 client、0 timers、0 guards；服务/client 必须匹配 diagnostic name，两个订阅必须 `/clock`。这是 node logical inventory，不是 DDS participant endpoint 总数；SDK executor 内部设施随 executor shutdown 释放。
- `request`: exact `{names:[nonce/request_id]}`；三个 request_id 枚举 `FAST,LATE,RECOVERY`。
- `response`: exact `{values:[{type:4,string_value:nonce/request_id}]}`；bool 和 float 不能冒充整数 discriminator。
- `ros_ns`: 两个整数 0；`active`: 两个 bool true；`use_sim_time`: 两个 `{type:1,value:true}`。顺序固定 server、client_node。
- `status`: terminal 只许 FAST/RECOVERY SUCCESS、LATE TIMED_OUT；FINAL 为 PASS 或 FAIL。
- `terminals`: exact keys FAST、LATE、RECOVERY；值为 SUCCESS/TIMED_OUT/UNKNOWN，必须与已记录 terminal 一致。
- `future_pending`: bool，LATE timeout 必须 true，SUCCESS 必须 false。
- `TERMINAL_SYNC_COMPLETED`：紧接对应 REQUEST_TERMINAL 的唯一下一行；`terminal_seq` 为非 bool 整数且指向原行，L 三键完全相同；clock 固定 `time.monotonic_ns`，before_ns/completed_ns 遵守上述真实完成上界和原始窗口。三条都必须存在且有效；FAST/LATE 的完成检查分别是 LATE/RECOVERY submit 的前置条件，RECOVERY 完成检查是 LATE client observation 的前置条件。失败 prefix 可缺 receipt 但永不 PASS；已记录非法采样仍会被 validator 拒绝，不因 require_complete=False 忽略。
- cleanup flags 是 bool；released_futures 是恰好所有已提交 future id 的无重复 list；destroyed_nodes 是真实成功销毁的 node ids；PASS 要求二者均销毁、executor/context shutdown 和 entities_released 全 true。
- `errors`: 与 ERROR 事件逐项一致的 `{class,message}` list；字符串各最多 2048 字符。PASS 必须为空。

## 独立 oracle 与纯测试

`read_jsonl(path)` 有界读取 UTF-8 JSONL，要求末尾换行、无空行、无重复 key、无 nonfinite；`validate_transcript(rows)` 重放闭集 schema、monotonic、原始 cutoff、请求/future/owners 关联、三条真实 client response 投影、deadline、超时窗口、服务延迟、隔离顺序、终态不可变、最终 clocks 与 cleanup。只有完整 PASS 返回 True。`require_complete=False` 只检查结构/因果有效的失败 transcript 或 prefix，永远 False，拒绝 PASS；初始化前失败证据可能没有完整 inventory，不能据此取得完整合格证书。

纯测试包含闭合 synthetic fixture、非冻结/未激活 clock、参数类型、超时提前/过迟、pending 状态、复活、错 nonce/future、typed discriminator、server-ready 代替 client evidence、过短服务延迟、先 recovery 后 timeout、T3 串线、同 context/client 被替换、原始 cutoff/deadline 重置、cleanup/最终 clock 遗漏、重复 FINAL/seq、未知键/辅助 entity、nonfinite/bool numeric、JSON duplicate keys/UTF-8/边界，以及入口参数校验不会进入 SDK。未在本轮运行这些测试。

保留原 20 个 test 方法及原接受/拒绝目标；fixture 增加独立手写 completion 样本，原 timeout 窗口负控同步平移完成样本而不改变窗口。新增纯测试逐条覆盖 FAST/LATE/RECOVERY 原事件时间合法、真实完成上界恰等 deadline 可接受、超过 1ns 必拒；移动后续工作仅为保持 synthetic 因果和 monotonic，目标原 submit/terminal 均不动。另覆盖 receipt 缺失/重复/错序/错关联/未知成功字段、旧 schema、完成钟类型/倒退/超过 receipt/时钟域失配，以及 injected sync 前后真实调用次序、fsync 异常和初始/完成取钟异常传播。helper producer 与完成 checker 的纯慢完成控制不调用 SDK 或真实 fsync；runtime 始终使用真实 os.fsync。本轮全部仅写入，未执行或声称通过，仍需 Root 独立审核与另行授权的纯测试。

最终真实验收需同时审阅 raw、独立 oracle、实际进程状态、原始 clock admission、cleanup 与 owned CID absence。纯 oracle 只能检查记录自洽，不能自己认证日志来自真实 DDS；真实执行与 wrapper provenance 必须另行核验。
