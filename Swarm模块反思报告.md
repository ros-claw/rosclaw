# Swarm 模块深入反思报告

**基于《ROSClaw v1.0 深入验收指南》的三个核心问题反思**

---

## 一、多 Robot 调度公平性

### 当前实现

`SwarmCoordinator.allocate_task()` 使用**单一维度拍卖**：

```python
winner = min(bids, key=lambda b: b.cost)
```

cost = 1.0 + distance_penalty。仅考虑空间距离，不考虑：
- 该 agent 已完成多少任务（历史负载）
- 该 agent 当前 battery / energy 状态
- 该 agent 的 skill 熟练度 / success_rate
- 任务紧急度与 agent 可用时间的匹配

### 公平性缺口

| 场景 | 当前行为 | 理想行为 |
|------|----------|----------|
| g1 已完成 10 个任务，ur5e 空闲 | g1 仍然可能中标（离目标更近） | 轮询或加权负载均衡 |
| 两个 agent 能力相同、距离相同 | 先注册者中标（list 遍历顺序） | 随机或历史负载加权 |
| 高价值任务 vs 低价值任务 | 同一拍卖逻辑 | 高价值任务优先分配给 success_rate 更高的 agent |

### 建议改进

```python
# 多因子评分 = success_rate * 0.3 + proximity_bonus * 0.3 - load_penalty * 0.2 - battery_penalty * 0.2
def compute_fair_bid(agent, task):
    base = 1.0
    proximity = 1.0 / (1 + distance(agent.position, task.target))
    load_penalty = agent.completed_tasks * 0.1
    battery_factor = agent.battery_level / 100.0
    success_rate = agent.skill_success_rate.get(task.skill, 0.5)
    return base + proximity * success_rate - load_penalty + battery_factor
```

---

## 二、资源竞争

### 当前实现

1. **SafetyZone**：仅球形包络碰撞检测，无共享资源语义
2. **ForceStateShare**：检测负载不平衡但不触发重分配
3. **DDSGroupManager**：管理 topic namespace，无网络带宽竞争控制

### 竞争场景缺口

| 竞争类型 | 当前状态 | 风险等级 |
|----------|----------|----------|
| 同一物体被两个 agent 同时抓取 | ❌ 无检测 | 🔴 P0 |
| 两个 agent 同时进入狭窄通道 | ⚠️ SafetyZone 可检测球形重叠 | 🟡 P1 |
| 共享计算资源（GPU/推理） | ❌ 无调度 | 🟡 P1 |
| 共享通信带宽（DDS） | ❌ 无 QoS | 🟢 P2 |
| 电池充电站竞争 | ❌ 无建模 | 🟢 P2 |

### 关键缺口：互斥锁（Mutex）

协作搬运中，table_001 是**共享资源**。当前代码：

```python
# agent A 分配到了 grasp table_001
# agent B 也分配到了 grasp table_001 （如果分解出两个 subtask）
# → 无冲突检测！
```

建议增加 **ResourceLockManager**：

```python
class ResourceLockManager:
    def __init__(self):
        self._locks: dict[str, str] = {}  # resource_id -> agent_id
    
    def acquire(self, resource_id: str, agent_id: str) -> bool:
        if resource_id in self._locks:
            return False
        self._locks[resource_id] = agent_id
        return True
    
    def release(self, resource_id: str, agent_id: str) -> bool:
        if self._locks.get(resource_id) == agent_id:
            del self._locks[resource_id]
            return True
        return False
```

并在 `allocate_task` 前检查资源是否已被锁定。

---

## 三、与 Runtime 的协作接口

### 当前架构问题

验收指南明确要求：

> "模块之间不能互相乱调，sandbox 应只发布事件，Practice、Memory、Dashboard、Darwin、How 等模块订阅事件，否则模块会互相缠死。"

**当前 Swarm 的实际情况**：

```
SwarmCoordinator (direct method calls)
  ├─ register_agent()          ← direct call
  ├─ decompose_task()          ← direct call
  ├─ allocate_task()           ← direct call + 内部 event_bus.publish()
  └─ propose_state()           ← direct call + 内部 event_bus.publish()

SwarmRuntimeManager (EventBus only)
  ├─ _on_register_request()    ← EventBus subscriber
  ├─ _on_allocate_request()    ← EventBus subscriber
  └─ _on_status_request()      ← EventBus subscriber
```

**问题**：`SwarmCoordinator` 和 `SwarmRuntimeManager` 是**两个独立类**，没有统一接口。外部模块不知道该调用谁。

### 理想中的 Swarm-Runtime 协作接口

```
User / Agent Runtime
  ↓ Event: "swarm.task.requested"
SwarmRuntimeManager (统一入口)
  ↓ 调用 SwarmCoordinator
Task Decomposition + Auction
  ↓ Event: "swarm.task_allocated"
Sandbox (安全检查)
  ↓ Event: "sandbox.action.allowed"
Runtime (执行)
  ↓ Event: "skill.execution.start / .complete"
Practice (记录)
  ↓ Event: "praxis.completed"
Memory (沉淀)
  ↓ Event: "memory.write.completed"
Dashboard (展示)
```

### 具体改进建议

#### 1. 统一 Swarm Facade

```python
class SwarmRuntimeManager(LifecycleMixin):
    def __init__(self, event_bus: EventBus):
        self._coordinator = SwarmCoordinator(event_bus=event_bus)
        self._consensus = {}  # key -> RaftLikeConsensus
        self._resource_locks = ResourceLockManager()
        
    def submit_task(self, task: dict) -> str:
        """统一任务提交入口。"""
        # 1. 分解
        subtasks = self._coordinator.decompose_task(task)
        # 2. 资源锁检查
        for st in subtasks:
            obj = st.get("object")
            if obj and not self._resource_locks.acquire(obj, "pending"):
                raise ResourceConflictError(f"Object {obj} already locked")
        # 3. 拍卖
        allocation = self._coordinator.allocate_task(task)
        # 4. 发布事件
        self.event_bus.publish(Event(
            topic="swarm.task.allocated",
            payload={...},
        ))
        return allocation.task_id
```

#### 2. EventBus 必须成为唯一通道

当前 `SwarmCoordinator.allocate_task()` 内部直接修改 `self._agents[agent_id]["status"] = "busy"`，然后**顺便**发个事件。这违反了验收指南的架构原则。

改进：所有状态变更必须通过 EventBus → StateMachine → 订阅者回写。

#### 3. Swarm 必须接入完整闭环

| 闭环环节 | 当前状态 | 缺口 |
|----------|----------|------|
| Agent Runtime | ⚠️ CLI 有，MCP 未验证 | 需验证 Claude Code 能调用 |
| Provider Router | ❌ 未接入 | Swarm 不知道 Provider  latency / health |
| Sandbox | ❌ 未接入 | 协作任务未做 safety check |
| Practice | ⚠️ Demo 中有 EpisodeRecorder | 但无 `rosclaw practice list` 命令 |
| Memory | ❌ 未接入 | Swarm 任务未写入 Memory |
| How | ❌ 未接入 | 任务失败无 recovery hint |
| Dashboard | ❌ 未接入 | 无 swarm trace 展示 |

---

## 四、按验收指南的评分修正

| 维度 | 主管初评 | 反思后修正 | 原因 |
|------|----------|------------|------|
| 代码完成度 | 60% | 60% | 代码写了，但架构未统一 |
| 测试通过 | ✅ | ✅ | 74 tests pass |
| 文档 | ⚠️ | ⚠️ | 有 docstring，无用户指南 |
| **用户闭环** | ❌ | ❌ | **关键：未接入 Runtime → Provider → Sandbox → Practice → Memory → How → Dashboard** |
| **评分** | 5/10 | **4/10** | 距验收指南 L0-L5 要求差距明显 |

---

## 五、P0 修复清单（阻塞发布）

1. **统一 SwarmRuntimeManager 为唯一入口** — 合并 Coordinator + Manager
2. **ResourceLockManager** — 防止共享资源冲突
3. **多因子公平拍卖** — success_rate + load + battery
4. **Swarm → Sandbox 事件链路** — 协作任务必须经过 safety check
5. **Swarm → Practice 自动记录** — 每次 task submit 生成 episode
6. **Swarm → Memory 写入** — 任务结果沉淀到 memory

**预计工时：16h**

---

**结论：Swarm 模块目前是「有代码、能测试、未闭环」。距离 v1.0 GA 需要完成与 Runtime / Sandbox / Practice / Memory / How / Dashboard 的完整事件驱动集成。**
