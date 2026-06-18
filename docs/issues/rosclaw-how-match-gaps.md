# rosclaw-how 匹配缺口报告

> 整理自 ROSClaw 集成测试 `tests/test_know_how_integration_eval.py`
> 可直接转发给 rosclaw-how 团队

## 问题概述

在端到端测试 `test_how_client_end_to_end_recovery_hints` 中，所有查询都通过 `rosclaw-how` 的 `/wiki/v1/prompt/build` 接口请求恢复提示。大多数 curated cluster 能够正常匹配并返回 hint，但以下两个规则目前返回 `None`（ABSTAIN），导致测试不得不标为 `xfail`。

## 受影响规则

| 查询关键词 | 期望规则 ID | 所属 curated cluster | 当前结果 |
|---|---|---|---|
| `entropy collapse ppo policy value loss degenerate episodes kl exploration` | `ppo_entropy_collapse_guard` | `rl-training-stability` | `hint is None` |
| `Gradient magnitude explodes during backprop causing weight overflow` | `gradient_clipping` | `rl-training-stability` | `hint is None` |

## 复现步骤

1. 启动 rosclaw-how 服务（默认 `http://127.0.0.1:47820`）
2. 使用 `rosclaw.how.client.HowClient` 调用：

```python
import asyncio
from rosclaw.how.client import HowClient

async def reproduce():
    client = HowClient("http://127.0.0.1:47820")
    await client.initialize()

    queries = [
        "entropy collapse ppo policy value loss degenerate episodes kl exploration",
        "Gradient magnitude explodes during backprop causing weight overflow",
    ]
    for q in queries:
        hint = await client.generate_recovery_hint(
            q,
            {"episode_id": "repro", "request_id": "repro"},
            previous_scores=[0.5, 0.5, 0.5, 0.5],
            current_iteration=10,
        )
        print(q, "->", hint)

asyncio.run(reproduce())
```

## 期望行为

两个查询都应返回非空 hint，且满足：

```python
hint["source"] == "how_catalyst"
hint["rule_id"] == expected_rule_id
```

## 实际行为

两个查询均返回 `None`。

## 环境信息

- `rosclaw-how` 版本：`>=1.0.0`
- `rosclaw-know` 版本：`>=1.0.1`
- 服务端点：`http://127.0.0.1:47820`
- 测试文件：`tests/test_know_how_integration_eval.py`
- 测试标记：`@pytest.mark.xfail(reason="rosclaw-how bridge_index is missing live matches for ppo_entropy_collapse_guard and gradient_clipping", strict=False)`

## 已知正常工作的对照 case

以下查询在同样环境下可以正常返回 hint，说明服务端整体可用，问题集中在上述两个规则的匹配链路：

| 查询 | 规则 ID | 策略 | 结果 |
|---|---|---|---|
| `Actuator torque saturation at 237 N·m during PIDTuning` | `SAFETY` | `SAFETY` | ✅ pass |
| `NaN in weights after optimizer step` | `Numerical_Instability` | `SAFETY` | ✅ pass |
| `battery capacity fade lithium plating li-ion fast charging cc-cv current taper c rate 4c thermal` | `multi_stage_cc_cv_fast_charging` | `CATALYST` | ✅ pass |
| `crypto throughput bottleneck aes aes-ni aes128 throughput mb/s cipher encryption decryption simd avx sse` | `simd_aes_ni_hardware_crypto` | `CATALYST` | ✅ pass |
| `combinatorial local optimum job-shop scheduling makespan tabu search simulated annealing genetic algorithm metaheuristic` | `metaheuristic_combinatorial_escape` | `CATALYST` | ✅ pass |
| `image motion blur rolling shutter imu deblur drone quadrotor exposure` | `motion_blur_imu_aided_deblur` | `CATALYST` | ✅ pass |

## 怀疑方向（供 how 团队排查）

1. `bridge_index.json` 中 `ppo_entropy_collapse_guard` / `gradient_clipping` 的 topic fingerprint 与查询语义不匹配。
2. `topic_group` filter 将这两个查询错误地路由到了不含对应 cluster 的组。
3. cluster 相似度阈值过高，导致 `rl-training-stability` cluster 被过滤掉。
4. `state_router` 对这两个 query 的意图识别错误，未进入 curated cluster 分支。

## 验收标准

当这两个查询能够返回正确 hint 时，ROSClaw 侧会：

1. 将 `HOW_TEST_CASES` 中对应的两个 case 保留为必须通过。
2. 移除 `test_how_client_end_to_end_recovery_hints` 上的 `@pytest.mark.xfail`。
3. 删除 `docs/issues/rosclaw-how-match-gaps.md` 或将其标记为已解决。

## 附件

- 测试代码：`tests/test_know_how_integration_eval.py`
- 端到端 client：`src/rosclaw/how/client.py`
