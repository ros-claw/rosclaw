"""System Identification / Digital Twin（MH17，0916 优化 §二十五-§二十六）。

算法核心复用 **MuJoCo 官方 sysid 工具箱**（nonlinear least squares +
box bounds + batched rollout；`mujoco.sysid._src`），ROSClaw 侧提供：
内容寻址数据集（record_dataset）、契约校验（SysIDSpec）、候选模型
经 patch 血缘派生（不另起炉灶）、**holdout 独立复算**（§26.3：
绝不 train residual 降了就升级 twin）、SysIDReceipt 落库。

观测通道 = 状态信号（qpos/qvel，MjStateQPos/MjStateQVel）——
真实机器人日志同款形态（关节空间时间序列）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: v1 支持的参数类型 → (目标对象类型, patch field)。识别阶段走
#: MjSpec modifier；候选模型落地走同一套 patch 白名单（血缘+校验）。
_PARAM_TYPES: dict[str, tuple[str, str]] = {
    "joint_damping": ("joint", "damping"),
    "geom_friction": ("geom", "friction"),
    "geom_mass": ("geom", "mass"),
    "actuator_kp": ("actuator", "kp"),
}

#: holdout 改进门槛（<5% 不算升级——数值噪声不是 twin 收益）。
_HOLDOUT_MIN_IMPROVEMENT = 0.05


def validate_sysid_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """契约校验（fail-closed）：schema/参数非空/类型白名单/边界倒挂。"""
    if spec.get("schema_version") != "rosclaw.sim.sysid_spec.v1":
        raise ValueError(f"SYSID_SPEC_INVALID: schema_version={spec.get('schema_version')!r}")
    parameters = spec.get("parameters") or []
    if not parameters:
        raise ValueError("SYSID_SPEC_INVALID: parameters 为空（无待识别参数）")
    if not spec.get("train_sequences") or not spec.get("holdout_sequences"):
        raise ValueError("SYSID_SPEC_INVALID: train/holdout 序列划分必填（§26.3）")
    for param in parameters:
        ptype = param.get("type")
        if ptype not in _PARAM_TYPES:
            raise ValueError(f"SYSID_PARAM_UNSUPPORTED: {ptype!r}（支持 {sorted(_PARAM_TYPES)}）")
        obj_type, _ = _PARAM_TYPES[ptype]
        if not param.get(obj_type):
            raise ValueError(f"SYSID_SPEC_INVALID: {ptype} 缺目标名（{obj_type}）")
        if not float(param.get("min", 0)) < float(param.get("max", 0)):
            raise ValueError(f"SYSID_SPEC_INVALID: {ptype} bounds 倒挂或为空")
    return parameters


def _param_name(param: dict[str, Any]) -> str:
    """参数名 = <目标名>_<字段>（hinge_damping 同款官方命名）。"""
    obj_type, field = _PARAM_TYPES[param["type"]]
    return f"{param[obj_type]}_{field}"


def _make_modifier(spec_obj: Any, param: dict[str, Any]):  # noqa: ANN001, ANN202
    """sysid Parameter.modifier：把候选值写进 MjSpec（与 patch
    白名单语义一致）。"""
    obj_type, field = _PARAM_TYPES[param["type"]]
    name = param[obj_type]

    def modifier(mj_spec, p) -> None:  # noqa: ANN001
        value = float(p.value[0])
        if obj_type == "joint":
            target = next(j for j in mj_spec.joints if j.name == name)
            target.damping[0] = value
        elif obj_type == "geom":
            target = next(g for g in mj_spec.geoms if g.name == name)
            if field == "friction":
                target.friction[0] = value
            else:
                target.mass = value
        elif obj_type == "actuator":
            target = next(a for a in mj_spec.actuators if a.name == name)
            target.gainprm[0] = value
            if len(target.biasprm) >= 2:
                target.biasprm[1] = -value

    return modifier


def _trace_to_sequence(backend, seq_record: dict[str, Any], model: Any):  # noqa: ANN001
    """数据集序列 → sysid 的 (initial_state, control, sensordata)
    三元组（观测 = qpos/qvel 状态信号）。

    initial_state 取 v2 快照 blob 的 FULLPHYSICS 前缀（精确初值，
    不从 trace 首行近似——首行是第一步之后的态）。"""
    import mujoco
    from mujoco.sysid._src import timeseries

    trace = backend.store.get(seq_record["trace_ref"])
    states = trace.get("states") or []
    if not states:
        raise ValueError(f"SYSID_DATASET_EMPTY: {seq_record['trace_ref']}")
    times = np.array([float(s["t"]) for s in states])
    qpos = np.array([[float(v) for v in s["qpos"]] for s in states])
    qvel = np.array([[float(v) for v in s["qvel"]] for s in states])
    ctrl = np.array([[float(v) for v in s.get("ctrl", [])] for s in states])
    if ctrl.ndim == 1 or ctrl.shape[1] == 0:
        ctrl = np.zeros((len(states), model.nu))

    qpos_map, qvel_map, _act_map, ctrl_map = timeseries.TimeSeries.compute_all_state_mappings(model)
    state_map = qpos_map | qvel_map

    fullphysics_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state_ref = seq_record.get("initial_state_ref")
    if initial_state_ref:
        meta = backend.store.get(initial_state_ref)
        blob = backend.store.get(meta["state_vector_ref"])
        vector = np.frombuffer(blob, dtype=np.float64)
        initial = vector[:fullphysics_size].copy()
    else:
        initial = np.zeros(fullphysics_size)
        initial[1 : 1 + model.nq] = qpos[0]
        initial[1 + model.nq : 1 + model.nq + model.nv] = qvel[0]
    control_ts = timeseries.TimeSeries(times=times, data=ctrl, signal_mapping=ctrl_map)
    sensordata_ts = timeseries.TimeSeries(
        times=times, data=np.concatenate([qpos, qvel], axis=1), signal_mapping=state_map
    )
    observations = [(name, sig) for name, (sig, _idx) in state_map.items()]
    return initial, control_ts, sensordata_ts, observations


def run_sysid(backend, spec: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    """执行 SysID：参数识别 → 候选模型（patch 血缘）→ holdout
    独立复算 → SysIDReceipt 落库。"""
    import mujoco
    from mujoco.sysid._src import optimize, parameter, residual

    parameters = validate_sysid_spec(spec)
    base_ref = spec["base_model_ref"]
    manifest = backend._manifest(base_ref)
    base_spec = backend._spec_from_manifest(manifest)
    base_model = base_spec.compile()

    dataset = backend.store.get(spec["dataset_ref"])
    sequences = dataset.get("sequences") or []
    dataset_digest = spec["dataset_ref"]  # 内容寻址 ref 即数据摘要

    train_idx = [int(i) for i in spec["train_sequences"]]
    holdout_idx = [int(i) for i in spec["holdout_sequences"]]
    for idx in train_idx + holdout_idx:
        if idx < 0 or idx >= len(sequences):
            raise ValueError(f"SYSID_SPEC_INVALID: 序列索引越界 {idx}（共 {len(sequences)}）")

    def build_sequences(indices: list[int]):
        from mujoco.sysid._src import residual as residual_mod

        initials, controls, sensordatas, names = [], [], [], []
        observations = None
        for idx in indices:
            initial, control_ts, sensordata_ts, obs = _trace_to_sequence(
                backend, sequences[idx], base_model
            )
            initials.append(initial)
            controls.append(control_ts)
            sensordatas.append(sensordata_ts)
            names.append(f"seq{idx}")
            observations = obs
        seqs = residual_mod.ModelSequences(
            name="sysid",
            spec=base_spec,
            sequence_name=names,
            initial_state=initials,
            control=controls,
            sensordata=sensordatas,
            allow_missing_sensors=True,
        )
        return seqs, observations

    # 参数集（box bounds 来自 spec；nominal = 基座模型现值）。
    params = parameter.ParameterDict()
    params_before: dict[str, float] = {}
    for param in parameters:
        nominal = _current_value(base_spec, param)
        pname = _param_name(param)
        params_before[pname] = nominal
        params.add(
            parameter.Parameter(
                name=pname,
                nominal=np.array([nominal]),
                min_value=np.array([float(param["min"])]),
                max_value=np.array([float(param["max"])]),
                modifier=_make_modifier(base_spec, param),
            )
        )

    train_seqs, observations = build_sequences(train_idx)
    residual_fn = residual.build_residual_fn(
        models_sequences=train_seqs, enabled_observations=observations
    )

    x0 = params.as_vector()
    baseline_res, _, _ = residual_fn(x0, params)
    baseline_res = np.asarray(baseline_res, dtype=float)

    # 诚实负例：观测通道退化（零运动/零信息量）时残差含 NaN
    # （normalize 除零范数）——这是"数据里没有可识别性"的物理
    # 事实，直接 NOT_IDENTIFIABLE，不启动优化不升级 twin。
    if not np.isfinite(baseline_res).all():
        from rosclaw.sim.contracts import SysIDReceipt

        receipt = SysIDReceipt(
            backend="mujoco",
            backend_version=str(mujoco.__version__),
            base_model_ref=base_ref,
            dataset_digest=str(dataset_digest),
            parameters_before=params_before,
            parameters_after=dict(params_before),
            identifiability_warning=True,
            verdict="NOT_IDENTIFIABLE",
        )
        receipt_dict = receipt.to_canonical_dict()
        receipt_dict["receipt_ref"] = backend.store.put("experiments", receipt_dict)
        return receipt_dict
    # §24 excitation 纪律在 NOT_IDENTIFIABLE 之后：零运动的诚实
    # 负例要先走残差退化判定，数据泄漏检查只管"有运动但同激励"。
    _excitation_check(dataset, train_idx + holdout_idx)
    baseline_cost = 0.5 * float(np.square(baseline_res).sum())

    opt_params, result = optimize.optimize(params, residual_fn, optimizer="scipy", verbose=False)
    optimized_cost = float(result.cost)

    params_after = {name: float(p.value[0]) for name, p in opt_params.items()}
    bounds_hit = [
        name
        for name, p in opt_params.items()
        if abs(float(p.value[0]) - float(p.min_value[0])) < 1e-9
        or abs(float(p.value[0]) - float(p.max_value[0])) < 1e-9
    ]

    # holdout 独立复算（同一残差管线，另一批序列）。
    holdout_seqs, holdout_obs = build_sequences(holdout_idx)
    holdout_fn = residual.build_residual_fn(
        models_sequences=holdout_seqs, enabled_observations=holdout_obs
    )
    holdout_base_res, _, _ = holdout_fn(x0, params)
    holdout_baseline = 0.5 * float(np.square(np.asarray(holdout_base_res)).sum())
    x_opt = opt_params.as_vector()
    holdout_opt_res, _, _ = holdout_fn(x_opt, opt_params)
    holdout_optimized = 0.5 * float(np.square(np.asarray(holdout_opt_res)).sum())

    if holdout_baseline > 1e-12:
        holdout_improvement = 1.0 - holdout_optimized / holdout_baseline
    else:
        holdout_improvement = 0.0

    identifiability_warning = bool(bounds_hit)
    improved = holdout_improvement >= _HOLDOUT_MIN_IMPROVEMENT

    # MH22 §23：Jacobian 可识别性诊断（不只 bounds_hit）。
    identifiability = _identifiability_diagnostics(result, opt_params)

    candidate_ref = ""
    audit_status = "NOT_EVALUATED"
    if improved:
        patch_ops = [
            {
                "op": "set",
                "target": {
                    "type": _PARAM_TYPES[p["type"]][0],
                    "name": p[_PARAM_TYPES[p["type"]][0]],
                },
                "field": _PARAM_TYPES[p["type"]][1],
                "value": params_after[_param_name(p)],
            }
            for p in parameters
        ]
        candidate_ref = backend.patch_model(base_ref, patch_ops).new_model_ref
        # MH22 §26 twin promotion 门：holdout 改进 + 可识别 +
        # physical audit——"更拟合日志" ≠ "更好的物理模型"。
        audit_status = backend.audit(candidate_ref).status

    identifiable = identifiability["classification"] == "IDENTIFIABLE"
    twin_candidate = bool(
        improved and identifiable and audit_status in ("PASS", "WARN")
    )
    simulation_profile: dict[str, Any] = {}
    if improved:
        simulation_profile = {
            "simulation": {
                "mujoco": {
                    "current_model_ref": base_ref,
                    "calibration": {
                        "candidate_model_ref": candidate_ref,
                        "dataset_digest": str(dataset_digest),
                        "method": "mujoco_sysid",
                        "holdout_improvement": holdout_improvement,
                        "parameters": {
                            name: {
                                "nominal": params_before[name],
                                "identified": params_after[name],
                                "confidence": identifiability["classification"],
                            }
                            for name in params_after
                        },
                    },
                }
            }
        }

    from rosclaw.sim.contracts import SysIDReceipt

    receipt = SysIDReceipt(
        backend="mujoco",
        backend_version=str(mujoco.__version__),
        base_model_ref=base_ref,
        dataset_digest=str(dataset_digest),
        parameters_before=params_before,
        parameters_after=params_after,
        bounds_hit=bounds_hit,
        baseline_residual_train=baseline_cost,
        optimized_residual_train=optimized_cost,
        holdout_baseline_residual=holdout_baseline,
        holdout_optimized_residual=holdout_optimized,
        holdout_improvement=holdout_improvement,
        identifiability_warning=identifiability_warning,
        identifiability=identifiability,
        promotion={
            "audit_status": audit_status,
            "twin_candidate": twin_candidate,
            "note": "TWIN_CANDIDATE 不自动覆盖 e-URDF——promotion 由 operator/policy 控制（§26）",
        },
        simulation_profile=simulation_profile,
        candidate_model_ref=candidate_ref,
        verdict="IMPROVED" if improved else "NO_IMPROVEMENT",
    )
    receipt_dict = receipt.to_canonical_dict()
    receipt_ref = backend.store.put("experiments", receipt_dict)
    receipt_dict["receipt_ref"] = receipt_ref
    return receipt_dict


def _current_value(spec_obj: Any, param: dict[str, Any]) -> float:  # noqa: ANN001
    """基座模型中参数的现值（nominal 起点）。"""
    obj_type, field = _PARAM_TYPES[param["type"]]
    name = param[obj_type]
    if obj_type == "joint":
        target = next(j for j in spec_obj.joints if j.name == name)
        return float(target.damping[0])
    if obj_type == "geom":
        target = next(g for g in spec_obj.geoms if g.name == name)
        return float(target.friction[0] if field == "friction" else target.mass)
    target = next(a for a in spec_obj.actuators if a.name == name)
    return float(target.gainprm[0])


# ---------------------------------------------------------------- MH22 v2

#: 相关性阈值（|corr|>0.95 = 参数对分不开）。
_WEAK_CORRELATION = 0.95
#: 条件数阈值（Jacobian 病态）。
_ILL_CONDITIONED = 1e8


def _identifiability_diagnostics(result, opt_params) -> dict[str, Any]:  # noqa: ANN001
    """Jacobian 诊断（§23）：rank/condition/sensitivity/相关对——
    不只看 bounds_hit。

    实证锚点：自由摆 damping+mass 轨迹完美拟合但参数错
    （0.19/0.95 vs 0.3/1.5）——Jacobian 两列近平行，必须被
    WEAKLY_IDENTIFIABLE 逮住。
    """
    raw_jac = getattr(result, "jac", None)
    jac = np.asarray(raw_jac, dtype=float) if raw_jac is not None else np.zeros((0, 0))
    names = list(opt_params.keys())
    n_params = len(names)
    diagnostics: dict[str, Any] = {
        "jacobian_rank": 0,
        "condition_number": 0.0,
        "parameter_sensitivity": {},
        "weak_parameter_pairs": [],
        "classification": "IDENTIFIABLE",
    }
    if jac.size == 0 or jac.ndim != 2 or jac.shape[1] != n_params:
        diagnostics["classification"] = "NOT_IDENTIFIABLE"
        diagnostics["note"] = "no_jacobian"
        return diagnostics
    column_norms = np.linalg.norm(jac, axis=0)
    diagnostics["parameter_sensitivity"] = {
        name: float(column_norms[i]) for i, name in enumerate(names)
    }
    if np.any(column_norms <= 1e-12):
        # 某列零敏感——该参数对数据完全不可见。
        diagnostics["classification"] = "NOT_IDENTIFIABLE"
        diagnostics["note"] = "zero_sensitivity_parameter"
        return diagnostics
    # 列归一化后 SVD。
    normalized = jac / column_norms
    singulars = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(singulars > max(normalized.shape) * np.finfo(float).eps * singulars[0]))
    diagnostics["jacobian_rank"] = rank
    condition = float(singulars[0] / singulars[-1]) if singulars[-1] > 0 else float("inf")
    diagnostics["condition_number"] = condition
    # J^T J 归一化相关矩阵 → 弱相关对。
    gram = normalized.T @ normalized
    weak_pairs = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            if abs(float(gram[i, j])) > _WEAK_CORRELATION:
                weak_pairs.append(
                    {"pair": [names[i], names[j]], "correlation": float(gram[i, j])}
                )
    diagnostics["weak_parameter_pairs"] = weak_pairs
    if rank < n_params:
        diagnostics["classification"] = "NOT_IDENTIFIABLE"
    elif condition > _ILL_CONDITIONED or weak_pairs:
        diagnostics["classification"] = "WEAKLY_IDENTIFIABLE"
    return diagnostics


def _excitation_check(dataset: dict[str, Any], indices: list[int]) -> None:
    """§24 excitation 纪律：train+holdout 全部同激励同初值 =
    数据泄漏（同一条轨迹切两半不算 holdout）。"""
    seen = set()
    for idx in indices:
        seq = dataset["sequences"][idx]
        key = (
            str(seq.get("controller")),
            tuple(seq.get("qpos0") or []),
            float(seq.get("duration_s", 0)),
        )
        seen.add(key)
    if len(seen) < 2 and len(indices) >= 2:
        raise ValueError(
            "EXCITATION_INSUFFICIENT: train/holdout 序列激励完全相同"
            "（同轨迹切两半不算 holdout——不同初值/激励/负载才有效）"
        )
