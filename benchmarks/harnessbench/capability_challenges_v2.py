"""Public synthetic challenge contracts and finite plan; no private answers.

Isolated from tasks.py, old case registries, physics, providers and ROS.
Call build_registry() explicitly. Importing this module performs no I/O.
"""
from __future__ import annotations

import hashlib
import json
import struct
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

VERSION = "rosclaw.capability_challenges.v2.2"
SCOPE = "FIXTURE_ONLY"
ABS_TOL = 1e-9
REL_TOL = 1e-9
ROLES = ("success", "invalid", "companion")
ORIGINALS = {
    "H01": ("C01", "C02", "C03", "C04", "C05", "C07"),
    "H02": ("C08", "C09", "C10", "C14", "C23", "C33", "C34"),
    "H03": ("C06", "C11", "C12", "C47"),
    "H04": ("C13", "C17", "C18", "C19", "C20", "C21"),
    "H05": ("C22", "C24", "C25", "C28"),
    "H06": ("C26", "C27", "C29", "C30"),
    "H07": ("C31", "C32", "C39"),
    "H08": ("C35", "C36", "C37", "C38"),
    "H09": ("C15", "C16", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "CX12"),
    "H10": ("C48", "C49"),
}

# Contracts precede fixtures and references. These define NEW semantics, not
# an assertion that unavailable original-case tie rules have been reproduced.
CONTRACTS = {
    "H01": "Each edge has exact type strings, domain integers and requested/offered "
    "reliability (best_effort<reliable) and durability (volatile<transient_local). "
    "Compatible iff type/domain equal and offered ranks >= requested ranks. "
    "Resolve absolute names unchanged, relative names under namespace, ~ names "
    "under namespace/node. Lifecycle configure:unconfigured->inactive, "
    "activate:inactive->active, deactivate:active->inactive; other transition "
    "rejects the whole input. Output links sorted by id, resolved subscriber topic, "
    "and final lifecycle. Missing type rejects, never compatible."
    " Output JSON values has exactly links,topic,state. links is an array of exact "
    "{id:string,compatible:boolean} objects, one per input edge, ascending id. "
    "topic is the resolved topic string; state is the final lifecycle string "
    "(unconfigured, inactive or active). The key is state, never lifecycle. "
    "No omitted/extra keys at any level; no aliases. All numeric fields anywhere "
    "are finite JSON numbers, excluding booleans; integer fields require integer "
    "tokens, not 1.0, strings or booleans. Booleans are true/false only. Arrays "
    "are ordered and never replaced by scalars; null only where explicitly allowed.",
    "H02": "SI units; quaternion [x,y,z,w] unit within 1e-12; active right-handed "
    "rotation. p_target=R(q)*p_source+t; inverse applied to it must recover p. "
    "Output point, roundtrip, gravity_free=R(q)*accel-[0,0,9.81], "
    "signed plane distance (n dot point+d)/norm(n), translated torque=t cross force "
    "+torque, target frame, fresh iff 0<=now-stamp<=max_age. Reject frame mismatch, "
    "nonunit rotation, zero plane normal or nonfinite numbers. q and -q equivalent."
    " Output JSON values has exactly point,roundtrip,gravity_free,distance,torque,"
    "frame,fresh. point,roundtrip,gravity_free,torque are length-3 arrays of finite "
    "numbers in x,y,z order; distance is a finite number (key distance, never "
    "plane_distance); frame is a string; fresh is a boolean. No nulls, aliases, "
    "omitted or extra keys; booleans are not numbers; arrays cannot be scalars.",
    "H03": "Integer ticks. Process anchor stream in given order. For each anchor "
    "and each secondary stream choose unused sample within inclusive window, "
    "minimum (absolute delta, timestamp, original index); consume immediately. "
    "Duplicates remain distinct. Output matched index rows (-1 absent), rewind "
    "indices where clock[i]<clock[i-1], lost=sum(max(0,seq[i]-seq[i-1]-1)); "
    "action fresh iff 0<=now-action_stamp<=max_age. Reject noninteger stamps."
    " Output JSON values has exactly matches,rewinds,lost,fresh. matches is an "
    "array of len(anchors) rows, each an array of len(streams) genuine integers, "
    "in original anchor/stream order; each entry is -1 or a zero-based index in "
    "that stream. rewinds is an ascending array of zero-based clock encounter "
    "indices (1 through len(clock)-1); lost is a nonnegative genuine integer; "
    "fresh is boolean. Integer tokens only, not decimal tokens, strings or bools; "
    "no nulls, omitted/extra keys or scalar-for-array substitutions.",
    "H04": "Decode width*height pixels of three uint16 channels from hex buffer, "
    "declared endian, row stride including padding, permutation channel_order. "
    "Order maps stored channels to R,G,B. Output RGB triplets in row-major order, valid_depth "
    "for depth>0 finite numeric samples (null invalid), and deprojected xyz "
    "for valid samples at pixel (u,v), z=depth*scale, x=(u-cx)*z/fx, "
    "y=(v-cy)*z/fy; invalid xyz=null. Require exact height*stride bytes, "
    "stride>=width*6, positive intrinsics and scale, depth length=width*height."
    " Output JSON values has exactly rgb,valid_depth,xyz, each an array of "
    "width*height entries in row-major pixel order. rgb entries are length-3 "
    "R,G,B arrays of genuine integers in [0,65535]. valid_depth entries are "
    "booleans. xyz entries are length-3 finite-number x,y,z arrays when the "
    "corresponding valid_depth is true, otherwise null. No other nulls, aliases, "
    "omitted or extra keys. Numbers exclude booleans; integer tokens only; "
    "ordered arrays cannot be scalars.",
    "H05": "floor each 2D point/resolution (including negatives); output sorted "
    "unique voxel pairs. Cells -1 unknown,0 free,100 occupied. Unknown is blocked. "
    "Inflate blocked cells by Chebyshev radius, clipped to grid; output sorted "
    "[x,y] blocked cells. Inserting obstacles cannot remove blocked cells. "
    "Frame must equal required_frame; positive resolution, rectangular grid."
    " Output JSON values has exactly voxels,blocked. Each is an array of unique "
    "length-2 [x,y] genuine-integer coordinate arrays, sorted lexicographically "
    "by x then y (not row-major). voxels may have negative coordinates; blocked "
    "coordinates are inside the input grid. Empty arrays are allowed. No nulls, "
    "omitted/extra keys or scalars; integer tokens exclude decimals and booleans.",
    "H06": "Undirected graph, unique node names, positive weights; forbidden edges "
    "removed. Diagonal edge allowed only if every listed corner cell is clear. "
    "Choose minimum cost simple path then lexicographically smallest full node "
    "sequence. Unreachable yields reachable=false,path=[],cost=null. Output also "
    "braking_safe=(speed^2/(2*deceleration)<=clearance) inclusive, "
    "time_valid=all successive times strictly increasing. Bad parameters reject."
    " Output JSON values has exactly reachable,cost,path,braking_safe,time_valid. "
    "reachable,braking_safe,time_valid are booleans. When reachable=true, cost "
    "is a finite nonnegative number and path is an ordered string array from "
    "start to goal, inclusive. When reachable=false, cost is null and path is []. "
    "No other nulls, omitted/extra keys or aliases; numbers exclude booleans "
    "and ordered arrays cannot be scalars.",
    "H07": "Normalize m/mm to meters. Inclusive annulus min_reach<=hypot(x,y) "
    "<=max_reach and width<=max_width (nonnegative). Output reachable, graspable, "
    "and odometry [x,y,theta] starting zero: for each [v,omega,dt], translate "
    "v*dt at PREVIOUS theta then theta+=omega*dt (no wrap); rad/s and seconds. "
    "v is in input length units/s. Reject dt<=0 and invalid annulus."
    " Output JSON values has exactly reachable,graspable,odometry. reachable "
    "and graspable are booleans; odometry is a length-3 finite-number array "
    "ordered [x,y,theta], x,y in meters and theta in radians. No nulls, aliases, "
    "omitted/extra keys or scalar arrays; numbers exclude booleans.",
    "H08": "Finite samples; moving average partial prefixes up to integer window; "
    "EMA first sample initializes, then alpha*x+(1-alpha)*previous. Scalar "
    "Kalman: prior state/variance, per sample Pminus=P+Q, K=Pminus/(Pminus+R), "
    "x=x+K*(z-x), P=(1-K)*Pminus; require P,Q>=0,R>0,alpha in [0,1]. "
    "Gate samples by abs(sample-center)<=gate inclusive. Output ma,ema, "
    "posterior,variance,accepted indices. No silent uncertainty repair."
    " Output JSON values has exactly ma,ema,posterior,variance,accepted. ma,ema,"
    "posterior,variance are EACH arrays of len(samples) finite numbers aligned "
    "with the ORIGINAL sample order, including gated-out samples. posterior "
    "and variance contain the per-sample updated state and variance, NOT final "
    "scalars and NOT an initial-state prefix. accepted is an ascending array "
    "of genuine integer zero-based original sample indices, possibly empty. "
    "No nulls, omitted/extra keys, aliases or scalar-for-array shortcuts; "
    "numbers exclude booleans; integer tokens exclude decimals and strings.",
    "H09": "Unique named channels with unique nonnegative integer addresses; "
    "requested names must be an exact permutation. Encoder signed delta is "
    "((new-old+period/2)%period)-period/2 (positive half maps negative). "
    "PD=kp*(target-position)-kd*velocity; clamp to [lo,hi], then slew around "
    "previous by rate*dt. Joint force=command*gear (signed nonzero). "
    "Normalized action a in [-1,1] maps lo+(a+1)*(hi-lo)/2. Output sorted by "
    "name rows of name,address,delta,command,force,action; units rad and Nm; "
    "dt,period>0,rate>=0; no model inference or applied control."
    " Output JSON values has exactly joints (never rows). joints is an array "
    "of one object per input channel sorted ascending name; each object has "
    "exactly name,address,delta,command,force,action. name is string; address "
    "is a nonnegative genuine integer; delta,command,force,action are finite "
    "numbers. action is the mapped command, not the normalized input a. "
    "No nulls, omitted/extra keys, aliases or scalars; numbers exclude booleans "
    "and integer tokens exclude decimals and strings.",
    "H10": "Read batch.json and checkpoint.json as immutable bytes. Batch revision "
    "and checkpoint revision must equal input revision; checkpoint batch_sha256 "
    "must match exact batch bytes. History must be a duplicate-free list of "
    "terminal names; preserve it, append result.json once only. Required calls= "
    "number of batch files+1 final registration minus 1 if result.json already "
    "terminal. If budget<required return BLOCKED/BUDGET_EXHAUSTED without new "
    "terminals. Choose unique landmark with requested label; multiple=>AMBIGUOUS, "
    "none=>NOT_FOUND. For unique landmark act only if ordered events contain "
    "open before enter (first occurrence each); else WAIT. Output decision, "
    "target or null, history and required_calls. Damaged JSON=>INVALID_JSON; "
    "stale revision/hash=>STALE_CHECKPOINT. This is a symbolic recovery plan, "
    "not proof of tool calls, cost, STOP or actual artifact registration."
    " For status=OK/code=OK, values has exactly decision,target,history,"
    "required_calls. decision is ACT,WAIT,AMBIGUOUS or NOT_FOUND; target is a "
    "string iff there is exactly one matching landmark (ACT or WAIT), otherwise "
    "null. history is a duplicate-free string array in preserved/append order, "
    "not sorted; required_calls is a nonnegative genuine integer. For status="
    "BLOCKED/code=BUDGET_EXHAUSTED, values has exactly history,required_calls "
    "with the same types and unchanged history, no decision or target. For "
    "status=INVALID, values=null and code identifies the input error, including "
    "INVALID_JSON or STALE_CHECKPOINT. No other nulls, omitted/extra keys, "
    "aliases or scalar arrays; integers exclude decimals, strings and bools.",
}
ERRORS = {
    "H01": "MISSING_TYPE", "H02": "INVALID_ROTATION", "H03": "INVALID_STAMP",
    "H04": "BUFFER_LENGTH", "H05": "FRAME_MISMATCH", "H06": "INVALID_GRAPH",
    "H07": "INVALID_KINEMATICS", "H08": "INVALID_UNCERTAINTY",
    "H09": "INVALID_JOINT_MAP", "H10": "INVALID_JSON",
}
RELATIONS = {
    "H01": "Prefix edge ids/node/topic with r_, reverse edges: compatible and lifecycle invariant; names equivariant.",
    "H02": "Negate quaternion and rename source/target frames: all numeric outputs invariant; target renamed.",
    "H03": "Add 100 to all clocks/stamps/now: matches, loss, rewind indices and freshness invariant.",
    "H04": "Reencode little->big endian, RGB->BGR, add row padding: all outputs invariant.",
    "H05": "Reverse point order: voxels and inflated cells invariant.",
    "H06": "Prefix all nodes with r_: path equivariant, all other outputs invariant (order-preserving bijection).",
    "H07": "Scale lengths and velocities m->mm: all SI outputs invariant.",
    "H08": "Add 7 to samples, initial state and gate center: ma/ema/posterior +7; variance/accepted invariant.",
    "H09": "Reverse channel and request ordering: all sorted named rows invariant.",
    "H10": "Prefix landmark ids with r_: target equivariant; decision/history/required_calls invariant.",
}


# Public structural schemas only; no computed answers. Dynamic cardinalities,
# ordering and cross-field constraints are specified by CONTRACTS and prompt.
RESULT_VALUE_SCHEMAS = {
    "H01": {
        "type": "object", "additionalProperties": False,
        "required": ["links", "topic", "state"],
        "properties": {
            "links": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "compatible"],
                "properties": {"id": {"type": "string"}, "compatible": {"type": "boolean"}}}},
            "topic": {"type": "string"},
            "state": {"type": "string", "enum": ["unconfigured", "inactive", "active"]}}},
    "H02": {
        "type": "object", "additionalProperties": False,
        "required": ["point", "roundtrip", "gravity_free", "distance", "torque", "frame", "fresh"],
        "properties": {
            "point": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "roundtrip": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "gravity_free": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "distance": {"type": "number"},
            "torque": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
            "frame": {"type": "string"}, "fresh": {"type": "boolean"}}},
    "H03": {
        "type": "object", "additionalProperties": False,
        "required": ["matches", "rewinds", "lost", "fresh"],
        "properties": {
            "matches": {"type": "array", "items": {"type": "array", "items": {"type": "integer", "minimum": -1}}},
            "rewinds": {"type": "array", "uniqueItems": True, "items": {"type": "integer", "minimum": 1}},
            "lost": {"type": "integer", "minimum": 0}, "fresh": {"type": "boolean"}}},
    "H04": {
        "type": "object", "additionalProperties": False,
        "required": ["rgb", "valid_depth", "xyz"],
        "properties": {
            "rgb": {"type": "array", "items": {"type": "array", "minItems": 3, "maxItems": 3,
                      "items": {"type": "integer", "minimum": 0, "maximum": 65535}}},
            "valid_depth": {"type": "array", "items": {"type": "boolean"}},
            "xyz": {"type": "array", "items": {"oneOf": [
                {"type": "null"}, {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}]}}}},
    "H05": {
        "type": "object", "additionalProperties": False, "required": ["voxels", "blocked"],
        "properties": {
            "voxels": {"type": "array", "uniqueItems": True, "items": {
                "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}},
            "blocked": {"type": "array", "uniqueItems": True, "items": {
                "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0}}}}},
    "H06": {
        "type": "object", "additionalProperties": False,
        "required": ["reachable", "cost", "path", "braking_safe", "time_valid"],
        "properties": {
            "reachable": {"type": "boolean"},
            "cost": {"oneOf": [{"type": "number", "minimum": 0}, {"type": "null"}]},
            "path": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
            "braking_safe": {"type": "boolean"}, "time_valid": {"type": "boolean"}},
        "allOf": [{"if": {"properties": {"reachable": {"const": False}}},
                   "then": {"properties": {"cost": {"type": "null"}, "path": {"maxItems": 0}}},
                   "else": {"properties": {"cost": {"type": "number", "minimum": 0}, "path": {"minItems": 1}}}}]},
    "H07": {
        "type": "object", "additionalProperties": False,
        "required": ["reachable", "graspable", "odometry"],
        "properties": {"reachable": {"type": "boolean"}, "graspable": {"type": "boolean"},
                       "odometry": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}}},
    "H08": {
        "type": "object", "additionalProperties": False,
        "required": ["ma", "ema", "posterior", "variance", "accepted"],
        "properties": {
            "ma": {"type": "array", "items": {"type": "number"}},
            "ema": {"type": "array", "items": {"type": "number"}},
            "posterior": {"type": "array", "items": {"type": "number"}},
            "variance": {"type": "array", "items": {"type": "number", "minimum": 0}},
            "accepted": {"type": "array", "uniqueItems": True, "items": {"type": "integer", "minimum": 0}}}},
    "H09": {
        "type": "object", "additionalProperties": False, "required": ["joints"],
        "properties": {"joints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "address", "delta", "command", "force", "action"],
            "properties": {"name": {"type": "string"}, "address": {"type": "integer", "minimum": 0},
                           "delta": {"type": "number"}, "command": {"type": "number"},
                           "force": {"type": "number"}, "action": {"type": "number"}}}}}},
    "H10": {
        "type": "object", "additionalProperties": False,
        "required": ["decision", "target", "history", "required_calls"],
        "properties": {
            "decision": {"type": "string", "enum": ["ACT", "WAIT", "AMBIGUOUS", "NOT_FOUND"]},
            "target": {"type": ["string", "null"]},
            "history": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
            "required_calls": {"type": "integer", "minimum": 0}},
        "allOf": [{"if": {"properties": {"decision": {"enum": ["ACT", "WAIT"]}}},
                   "then": {"properties": {"target": {"type": "string"}}},
                   "else": {"properties": {"target": {"type": "null"}}}}]},
}

# Per-family INVALID result branches. The public prompt combines each with
# OK (and H10 BLOCKED) into the complete answer schema. No reference imports.
RESULT_SCHEMAS = {
    "H01": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["MISSING_TYPE", "INVALID_QOS", "INVALID_TRANSITION", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H02": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_ROTATION", "FRAME_MISMATCH", "INVALID_GEOMETRY", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H03": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_STAMP", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H04": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["BUFFER_LENGTH", "BUFFER_SHAPE", "BUFFER_TYPE", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H05": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["FRAME_MISMATCH", "INVALID_GRID", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H06": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_GRAPH", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H07": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_KINEMATICS", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H08": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_UNCERTAINTY", "INVALID_FILTER", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H09": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_JOINT_MAP", "INVALID_LIMITS", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
    "H10": {"type": "object", "additionalProperties": False, "required": ["status", "code", "values"],
            "properties": {"status": {"const": "INVALID"}, "code": {"type": "string", "enum": ["INVALID_JSON", "STALE_CHECKPOINT", "INVALID_HISTORY", "INVALID_BUDGET", "NONFINITE_INPUT"]}, "values": {"type": "null"}}},
}


def encode(value: Any) -> bytes:
    """Canonical UTF-8: sorted keys, compact separators, newline; reject NaN."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Instance:
    task_id: str
    family: str
    role: str
    seed: int
    files: tuple[tuple[str, bytes], ...]

    def staged(self) -> dict[str, bytes]:
        return dict(self.files)

    def manifest(self) -> dict[str, Any]:
        return {"version": VERSION, "task_id": self.task_id, "family": self.family,
                "role": self.role, "seed": self.seed,
                "generator": "sha256-seed-v1", "scope": SCOPE,
                "input_hashes": {k: sha(v) for k, v in self.files},
                "contract_sha256": sha(CONTRACTS[self.family].encode()),
                "abs_tol": ABS_TOL, "rel_tol": REL_TOL,
                "original_direction_ids": list(ORIGINALS[self.family]),
                "companion_of": f"{self.family}-S" if self.role == "companion" else None,
                "relation": RELATIONS[self.family]}

    def prompt(self) -> str:
        # Only public shapes/cardinalities, never calculations or reference values.
        data = json.loads(self.staged()["input.json"])["data"]
        values = deepcopy(RESULT_VALUE_SCHEMAS[self.family])
        props = values["properties"]
        lengths = {}
        if self.family == "H01":
            lengths = {"links": len(data["edges"])}
        elif self.family == "H03":
            lengths = {"matches": len(data["anchors"])}
            row = props["matches"]["items"]
            row.update(minItems=len(data["streams"]), maxItems=len(data["streams"]))
            row["prefixItems"] = [{"type": "integer", "minimum": -1, "maximum": len(s)-1}
                                  for s in data["streams"]]
            props["rewinds"]["items"]["maximum"] = len(data["clock"])-1
        elif self.family == "H04":
            lengths = dict.fromkeys(("rgb", "valid_depth", "xyz"), data["width"]*data["height"])
        elif self.family == "H08":
            lengths = {key: len(data["samples"]) for key in ("ma", "ema", "posterior", "variance")}
            props["accepted"]["items"]["maximum"] = len(data["samples"])-1
        elif self.family == "H09":
            lengths = {"joints": len(data["channels"])}
        for key, size in lengths.items():
            props[key].update(minItems=size, maxItems=size)
        branches = [
            {"type": "object", "additionalProperties": False,
             "required": ["status", "code", "values"],
             "properties": {"status": {"const": "OK"}, "code": {"const": "OK"}, "values": values}},
            deepcopy(RESULT_SCHEMAS[self.family]),
        ]
        if self.family == "H10":
            branches.append({
                "type": "object", "additionalProperties": False,
                "required": ["status", "code", "values"],
                "properties": {"status": {"const": "BLOCKED"}, "code": {"const": "BUDGET_EXHAUSTED"},
                               "values": {"type": "object", "additionalProperties": False,
                                          "required": ["history", "required_calls"],
                                          "properties": {"history": {"type": "array", "uniqueItems": True,
                                                                     "items": {"type": "string"}},
                                                         "required_calls": {"type": "integer", "minimum": 0}}}}})
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": VERSION + " " + self.family + " answer envelope",
            "type": "object", "additionalProperties": False,
            "required": ["scope", "result", "input_sha256", "reason"],
            "properties": {"scope": {"const": "FIXTURE_ONLY"},
                           "result": {"oneOf": branches},
                           "input_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                           "reason": {"type": "string", "minLength": 1, "pattern": "\\S"}},
        }
        return ("Synthetic offline task; no ROS/DDS/hardware/provider/engine access. "
                "Preserve every staged byte. Contract " + VERSION + ": "
                + CONTRACTS[self.family] + "\n"
                "Write answer.json with exactly scope,result,input_sha256,reason. "
                "scope=FIXTURE_ONLY; input_sha256 is the lowercase SHA256 of exact input.json bytes; "
                "reason is a non-whitespace explanation with fixture-only limits. "
                "result has exactly status,code,values. OK requires code=OK and the family values object. "
                "INVALID requires the applicable input-error code and values=null, never numeric success. "
                "BLOCKED is permitted only for H10 budget exhaustion, code=BUDGET_EXHAUSTED, "
                "values exactly unchanged history and required_calls. "
                "The planned invalid variant code is " + ERRORS[self.family] + ". "
                "Numeric tolerance absolute and relative 1e-9. ALL numbers must be finite JSON numbers; "
                "booleans are never numbers. Integer positions require genuine JSON integer tokens, "
                "not decimal/exponent tokens, strings or booleans (stricter than JSON Schema's "
                "mathematical integer convention). Boolean positions require true/false. "
                "No duplicate, omitted or extra keys at ANY object depth; no aliases. "
                "Arrays are ordered, never scalars; only explicitly nullable positions allow null. "
                "The structural schema below enforces keys/types/cardinalities/status branches; "
                "the contract additionally binds ordering, input-derived values, pixel validity, "
                "history preservation, index meaning and hashes. It is not a reference answer.\n"
                + "Public answer schema: " + encode(schema).decode()
                + "Public provenance: " + encode(self.manifest()).decode())


def _payload(family: str, seed: int) -> dict[str, Any]:
    # One bounded degree of variation, explicitly not a distribution claim.
    k = int.from_bytes(hashlib.sha256(str(seed).encode("ascii")).digest()[:2], "big") % 5 + 1
    if family == "H01":
        return {"namespace": "/lab", "node": "reader", "topic": "~data",
                "state": "unconfigured", "events": ["configure", "activate"],
                "edges": [{"id": "a", "type_requested": "T", "type_offered": "T",
                           "domain_requested": k, "domain_offered": k,
                           "reliability_requested": "reliable", "reliability_offered": "reliable",
                           "durability_requested": "transient_local", "durability_offered": "volatile"},
                          {"id": "b", "type_requested": "T", "type_offered": "T",
                           "domain_requested": k, "domain_offered": k,
                           "reliability_requested": "best_effort", "reliability_offered": "reliable",
                           "durability_requested": "volatile", "durability_offered": "volatile"}]}
    if family == "H02":
        return {"source": "sensor", "target": "map", "point_frame": "sensor",
                "point": [k, -2, 1], "q": [0, 0, 1, 0], "t": [1, k, 0],
                "accel": [0, 0, 9.81], "plane": [0, 0, 1, -1],
                "force": [1, 0, 0], "torque": [0, 0, 0],
                "stamp": 90, "now": 100, "max_age": 10}
    if family == "H03":
        return {"anchors": [10, 20, 20], "streams": [[8, 12, 20, 20], [11, 19, 21]],
                "window": 2, "clock": [0, k, 0, 2], "seq": [1, 2, 5, 5],
                "action_stamp": 10, "now": 20, "max_age": 10}
    if family == "H04":
        return {"width": 2, "height": 1, "stride": 14, "endian": "little",
                "channel_order": "RGB", "buffer": struct.pack("<6H", k, 2, 3, 4, 5, 6).hex() + "abcd",
                "depth": [k, None], "scale": 0.001, "fx": 2, "fy": 2, "cx": 0, "cy": 0}
    if family == "H05":
        return {"frame": "map", "required_frame": "map", "resolution": 0.5,
                "points": [[-0.5, 0], [-0.01, 0], [k, 0]],
                "grid": [[100, 0, 0], [0, -1, 0], [0, 0, 0]], "radius": 1}
    if family == "H06":
        return {"nodes": ["a", "b", "c", "d"], "start": "a", "goal": "d",
                "edges": [["a", "b", k], ["b", "d", 1], ["a", "c", k], ["c", "d", 1]],
                "forbidden": [], "diagonals": [{"edge": ["a", "d"], "clear": [True, False]}],
                "speed": 2, "deceleration": 2, "clearance": 1, "times": [0, 1, 2]}
    if family == "H07":
        return {"unit": "m", "point": [0.3, 0.4], "min_reach": 0.1, "max_reach": 0.5,
                "width": 0.08, "max_width": 0.08,
                "steps": [[k, 0, 1], [0, 1, 1], [1, 0, 0.5]]}
    if family == "H08":
        return {"samples": [k, k, k + 10, k], "window": 2, "alpha": 0.5,
                "initial": k, "P": 1, "Q": 0, "R": 1, "center": k, "gate": 0}
    if family == "H09":
        row = {"name": "a", "address": 0, "old": 0, "new": 8, "period": 16,
               "kp": 2, "kd": 1, "target": k, "position": 0, "velocity": 0,
               "lo": -2, "hi": 3, "previous": 0, "rate": 1, "dt": 1,
               "gear": -2, "a": 1}
        other = dict(row, name="b", address=3, new=15, target=-k, gear=3, a=-1)
        return {"requested": ["b", "a"], "channels": [row, other]}
    if family == "H10":
        return {"revision": k, "budget": 3, "label": "door",
                "landmarks": [{"id": "x", "label": "door"}, {"id": "y", "label": "table"}],
                "events": ["open", "enter"]}
    raise ValueError("UNKNOWN_FAMILY")


def make_instance(family: str, role: str) -> Instance:
    if family not in CONTRACTS or role not in ROLES:
        raise ValueError("UNKNOWN_PLAN_ENTRY")
    seed = 2026101600 + int(family[1:])  # same source seed for paired transformations
    p = deepcopy(_payload(family, seed))
    extras: dict[str, bytes] = {}
    if family == "H10":
        extras["batch.json"] = encode({"revision": p["revision"], "files": ["one.json", "two.json"]})
        extras["checkpoint.json"] = encode({"revision": p["revision"], "history": [],
                                                  "batch_sha256": sha(extras["batch.json"])})
    if role == "invalid":
        if family == "H01":
            del p["edges"][0]["type_requested"]
        elif family == "H02":
            p["q"] = [0, 0, 0, 0]
        elif family == "H03":
            p["anchors"][0] = "10"
        elif family == "H04":
            p["buffer"] = p["buffer"][:-2]
        elif family == "H05":
            p["frame"] = "odom"
        elif family == "H06":
            p["edges"][0][2] = -1
        elif family == "H07":
            p["steps"][0][2] = 0
        elif family == "H08":
            p["R"] = 0
        elif family == "H09":
            p["channels"][1]["name"] = "a"
        else:
            extras["checkpoint.json"] = b'{"revision":'
    if role == "companion":
        if family == "H01":
            p["node"] = "r_" + p["node"]
            p["topic"] = "~r_data"
            for edge in p["edges"]:
                edge["id"] = "r_" + edge["id"]
            p["edges"].reverse()
        elif family == "H02":
            p["q"] = [-v for v in p["q"]]
            p.update(source="r_sensor", point_frame="r_sensor", target="r_map")
        elif family == "H03":
            for key in ("anchors", "clock"):
                p[key] = [v + 100 for v in p[key]]
            p["streams"] = [[v + 100 for v in stream] for stream in p["streams"]]
            p["action_stamp"] += 100
            p["now"] += 100
        elif family == "H04":
            k = struct.unpack("<6H", bytes.fromhex(p["buffer"])[:12])
            p.update(buffer=struct.pack(">6H", k[2], k[1], k[0], k[5], k[4], k[3]).hex() + "00000000",
                     endian="big", channel_order="BGR", stride=16)
        elif family == "H05":
            p["points"].reverse()
        elif family == "H06":
            p["nodes"] = ["r_" + v for v in p["nodes"]]
            for key in ("start", "goal"):
                p[key] = "r_" + p[key]
            p["edges"] = [["r_" + a, "r_" + b, w] for a, b, w in p["edges"]]
            p["diagonals"][0]["edge"] = ["r_a", "r_d"]
        elif family == "H07":
            p["unit"] = "mm"
            for key in ("min_reach", "max_reach", "width", "max_width"):
                p[key] *= 1000
            p["point"] = [v * 1000 for v in p["point"]]
            p["steps"] = [[v * 1000, w, dt] for v, w, dt in p["steps"]]
        elif family == "H08":
            p["samples"] = [v + 7 for v in p["samples"]]
            p["initial"] += 7
            p["center"] += 7
        elif family == "H09":
            p["channels"].reverse()
            p["requested"].reverse()
        else:
            for landmark in p["landmarks"]:
                landmark["id"] = "r_" + landmark["id"]
    envelope = {"version": VERSION, "family": family, "role": role, "seed": seed, "data": p,
                "aux_sha256": {name: sha(data) for name, data in extras.items()}}
    files = {"input.json": encode(envelope), **extras}
    suffix = {"success": "S", "invalid": "I", "companion": "M"}[role]
    return Instance(f"{family}-{suffix}", family, role, seed, tuple(sorted(files.items())))


def build_registry() -> dict[str, Instance]:
    """Exactly first30, never silently extends old registries."""
    entries = [make_instance(family, role) for family in CONTRACTS for role in ROLES]
    return {entry.task_id: entry for entry in entries}


def as_bench_task(instance: Instance) -> Any:
    """Optional adapter; lazy public interface import, no old oracle import."""
    from benchmarks.harnessbench.task_common import BenchTask

    canonical = make_instance(instance.family, instance.role)
    if instance != canonical:
        raise ValueError("UNREGISTERED_INSTANCE")
    return BenchTask(task_id=instance.task_id, category="offline_contract_v2",
                     prompt=instance.prompt(),
                     staged_files={name: data.decode("utf-8") for name, data in instance.files},
                     oracle={"kind": "offline_capability_challenge_v2", "task_id": instance.task_id})
