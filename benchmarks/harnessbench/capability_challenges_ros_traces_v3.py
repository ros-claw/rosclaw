"""Public v3 symbolic ROS fixtures. No execution, I/O, or private judge import."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

VERSION = "rosclaw.capability_challenges.ros_traces.v3.1"

# Public, inclusive resource ceilings. The private candidate uses these same limits.
MAX_BYTES = 1048576
JSON_DEPTH = 32
JSON_TOKENS = 20000
INTEGER_DIGITS = 20
NUMBER_CHARS = 64
EXPONENT_ABS = 100
FLOAT_ABS = 1e100
INPUT_NODES = 2048
INPUT_CONTAINER = 128
INPUT_STRING_BYTES = 256
INPUT_TOTAL_STRING_BYTES = 16384
INPUT_INTEGER_ABS = 10**15
INPUT_FLOAT_ABS = 1e50
H17_GROUPS = 8
H17_CALLBACKS = 8
H17_EVENTS = 128
H17_EDGES = 16
H18_PUBLICATIONS = 128
H18_TIME = 10**12
H18_OUTPUT_EVENTS = 4096
SCOPE = "FIXTURE_ONLY"
ROLES = ("subject", "invalid", "companion")
FAMILIES = tuple(f"H{i}" for i in range(11, 21))
CAP80_IDS = dict(zip(FAMILIES, ("CAP80_001", "CAP80_002", "CAP80_003", "CAP80_004", "CAP80_005", "CAP80_006", "CAP80_007", "CAP80_010", "CAP80_011", "CAP80_012"), strict=False))
ABS_TOL = REL_TOL = 1e-9

COMMON = """Offline symbolic fixtures only, not ROS runtime or physics evidence.
Input is input.json: exact keys version,family,data. Numbers finite; booleans are
not numbers; integers require integer JSON tokens, never 1.0. All objects have
exact documented keys, including nested ones. Nonempty strings unless stated.
Generic malformed shape/type input => INVALID_INPUT. Family-specific checks
follow the resource admission and shape checks below. Stop at first failure.
JSON decoding/duplicate keys/invalid UTF8/NaN/Infinity => INVALID_JSON.
No resource execution. Admission order (inclusive ceilings):
1. Raw input bytes above 1048576 => SIZE_LIMIT; strict UTF8 decode next.
2. Before JSON allocation, scan lexemes left-to-right, ignoring JSON whitespace,
commas, colons and closing brackets. Each opening container, quoted string
(including object keys), number or literal is one token; at most 20000 tokens.
At most 32 simultaneously open arrays/objects (root container has depth 1).
Numeric lexemes are maximal non-whitespace runs bounded by []{},:\". Before
conversion: at most 64 characters per number, at most 20 digits for integer
lexemes (minus excluded), explicit float exponent absolute value <=100.
Then require JSON number grammar; floats must be finite and abs <=1e100.
Any exceeded numeric/depth/token ceiling => RESOURCE_LIMIT, even if later JSON
syntax is bad. Within each numeric lexeme check length, integer digit count,
grammar, explicit exponent, converted magnitude in that order. Bad numeric
grammar => INVALID_JSON. A literal NaN/Infinity is INVALID_JSON, not a number.
3. Decode complete JSON, rejecting duplicates and syntax errors as INVALID_JSON.
These lexical limits apply to answers too; bad answers fail ANSWER_ENVELOPE.
4. INPUT ONLY: walk the entire decoded envelope in depth-first insertion order.
Count every value and object key once (root counts); <=2048 nodes, <=128 entries
per array/object, <=256 UTF8 bytes per string/key and <=16384 total string/key
bytes. For this count only use UTF8 surrogatepass (3 bytes per lone surrogate);
family H11 still requires strict UTF8 when encoding its label. Integer values
must have abs <=1000000000000000 (10^15), float values abs <=1e50; booleans
are not subject to numeric ceilings. These input-only headroom bounds ensure
integer increments and TF compositions remain within the shared JSON output
numeric limits. All exceeded input domain ceilings => RESOURCE_LIMIT, before
envelope/family shape checks.
5. Check envelope, family shapes and semantics in their documented order.
H17/H18 add the explicit domain checks below. No hidden resource rejection.
These ceilings bound traversal and supported output; an accepted computation
admits a compact answer with a short reason well below the 1 MiB answer ceiling.
Output exact keys scope,result,input_sha256,reason; scope FIXTURE_ONLY; hash is
lowercase SHA256 of exact input.json bytes; reason contains non-whitespace.
result exact keys status,code,values. OK/OK uses exact values schema; INVALID
uses an applicable public code and values null. No BLOCKED. Array order matters;
integers/strings/booleans exact; only real-valued H14 rotations/translations and
H15 samples use abs/relative tolerance 1e-9. No hidden field or ordering rules.
Input JSON max 1 MiB; answer max 1 MiB; SIZE_LIMIT for oversized input. Public
fixtures are bounded; these limits do not authorize external processing.
"""
CONTRACTS = {
    "H11": """data keys mode,byteorder,payload,wire_names. mode encode/decode;
wire_names is three unique nonempty strings, documentary schema-field aliases
only (renaming them does not rename decoded output keys). byteorder must equal
little else BYTEORDER. Wire layout: uint32 id; uint32 UTF8 byte length then label
bytes (empty allowed); uint32 group count; each group uint8 tag, uint32 sample
count then signed int16 samples. All multibyte numbers little-endian, no padding.
Encode payload exact {id:integer,label:string,groups:[{tag:integer,samples:[integer]}]}.
Reject id outside [0,2^32-1], tag outside [0,255], samples outside [-32768,32767]
or length above uint32 with WIDTH. Label must encode strict UTF8 else UTF8.
Decode payload is lowercase even hex string (empty allowed); invalid hex => HEX;
short reads => TRUNCATED, invalid strict UTF8 => UTF8, remaining bytes => TRAILING.
Decode checks occur in wire order. Output encoded_hex canonical lowercase hex,
decoded exact object above, byte_count exact wire length. Roundtrip canonical.
""",
    "H12": """data keys namespace,key,parameters,remaps,expect. namespace and all
parameter paths canonical absolute paths: / or nonempty slash-separated segments,
no empty segments, '.' or '..'; keys/remap endpoints may instead be canonical
relative paths, no '~'. Bad path => PATH. parameters/remaps are objects; remap
values strings. All values finite JSON (including nested arrays/objects/null).
Reject any remap graph cycle => REMAP_CYCLE; then apply exact key remap ONCE.
Relative key: candidates namespace/key then each ancestor/key through /key;
absolute key: only itself. visited includes candidates examined up to first hit,
or all if missing. found false: resolved_path/value null, OK. expect one of
any,null,boolean,integer,number,string,array,object. number includes integers but
not bool; integer only integer tokens. Found wrong type => TYPE_MISMATCH.
Root itself can be an absolute key; empty relative key invalid. Ordering maps
never matters. Prefixing namespace, parameter paths and absolute remaps by /m
preserves hits/values and prefixes resolved names (extra root miss visits may
appear only for missing searches).
""",
    "H13": """data keys records,clock,anchor_topic,streams,maxgap. records exact
{topic:string,stamp_ns:nonnegative integer,sequence:nonnegative integer} arrays;
clock nonnegative integer array of equal length else LENGTH. maxgap nonnegative
integer; invalid time/sequence => TIME. streams unique strings excluding
anchor_topic else IDENTITY. Strict clock decrease starts epoch before that record;
equal clock same epoch; epoch zero initially. Duplicate (epoch,topic,sequence)
=> IDENTITY; sequence reset across epochs allowed, gaps do not synthesize records.
For each anchor in arrival order and each stream in list order choose SAME-epoch
sample minimizing (absolute stamp gap,stamp,original index), inclusive maxgap;
not consumed, reuse allowed. -1 absent. epochs one per record, rewinds original
indices, matches anchor rows by public stream order. Shifting stamps and clocks
of one epoch is invariant only if nonnegative, partition and gap bindings preserved.
""",
    "H14": """data keys edges,source,target. edges exact {parent,child,rotation,
translation}; strings nonempty; rotation length9 row-major, translation length3
finite numbers. Tparent_child maps child into parent. Validate all rotations
R^T R=I and det=+1 with absolute tolerance 1e-9 => BAD_ROTATION. Then reject
selfedge or repeated child (even duplicate identical edge) => MULTIPLE_PARENTS;
then directed cycle => CYCLE. No redundant-parent loop closure supported.
For query map coordinates in source into target; path lists frames source to
target. Compose/invert along unique undirected tree path. Unknown different
frames/disconnection OK reachable false, rotation/translation null,path[].
source==target always reachable identity,path[source], including unknown frames.
Global basis B means edges R'=B R B^T,t'=B t and query transforms likewise.
Rename is a bijection. No quaternion input accepted (wrong shape INVALID_INPUT).
""",
    "H15": """data exact header,duration,samples,bound,allow_seq_drop. header
{seq:integer,stamp:{sec:integer,nsec:integer},frame_id:string (empty allowed)};
duration {sec:integer,nsec:integer}, samples finite number array, bound nonnegative
integer, allow_seq_drop boolean. seq uint32 else OVERFLOW. Normalize arbitrary
integer sec/nsec via total=sec*1e9+nsec, floor divmod. Stamp total negative =>
NEGATIVE_STAMP; normalized stamp and duration seconds must fit int32 else
OVERFLOW. Then sample count > bound => ARRAY_BOUND. ROS2 omits seq: nonzero seq
without allow_seq_drop => LOSSY; seq zero may drop implicitly as default.
dropped_fields is ['header.seq'] iff nonzero seq explicitly permitted, else [].
Output header stamp sec/nanosec and frame_id; duration sec/nanosec; preserve
sample order. Negative duration normalized with nonnegative nanosec. Roundtrip
lossless subset seq=0; declaration key order irrelevant.
""",
    "H16": """data required,offered exact manifests {architecture,abi,type_name,
type_hash,symbols}. First four fields nonempty exact strings; symbols array of
nonempty strings, duplicate/empty => MANIFEST. Compare strings in fixed order
architecture,abi,type_name,type_hash; mismatches array contains unequal fields
in that order. missing_symbols sorted set(required)-set(offered). compatible
iff both arrays empty. Incompatibility OK, never INVALID; malformed manifest
=> MANIFEST (including wrong shape/types). Symbols permutable; no library loading.
""",
    "H17": """data groups,callbacks,events; groups map names to mutuallyexclusive
or reentrant; callbacks map names to declared group names. After data exact-key
and groups/callbacks object, events array type checks, require <=8 groups,
<=8 callbacks, <=128 events => RESOURCE_LIMIT, before declaration validation.
Bad declarations => DECLARATION. At most 16 distinct simultaneously live wait
edges: validate each event's shape and legality first, then a new wait edge that
would exceed 16 => RESOURCE_LIMIT before insertion; duplicate waits do not count
again. Finish/unwait release capacity. No enumeration before these checks.
Simple-cycle traversal depth <=8, with <=8! partial paths per start; never accept
a larger graph merely because its encoded bytes fit. events exact {op,callback} except wait/unwait also target. op one
of start,finish,wait,unwait. Validate each event in input order => EVENT on failure.
start requires declared inactive callback and no active peer in exclusive group;
finish requires active callback and removes all incident wait edges (owner held
until finish); wait requires distinct active endpoints, duplicate edge idempotent;
unwait requires active endpoints, existing edge, removes it. Restart after finish
allowed. Undeclared callback, wrong event keys/op invalid. Return final active
sorted, waiting sorted source/target pairs. deadlocked iff final directed wait
cycle. Enumerate simple directed cycles, rotate to least vertex, append start;
choose lexicographically least closed string-list over all such cycles (not
shortest). Renaming may change chosen cycle but not existence. Independent
callbacks do not remove any wait obligation. No fairness/starvation measurement.
""",
    "H18": """data publications nonnegative integer time array, horizon nonnegative
integer, deadline and lease positive integers. After exact keys and publications
array type, require <=128 publications => RESOURCE_LIMIT; then invalid integer
types/signs/zero periods => TIME. Next require every time/period <=1000000000000
=> RESOURCE_LIMIT; next nonmonotonic publications or horizon < lastpublication
=> ORDER. Before emitting/allocating ANY expiration arrays, count aggregate
output events by integer arithmetic: for each publication t, end is nextpub-1,
or horizon for the last; D=max(0,(end-t)//deadline), L=1 iff t+lease<=end else 0,
R=L for nonfinal intervals else 0. Require sum(D+L+R)<=4096 => RESOURCE_LIMIT.
This includes deadline_misses, liveliness_lost AND recovered together, not just
each separately. Each event time <=horizon<=10^12; combined numeric arrays need
at most 57350 compact JSON bytes (4096 entries, <=13 digits each), plus fixed
envelope/keys. Equality at every resource ceiling is supported. Empty trace:
all arrays empty, alive false. First publication initializes timers. Before each
next publication emit deadline expirations strictly earlier than its time;
repeat deadline by period since latest publication. Liveliness emits once at
lastpub+lease strictly before next publication; next publication then reports
recovered iff lost, and resets timers. Publication at exact expiration prevents
it (equal-time publications processed first). After last publication emit through
horizon inclusive. alive iff a publication occurred and no final loss. Shift
all publication/horizon times equally: event times shift, identities unchanged.
""",
    "H19": """data parameters,updates,expected_revision,revision,constraints.
parameters/updates map nonempty names to finite JSON scalars (including null);
revisions nonnegative integers. constraints exact {id,left,op,right}: unique
nonempty id; left/right names in merged map; op eq,ne,lt,le,gt,ge.
Malformed declarations, missing referenced name or invalid operand types =>
PREDICATE. Validate declarations (keys/op/names) before stale check; stale returns
OK committedfalse, unchanged state/revision, failed_constraints[],rejection
STALE_REVISION WITHOUT evaluating operand types or predicates. At current
revision, eq/ne accept identical JSON scalar types, int/float same numeric type;
other mixed types invalid; order comparisons numbers only (not bool). All
operand types validated before evaluating predicates. Evaluate all constraints
on merged state, failed IDs declaration order. Any false: unchanged state and
revision, rejection CONSTRAINT. Otherwise commit entire merge, increment revision
once even empty update, rejection NONE. Keys permutable. No eval or code.
""",
    "H20": """data events exact {op,id} array; strings nonempty, ops submit,accept,
start,cancel_request,cancel_ack,result_success,result_cancelled,result_failure.
Unknown op/wrong shape => EVENT even after terminal. Process in order: submit
new unique id -> SUBMITTED; duplicate submit always EVENT even after terminal.
Unknown id => EVENT. accept SUBMITTED->ACCEPTED; start ACCEPTED->RUNNING;
cancel_request ACCEPTED/RUNNING->CANCEL_PENDING sets cancel_requested true;
cancel_ack CANCEL_PENDING->CANCEL_ACKED, not terminal. Repeated accept in
ACCEPTED, start in RUNNING, cancel_request in CANCEL_PENDING/CANCEL_ACKED,
cancel_ack in CANCEL_ACKED are idempotent, ignored index recorded. Other
nonterminal transitions EVENT. result_success/result_failure legal from ACCEPTED,
RUNNING,CANCEL_PENDING,CANCEL_ACKED; result_cancelled only CANCEL_ACKED.
First legal terminal wins, storing original event index; later recognized events
except submit ignored and indexed. Goals sorted id; terminal_event null until
terminal; states SUCCEEDED,CANCELLED,FAILED plus five nonterminal states above.
Renaming ids equivariant. Inserting acknowledged duplicates preserves goal states,
but terminal/ignored indices must be adjusted by the insertion index map.
""",
}
ERROR_CODES = {
    "H11": ("BYTEORDER", "WIDTH", "UTF8", "HEX", "TRUNCATED", "TRAILING"),
    "H12": ("PATH", "REMAP_CYCLE", "TYPE_MISMATCH"),
    "H13": ("LENGTH", "TIME", "IDENTITY"),
    "H14": ("BAD_ROTATION", "MULTIPLE_PARENTS", "CYCLE"),
    "H15": ("OVERFLOW", "NEGATIVE_STAMP", "ARRAY_BOUND", "LOSSY"),
    "H16": ("MANIFEST",), "H17": ("DECLARATION", "EVENT"),
    "H18": ("TIME", "ORDER"), "H19": ("PREDICATE",), "H20": ("EVENT",),
}
PLANNED_INVALID = dict(zip(FAMILIES, ("TRUNCATED", "TYPE_MISMATCH", "LENGTH", "MULTIPLE_PARENTS", "LOSSY", "MANIFEST", "EVENT", "ORDER", "PREDICATE", "EVENT"), strict=False))
RELATIONS = {
    "H11": "Rename wire_names only: all values invariant; encode/decode roundtrip canonical.",
    "H12": "Move parameter subtree and namespace under /m: value/found invariant, resolved_path and successful visited paths prefixed /m.",
    "H13": "Shift only second epoch by +1000: epochs/matches/rewinds invariant.",
    "H14": "Rename frames r_ and basis B=diag(-1,-1,1): reachable invariant, path renamed, R'=BRB^T,t'=Bt.",
    "H15": "Reverse declaration key insertion order: all values invariant; seq=0 lossless roundtrip.",
    "H16": "Reverse offered/required symbol lists: all values invariant.",
    "H17": "Order-preserving callback rename r_: active/waiting/cycle renamed, deadlocked invariant.",
    "H18": "Shift publications/horizon +1000: all event times +1000, alive invariant.",
    "H19": "Reverse update key order: all values invariant.",
    "H20": "Rename all goal ids r_: goals renamed, states/indices invariant.",
}


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def arr(item: Any, n: int | None = None) -> dict[str, Any]:
    out = {"type": "array", "items": item}
    if n is not None:
        out.update(minItems=n, maxItems=n)
    return out


I = {"type": "integer"}  # noqa: E741 - Preserve the public integer-schema alias I.
N = {"type": "number"}
S = {"type": "string"}
B = {"type": "boolean"}
NULL = {"type": "null"}
STAMP = obj(sec=I, nanosec=I)
DECODED = obj(id=I, label=S, groups=arr(obj(tag=I, samples=arr(I))))
RESULT_VALUE_SCHEMAS = {
    "H11": obj(encoded_hex={"type": "string", "pattern": "^(?:[0-9a-f]{2})*$"}, decoded=DECODED, byte_count=I),
    "H12": obj(resolved_path={"type": ["string", "null"]}, found=B, value={}, visited=arr(S)),
    "H13": obj(epochs=arr(I), matches=arr(arr(I)), rewinds=arr(I)),
    "H14": obj(reachable=B, rotation={"oneOf": [arr(N, 9), NULL]}, translation={"oneOf": [arr(N, 3), NULL]}, path=arr(S)),
    "H15": obj(header=obj(stamp=STAMP, frame_id=S), duration=STAMP, samples=arr(N), dropped_fields=arr(S)),
    "H16": obj(compatible=B, missing_symbols=arr(S), mismatches=arr(S)),
    "H17": obj(active=arr(S), waiting=arr(arr(S, 2)), deadlocked=B, cycle=arr(S)),
    "H18": obj(deadline_misses=arr(I), liveliness_lost=arr(I), recovered=arr(I), alive=B),
    "H19": obj(committed=B, revision=I, parameters={"type": "object", "additionalProperties": {"type": ["string", "number", "boolean", "null"]}}, failed_constraints=arr(S), rejection={"enum": ["NONE", "STALE_REVISION", "CONSTRAINT"]}),
    "H20": obj(goals=arr(obj(id=S, state={"enum": ["SUBMITTED", "ACCEPTED", "RUNNING", "CANCEL_PENDING", "CANCEL_ACKED", "SUCCEEDED", "CANCELLED", "FAILED"]}, cancel_requested=B, terminal_event={"type": ["integer", "null"]})), ignored=arr(I)),
}


# Machine-readable supported-domain constraints complement COMMON's byte,
# aggregate-node and aggregate-event bounds; those require procedural checks.
RESULT_VALUE_SCHEMAS["H17"]["properties"]["active"]["maxItems"] = H17_CALLBACKS
RESULT_VALUE_SCHEMAS["H17"]["properties"]["waiting"]["maxItems"] = H17_EDGES
RESULT_VALUE_SCHEMAS["H17"]["properties"]["cycle"]["maxItems"] = H17_CALLBACKS + 1
for _name in ("deadline_misses", "liveliness_lost", "recovered"):
    RESULT_VALUE_SCHEMAS["H18"]["properties"][_name]["maxItems"] = H18_OUTPUT_EVENTS

_NAME = {"type": "string", "minLength": 1, "maxLength": INPUT_STRING_BYTES}
_TIME = {"type": "integer", "minimum": 0, "maximum": H18_TIME}
_PERIOD = {"type": "integer", "minimum": 1, "maximum": H18_TIME}
INPUT_DATA_SCHEMAS = {
    "H17": obj(
        groups={"type": "object", "maxProperties": H17_GROUPS,
                "propertyNames": _NAME,
                "additionalProperties": {"enum": ["mutuallyexclusive", "reentrant"]}},
        callbacks={"type": "object", "maxProperties": H17_CALLBACKS,
                   "propertyNames": _NAME, "additionalProperties": _NAME},
        events={"type": "array", "maxItems": H17_EVENTS, "items": {"oneOf": [
            obj(op={"enum": ["start", "finish"]}, callback=_NAME),
            obj(op={"enum": ["wait", "unwait"]}, callback=_NAME, target=_NAME),
        ]}},
    ),
    "H18": obj(
        publications={"type": "array", "maxItems": H18_PUBLICATIONS, "items": _TIME},
        horizon=_TIME, deadline=_PERIOD, lease=_PERIOD,
    ),
}


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
                "role": self.role, "seed": self.seed, "scope": SCOPE,
                "CAP80_ID": CAP80_IDS[self.family],
                "input_hashes": {k: sha(v) for k, v in self.files},
                "contract_sha256": sha((COMMON + CONTRACTS[self.family]).encode()),
                "companion_of": f"{self.family}-subject-{self.seed}" if self.role == "companion" else None,
                "relation": RELATIONS[self.family], "planned_invalid_code": PLANNED_INVALID[self.family]}

    def prompt(self) -> str:
        invalid = obj(status={"const": "INVALID"}, code={"enum": ["INVALID_INPUT", "INVALID_JSON", "SIZE_LIMIT", "RESOURCE_LIMIT", *ERROR_CODES[self.family]]}, values=NULL)
        ok = obj(status={"const": "OK"}, code={"const": "OK"}, values=deepcopy(RESULT_VALUE_SCHEMAS[self.family]))
        schema = obj(scope={"const": SCOPE}, result={"oneOf": [ok, invalid]}, input_sha256={"type": "string", "pattern": "^[0-9a-f]{64}$"}, reason={"type": "string", "pattern": "\\S"})
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        input_schema = obj(version={"const": VERSION}, family={"const": self.family},
                           data=deepcopy(INPUT_DATA_SCHEMAS.get(self.family, {"type": "object"})))
        input_schema["$schema"] = schema["$schema"]
        return (COMMON + CONTRACTS[self.family] + "\nS/I/M relation: " + RELATIONS[self.family]
                + "\nSupported input Draft202012 schema (prose also normative; invalid inputs return public codes):\n"
                + json.dumps(input_schema, sort_keys=True)
                + "\nAnswer Draft202012 schema:\n" + json.dumps(schema, sort_keys=True))


def _base(f: str, k: int) -> dict[str, Any]:
    ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if f == "H11":
        return {"mode": "encode", "byteorder": "little", "wire_names": ["id", "label", "groups"], "payload": {"id": k, "label": "µ机器人", "groups": [{"tag": 255, "samples": [-32768, k, 32767]}, {"tag": 0, "samples": []}]}}
    if f == "H12":
        return {"namespace": "/a/b", "key": "alias", "parameters": {"/a/b/gain": k, "/a/gain": 99, "/gain": 101}, "remaps": {"alias": "gain"}, "expect": "integer"}
    if f == "H13":
        return {"records": [{"topic": t, "stamp_ns": s, "sequence": q} for t, s, q in [("x", 10, 1), ("a", 11, 1), ("x", 12, 2), ("x", 10, 1), ("a", 11, 1), ("y", 11, 1)]], "clock": [10, 11, 12, 1, 2, 2], "anchor_topic": "a", "streams": ["x", "y"], "maxgap": 1}
    if f == "H14":
        return {"edges": [{"parent": "world", "child": "a", "rotation": ident, "translation": [k, 2, 0]}, {"parent": "a", "child": "b", "rotation": [0, -1, 0, 1, 0, 0, 0, 0, 1], "translation": [0, 1, 3]}], "source": "b", "target": "world"}
    if f == "H15":
        return {"header": {"seq": 0, "stamp": {"sec": k, "nsec": 1000000001}, "frame_id": "map"}, "duration": {"sec": 0, "nsec": -1}, "samples": [k, -2.5, 0], "bound": 3, "allow_seq_drop": False}
    if f == "H16":
        r = {"architecture": "aarch64", "abi": "v2", "type_name": "pkg/msg/T", "type_hash": f"hash-{k}", "symbols": ["create", "destroy"]}
        o = deepcopy(r)
        o["symbols"] = ["create", "extra"]
        o["abi"] = "v1"
        return {"required": r, "offered": o}
    if f == "H17":
        return {"groups": {"g": "reentrant", "solo": "mutuallyexclusive"}, "callbacks": {"a": "g", "b": "g", "c": "solo"}, "events": [{"op": "start", "callback": "a"}, {"op": "start", "callback": "b"}, {"op": "wait", "callback": "a", "target": "b"}, {"op": "wait", "callback": "b", "target": "a"}]}
    if f == "H18":
        return {"publications": [k, k + 5, k + 16], "horizon": k + 30, "deadline": 5, "lease": 7}
    if f == "H19":
        return {"parameters": {"lo": 0, "hi": k + 10, "on": True}, "updates": {"lo": k, "hi": k + 1}, "expected_revision": 2, "revision": 2, "constraints": [{"id": "ordered", "left": "lo", "op": "lt", "right": "hi"}]}
    return {"events": [{"op": o, "id": "g"} for o in ("submit", "accept", "start", "cancel_request", "cancel_ack", "result_success", "result_cancelled")]}


def make_instance(family: str, role: str, seed: int) -> Instance:
    if family not in FAMILIES or role not in ROLES or type(seed) is not int:
        raise ValueError("family/role/seed")
    k = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4], "little") % 100 + 1
    p = _base(family, k)
    if role == "invalid":
        if family == "H11":
            p.update(mode="decode", payload="01000000")
        elif family == "H12":
            p["expect"] = "string"
        elif family == "H13":
            p["clock"].pop()
        elif family == "H14":
            p["edges"].append(deepcopy(p["edges"][0]))
        elif family == "H15":
            p["header"]["seq"] = 1
        elif family == "H16":
            p["offered"]["symbols"] = ["create", "create"]
        elif family == "H17":
            p["events"].insert(0, {"op": "finish", "callback": "a"})
        elif family == "H18":
            p["publications"] = [k + 1, k]
        elif family == "H19":
            p["constraints"][0]["op"] = "eval"
        else:
            p["events"].insert(1, {"op": "submit", "id": "g"})
    elif role == "companion":
        if family == "H11":
            p["wire_names"] = ["renamed_id", "renamed_label", "renamed_groups"]
        elif family == "H12":
            p["namespace"] = "/m" + p["namespace"]
            p["parameters"] = {"/m" + key: value for key, value in p["parameters"].items()}
        elif family == "H13":
            for i in range(3, len(p["records"])):
                p["records"][i]["stamp_ns"] += 1000
                # Clock epoch bindings intentionally remain unchanged.
        elif family == "H14":
            signs = [-1, -1, 1]
            for e in p["edges"]:
                e["parent"], e["child"] = "r_" + e["parent"], "r_" + e["child"]
                e["rotation"] = [v * signs[i // 3] * signs[i % 3] for i, v in enumerate(e["rotation"])]
                e["translation"] = [v * signs[i] for i, v in enumerate(e["translation"])]
            p["source"], p["target"] = "r_" + p["source"], "r_" + p["target"]
        elif family == "H15":
            p["header"] = dict(reversed(list(p["header"].items())))
        elif family == "H16":
            for m in p.values():
                m["symbols"].reverse()
        elif family == "H17":
            p["callbacks"] = {"r_" + c: g for c, g in p["callbacks"].items()}
            for e in p["events"]:
                e["callback"] = "r_" + e["callback"]
                if "target" in e:
                    e["target"] = "r_" + e["target"]
        elif family == "H18":
            p["publications"] = [t + 1000 for t in p["publications"]]
            p["horizon"] += 1000
        elif family == "H19":
            p["updates"] = dict(reversed(list(p["updates"].items())))
        else:
            for e in p["events"]:
                e["id"] = "r_" + e["id"]
    raw = encode({"version": VERSION, "family": family, "data": p})
    # Preserve deliberately permuted declaration order in companion bytes.
    if role == "companion" and family in ("H15", "H19"):
        raw = (json.dumps({"version": VERSION, "family": family, "data": p}, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    return Instance(f"{family}-{role}-{seed}", family, role, seed, (("input.json", raw),))


def make_plan(seed: int) -> tuple[Instance, ...]:
    return tuple(make_instance(f, r, seed) for f in FAMILIES for r in ROLES)
