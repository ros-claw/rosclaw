# DF-18 — Runtime Config v2 native consumption (implementation note)

Phase-II P0 per `seekdb优化v2.md` §15–§16. Closes gap P0-3: Config v2
existed as file format (DF-02) but the Runtime still read legacy flat
fields and dict-shaped sections.

## What landed

### §15.1 typed models — `src/rosclaw/config/models.py` (new)

`StructuredStoreConfig` (backend/path/dsn/pool_size),
`RetrievalStoreConfig` (enabled/backend/mode/path/host/port/database),
`OutboxConfig` (enabled/path/batch_size/flush_interval_sec/max_records),
`ArtifactStoreConfig`, `StorageConfig` (the four above),
`KnowledgeConfig` (enabled/mode/store_mode/store_path/url/api_key/
timeout/curated_registry_enabled), `EvolutionConfig`
(enabled/require_human_approval/allow_code_patch/
trigger_failure_threshold), `DarwinConfig` (enabled/seeds/episodes).
All with `from_dict`.

§16 honored: the structured default path stays
`~/.rosclaw/data/memory/knowledge.sqlite` — no data move in this PR.

### §15.4 legacy compat — `src/rosclaw/config/compat.py` (new)

The only place pre-DF-18 shapes are interpreted.  RuntimeConfig's
`__post_init__` calls the four `normalize_*_config()` functions exactly
once.  Precedence: an explicitly passed typed model (or an explicit dict
key) always wins; flat legacy fields (`seekdb_backend/path/url`,
`know_store_mode/path`, `enable_auto`, `enable_darwin`,
`enable_knowledge`, dict `storage` flat keys) only fill slots the caller
never set.  DEPRECATED CONFIG is logged once per legacy key actually
applied.

### RuntimeConfig (core/runtime.py)

- New typed fields: `storage` (StorageConfig | dict | None),
  `knowledge`, `evolution`, and `darwin` (DarwinConfig | dict | None).
- `__post_init__` normalizes all four, then **mirrors the typed values
  back onto the legacy flat fields** so pre-DF-18 readers (tests,
  episode recorder, external constructors) keep observing consistent
  values — same philosophy as DF-02's `_mirror_legacy_sections`.

### §15.5 Runtime reads typed only

All internal reads switched: `storage.structured.backend/path/dsn/
pool_size`, `storage.retrieval.enabled/path`, `storage.outbox.*`,
`knowledge.enabled/mode/store_mode/store_path/url/api_key/timeout/
curated_registry_enabled`, `evolution.enabled/trigger_failure_threshold`,
`darwin.enabled/seeds/episodes`.  A source-discipline test greps
runtime.py for the forbidden legacy reads.

### §15.6 bare `seekdb` eliminated

The three `seekdb = getattr(self._memory, ...)` compat locals are gone;
modules receive `data_plane.structured_store` explicitly (Memory-held
client as the no-data-plane fallback — same object when a data plane
exists).  `grep -n "seekdb =" runtime.py` → 0 results (pinned by test).

### loader — `src/rosclaw/config/loader.py`

`load_typed_configs(home)` → (storage, knowledge, evolution, darwin)
from rosclaw.yaml for CLI/entrypoint callers, reusing FirstbootConfig's
file-format normalization.

## Tests — `tests/config/test_typed_config.py` (12 tests)

Model defaults/from_dict; legacy flat fold; dict-section fold; enable
flags; darwin dict; typed-wins precedence; mirror-back; plain-construction
defaults unchanged; runtime source discipline (§15.5/§15.6 greps);
loader on legacy yaml + missing file.

Full affected suites (core/firstboot/darwin/knowledge/auto/memory-v2/
storage/e2e): 594 passed, 1 pre-existing host-env failure
(`test_service_discovery` — local rosclaw-know predates `rosclaw_know.store`;
fails identically on main; CI installs the newer package).
