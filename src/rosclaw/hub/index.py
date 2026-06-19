"""SQLite-backed catalog index for ROSClaw Hub assets."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from rosclaw.firstboot.workspace import detect_platform
from rosclaw.hub.cache import HubCache

_CATALOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS catalog (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    description TEXT,
    tags TEXT,
    publisher_id TEXT,
    publisher_display_name TEXT,
    trust_level TEXT,
    visibility_scope TEXT,
    lifecycle_status TEXT,
    deprecated INTEGER,
    yanked INTEGER,
    os TEXT,
    arch TEXT,
    python_requires TEXT,
    ros_distributions TEXT,
    eurdf_profiles TEXT,
    body_kinds TEXT,
    required_devices TEXT,
    runtime_features TEXT,
    license_spdx TEXT,
    allowed_usage TEXT,
    sandbox_required INTEGER,
    network_isolation_recommended INTEGER,
    manifest_digest TEXT,
    manifest_url TEXT,
    size_bytes INTEGER,
    raw_json TEXT NOT NULL
);
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_catalog_type ON catalog(type);
CREATE INDEX IF NOT EXISTS idx_catalog_namespace ON catalog(namespace);
CREATE INDEX IF NOT EXISTS idx_catalog_name ON catalog(name);
CREATE INDEX IF NOT EXISTS idx_catalog_trust ON catalog(trust_level);
CREATE INDEX IF NOT EXISTS idx_catalog_license ON catalog(license_spdx);
"""

_FTS_COLUMNS = [
    "type",
    "namespace",
    "name",
    "version",
    "title",
    "summary",
    "description",
    "tags",
    "publisher_id",
    "trust_level",
    "license_spdx",
]


def _sanitize_registry(name: str) -> str:
    """Convert a registry URL into a safe directory name."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).strip("_") or "default"


def _entry_id(entry: dict[str, Any]) -> str:
    asset = entry.get("asset", {})
    return f"{asset.get('type')}:{asset.get('namespace')}:{asset.get('name')}:{asset.get('version')}"


def _json_list(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value)
    if value is None:
        return "[]"
    return json.dumps([value])


def _escape_fts_query(query: str) -> str:
    """Escape a free-text query for safe FTS5 MATCH usage.

    Reserved operators such as ``NOT``/``AND``/``OR`` are neutralised by
    double-quoting each term; a lone ``*`` is also quoted.
    """
    escaped: list[str] = []
    for term in query.split():
        term = term.replace('"', '""')
        if not term:
            continue
        escaped.append(f'"{term}"')
    return " ".join(escaped) if escaped else '""'


class CatalogIndex:
    """Local SQLite catalog index for fast search and filtering.

    The index is scoped per registry so that multiple registries can be cached
    side-by-side without collision.
    """

    def __init__(self, registry: str, cache: HubCache | None = None) -> None:
        self.registry = registry
        self.cache = cache or HubCache()
        self.db_path = self.cache.indexes_dir / _sanitize_registry(registry) / "catalog.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(_CATALOG_TABLE_SQL)
            conn.executescript(_INDEX_SQL)
            try:
                fts_cols = ", ".join(_FTS_COLUMNS)
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5({fts_cols});"
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                # FTS5 may be unavailable in some builds; fall back to LIKE.
                self._fts_enabled = False
            conn.commit()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove every indexed entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM catalog;")
            if self._fts_enabled:
                conn.execute("DELETE FROM catalog_fts;")
            conn.commit()

    def index_entries(self, entries: list[dict[str, Any]]) -> int:
        """Insert or replace catalog entries in the index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            for entry in entries:
                row = self._entry_to_row(entry)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO catalog
                    (id, type, namespace, name, version, title, summary, description, tags,
                     publisher_id, publisher_display_name, trust_level, visibility_scope,
                     lifecycle_status, deprecated, yanked, os, arch, python_requires,
                     ros_distributions, eurdf_profiles, body_kinds, required_devices,
                     runtime_features, license_spdx, allowed_usage, sandbox_required,
                     network_isolation_recommended, manifest_digest, manifest_url, size_bytes,
                     raw_json)
                    VALUES
                    (:id, :type, :namespace, :name, :version, :title, :summary, :description, :tags,
                     :publisher_id, :publisher_display_name, :trust_level, :visibility_scope,
                     :lifecycle_status, :deprecated, :yanked, :os, :arch, :python_requires,
                     :ros_distributions, :eurdf_profiles, :body_kinds, :required_devices,
                     :runtime_features, :license_spdx, :allowed_usage, :sandbox_required,
                     :network_isolation_recommended, :manifest_digest, :manifest_url, :size_bytes,
                     :raw_json)
                    """,
                    row,
                )
                if self._fts_enabled:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO catalog_fts
                        (type, namespace, name, version, title, summary, description, tags,
                         publisher_id, trust_level, license_spdx)
                        VALUES
                        (:type, :namespace, :name, :version, :title, :summary, :description, :tags,
                         :publisher_id, :trust_level, :license_spdx)
                        """,
                        row,
                    )
            conn.commit()
        return len(entries)

    def _entry_to_row(self, entry: dict[str, Any]) -> dict[str, Any]:
        asset = entry.get("asset", {})
        pub = entry.get("publisher", {})
        vis = entry.get("visibility", {})
        life = entry.get("lifecycle", {})
        compat = entry.get("compatibility", {})
        robot = compat.get("robot", {})
        hw = compat.get("hardware", {})
        py = compat.get("python", {})
        ros = compat.get("ros", {})
        lic = entry.get("license", {})
        dr = entry.get("data_rights", {})
        sec = entry.get("security", {})
        return {
            "id": _entry_id(entry),
            "type": asset.get("type"),
            "namespace": asset.get("namespace"),
            "name": asset.get("name"),
            "version": asset.get("version"),
            "title": asset.get("title"),
            "summary": asset.get("summary"),
            "description": asset.get("description"),
            "tags": _json_list(asset.get("tags")),
            "publisher_id": pub.get("id"),
            "publisher_display_name": pub.get("display_name"),
            "trust_level": pub.get("trust_level", "unknown"),
            "visibility_scope": vis.get("scope", "public"),
            "lifecycle_status": life.get("status", "stable"),
            "deprecated": 1 if life.get("deprecated") else 0,
            "yanked": 1 if life.get("yanked") else 0,
            "os": _json_list(compat.get("os")),
            "arch": _json_list(compat.get("arch")),
            "python_requires": py.get("requires"),
            "ros_distributions": _json_list(ros.get("distributions", [])),
            "eurdf_profiles": _json_list(robot.get("eurdf_profiles", [])),
            "body_kinds": _json_list(robot.get("body_kinds", [])),
            "required_devices": _json_list(hw.get("required_devices", [])),
            "runtime_features": _json_list(compat.get("runtime_features", [])),
            "license_spdx": lic.get("spdx"),
            "allowed_usage": _json_list(dr.get("allowed_usage", [])),
            "sandbox_required": 1 if sec.get("sandbox_required") else 0,
            "network_isolation_recommended": 1 if sec.get("network_isolation_recommended") else 0,
            "manifest_digest": entry.get("manifest_digest"),
            "manifest_url": entry.get("manifest_url"),
            "size_bytes": entry.get("size_bytes"),
            "raw_json": json.dumps(entry, ensure_ascii=False),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str | None = None,
        *,
        asset_type: str | None = None,
        namespace: str | None = None,
        official: bool = False,
        licenses: list[str] | None = None,
        robot: str | None = None,
        compatible: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search the catalog with text and structured filters."""
        filters: list[str] = []
        params: dict[str, Any] = {}

        if asset_type:
            filters.append("c.type = :type")
            params["type"] = asset_type
        if namespace:
            filters.append("c.namespace = :namespace")
            params["namespace"] = namespace
        if official:
            filters.append("c.trust_level = 'official'")
        if licenses:
            placeholders = ", ".join(f":license_{i}" for i in range(len(licenses)))
            filters.append(f"c.license_spdx IN ({placeholders})")
            for i, lic in enumerate(licenses):
                params[f"license_{i}"] = lic
        if robot:
            filters.append(
                "(c.eurdf_profiles LIKE :robot OR c.body_kinds LIKE :robot OR c.tags LIKE :robot)"
            )
            params["robot"] = f"%{robot}%"
        if compatible:
            filters.append(self._compatible_filter(params))

        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        if query and self._fts_enabled:
            params["query"] = _escape_fts_query(query)
            filter_clause = f"AND {' AND '.join(filters)}" if filters else ""
            sql = f"""
                SELECT c.raw_json FROM catalog c
                JOIN catalog_fts f ON c.rowid = f.rowid
                WHERE catalog_fts MATCH :query
                {filter_clause}
                LIMIT :limit
            """
        elif query:
            params["query"] = f"%{query}%"
            like_filters = [
                "c.title LIKE :query",
                "c.summary LIKE :query",
                "c.description LIKE :query",
                "c.tags LIKE :query",
                "c.name LIKE :query",
            ]
            conjunction = "AND" if filters else "WHERE"
            sql = f"""
                SELECT c.raw_json FROM catalog c
                {where}
                {conjunction} ({' OR '.join(like_filters)})
                LIMIT :limit
            """
        else:
            sql = f"SELECT c.raw_json FROM catalog c {where} LIMIT :limit"

        params["limit"] = limit

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [json.loads(row["raw_json"]) for row in rows]

    def _compatible_filter(self, params: dict[str, Any]) -> str:
        """Return a SQL fragment filtering for the current platform."""
        platform = detect_platform()
        current_os = platform.os.lower()
        current_arch = platform.arch
        params["current_os"] = f'%{current_os}%'
        params["current_arch"] = f'%{current_arch}%'
        return "(c.os LIKE :current_os AND c.arch LIKE :current_arch)"

    def count(self) -> int:
        """Return the number of indexed entries."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM catalog;").fetchone()
        return row[0] if row else 0

    def get(self, ref: str) -> dict[str, Any] | None:
        """Return a single catalog entry by its canonical reference."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT raw_json FROM catalog WHERE id = ?", (ref,)
            ).fetchone()
        return json.loads(row["raw_json"]) if row else None
