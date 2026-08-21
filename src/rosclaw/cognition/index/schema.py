"""Ecosystem Index schema（PR-N3，N 总纲 §4.1 第三层）。

SQLite + FTS5；关系与 manifest 用明确表结构（不引入向量库）。
- meta：索引版本 + 产品指纹（zoo 内容摘要 + 版本/commit）；
- entities：实体（kind/canonical_id/path/source/digest/quality/
  payload_json 关系）；
- entities_fts：FTS5 全文（name+canonical_id+text）。
"""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  canonical_id TEXT NOT NULL,
  name TEXT NOT NULL,
  path TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '',
  digest TEXT NOT NULL DEFAULT '',
  quality TEXT NOT NULL DEFAULT '',
  payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
  name, canonical_id, text,
  content='entities', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
  INSERT INTO entities_fts(rowid, name, canonical_id, text)
  VALUES (new.rowid, new.name, new.canonical_id,
          json_extract(new.payload_json, '$.text'));
END;

CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
  INSERT INTO entities_fts(entities_fts, rowid, name, canonical_id, text)
  VALUES ('delete', old.rowid, old.name, old.canonical_id,
          json_extract(old.payload_json, '$.text'));
END;
"""

INDEX_VERSION = "1"
