"""Labeled memory store on Postgres + pgvector (SPEC §7).

Every query that touches tenant data filters `tenant = %(tenant)s` before any vector
ordering; tests/unit/test_store_tenant_filter.py enforces this on this file's source.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from smriti.gateway.labels import CrossTenantError, Label, Memory, Sensitivity, ToolCall, Trust

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    ts: float
    tenant: str
    op: str
    memory_id: str | None
    detail: dict


class Store(Protocol):
    def insert(self, memory: Memory, embedding: list[float] | None) -> None: ...
    def get(self, tenant: str, memory_id: str) -> Memory | None: ...
    def search(self, tenant: str, embedding: list[float], k: int) -> list[Memory]: ...
    def append_ledger(self, tenant: str, op: str, memory_id: str | None, detail: dict) -> None: ...
    def ledger(self, tenant: str) -> list[LedgerEntry]: ...
    def log_tool_call(self, call: ToolCall) -> None: ...
    def tool_log(self, episode_id: str) -> list[ToolCall]: ...
    def reset(self) -> None: ...


_MEMORY_COLUMNS = """m.id::text, m.tenant, m.text, m.source, m.trust, m.sensitivity, m.flags,
       m.session_id, extract(epoch FROM m.created_at),
       ARRAY(SELECT e.parent::text FROM edges e WHERE e.child = m.id)"""

COUNT_PARENTS = """SELECT count(*) FROM memories
WHERE tenant = %(tenant)s AND id = ANY(%(parents)s::uuid[])"""

INSERT_MEMORY = """INSERT INTO memories
  (id, tenant, text, source, trust, sensitivity, flags, session_id, created_at, embedding)
VALUES (%(id)s, %(tenant)s, %(text)s, %(source)s, %(trust)s, %(sensitivity)s, %(flags)s,
        %(session_id)s, to_timestamp(%(created_at)s), %(embedding)s::vector)"""

INSERT_EDGE = "INSERT INTO edges (child, parent) VALUES (%(child)s, %(parent)s)"

GET_MEMORY = f"""SELECT {_MEMORY_COLUMNS}
FROM memories m
WHERE m.tenant = %(tenant)s AND m.id = %(id)s"""

SEARCH = f"""SELECT {_MEMORY_COLUMNS}
FROM memories m
WHERE m.tenant = %(tenant)s
  AND m.embedding IS NOT NULL
  AND NOT ('quarantined' = ANY(m.flags))
ORDER BY m.embedding <=> %(query)s::vector
LIMIT %(k)s"""

APPEND_LEDGER = """INSERT INTO ledger (tenant, op, memory_id, detail)
VALUES (%(tenant)s, %(op)s, %(memory_id)s, %(detail)s)"""

READ_LEDGER = """SELECT seq, extract(epoch FROM ts), tenant, op, memory_id::text, detail
FROM ledger
WHERE tenant = %(tenant)s
ORDER BY seq"""

INSERT_TOOL_CALL = """INSERT INTO tool_log
  (episode_id, session_id, step, tool, args, tainted, decision, memory_ids_in_context)
VALUES (%(episode_id)s, %(session_id)s, %(step)s, %(tool)s, %(args)s, %(tainted)s,
        %(decision)s, %(memory_ids)s::uuid[])"""

READ_TOOL_LOG = """SELECT episode_id, session_id, step, tool, args, tainted, decision,
       memory_ids_in_context::text[]
FROM tool_log
WHERE episode_id = %(episode_id)s
ORDER BY seq"""

RESET = "TRUNCATE memories, edges, ledger, tool_log RESTART IDENTITY CASCADE"


def _vector(embedding: list[float] | None) -> str | None:
    return None if embedding is None else "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _row_to_memory(row: tuple) -> Memory:
    mid, tenant, text, source, trust, sens, flags, session_id, created, parents = row
    return Memory(
        id=mid,
        text=text,
        label=Label(source, tenant, Trust(trust), Sensitivity(sens), frozenset(parents)),
        flags=frozenset(flags),
        session_id=session_id,
        created_at=float(created),
    )


class PgStore:
    def __init__(self, dsn: str, schema: str, embed_dim: int):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", schema):
            raise ValueError(f"bad schema name: {schema!r}")
        self.schema = schema
        self.embed_dim = int(embed_dim)
        self._conn = psycopg.connect(
            dsn, autocommit=True, options=f"-c search_path={schema},public"
        )

    def init_schema(self) -> None:
        ddl = SCHEMA_SQL.read_text().replace("EMBED_DIM", str(self.embed_dim))
        with self._conn.transaction():
            self._conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(self.schema)))
            self._conn.execute(ddl)

    def drop_schema(self) -> None:
        self._conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
            sql.Identifier(self.schema)))

    def close(self) -> None:
        self._conn.close()

    def insert(self, memory: Memory, embedding: list[float] | None) -> None:
        label = memory.label
        parents = sorted(label.parents)
        with self._conn.transaction():
            if parents:
                (found,) = self._conn.execute(
                    COUNT_PARENTS, {"tenant": label.tenant, "parents": parents}).fetchone()
                if found != len(parents):
                    raise CrossTenantError(label.tenant, "parent outside tenant")
            self._conn.execute(INSERT_MEMORY, {
                "id": memory.id, "tenant": label.tenant, "text": memory.text,
                "source": label.source, "trust": int(label.trust),
                "sensitivity": int(label.sensitivity), "flags": sorted(memory.flags),
                "session_id": memory.session_id, "created_at": memory.created_at,
                "embedding": _vector(embedding),
            })
            for parent in parents:
                self._conn.execute(INSERT_EDGE, {"child": memory.id, "parent": parent})

    def get(self, tenant: str, memory_id: str) -> Memory | None:
        row = self._conn.execute(GET_MEMORY, {"tenant": tenant, "id": memory_id}).fetchone()
        return _row_to_memory(row) if row else None

    def search(self, tenant: str, embedding: list[float], k: int) -> list[Memory]:
        rows = self._conn.execute(
            SEARCH, {"tenant": tenant, "query": _vector(embedding), "k": k}).fetchall()
        return [_row_to_memory(r) for r in rows]

    def append_ledger(self, tenant: str, op: str, memory_id: str | None, detail: dict) -> None:
        self._conn.execute(APPEND_LEDGER, {
            "tenant": tenant, "op": op, "memory_id": memory_id, "detail": Jsonb(detail)})

    def ledger(self, tenant: str) -> list[LedgerEntry]:
        rows = self._conn.execute(READ_LEDGER, {"tenant": tenant}).fetchall()
        return [LedgerEntry(seq, float(ts), t, op, mid, detail)
                for seq, ts, t, op, mid, detail in rows]

    def log_tool_call(self, call: ToolCall) -> None:
        self._conn.execute(INSERT_TOOL_CALL, {
            "episode_id": call.episode_id, "session_id": call.session_id, "step": call.step,
            "tool": call.tool, "args": Jsonb(call.args), "tainted": call.tainted,
            "decision": call.decision, "memory_ids": list(call.memory_ids_in_context),
        })

    def tool_log(self, episode_id: str) -> list[ToolCall]:
        rows = self._conn.execute(READ_TOOL_LOG, {"episode_id": episode_id}).fetchall()
        return [ToolCall(*row[:7], list(row[7])) for row in rows]

    def reset(self) -> None:
        self._conn.execute(RESET)
