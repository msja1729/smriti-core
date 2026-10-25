-- SPEC §7. EMBED_DIM is replaced at init with the embedding model's dimension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
  id           UUID PRIMARY KEY,
  tenant       TEXT NOT NULL,
  text         TEXT NOT NULL,
  source       TEXT NOT NULL,
  trust        SMALLINT NOT NULL,      -- 0 LOW, 1 MED, 2 HIGH
  sensitivity  SMALLINT NOT NULL,      -- 0 PUBLIC, 1 PII
  flags        TEXT[] NOT NULL DEFAULT '{}',
  session_id   TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding    vector(EMBED_DIM)
);
CREATE INDEX IF NOT EXISTS memories_tenant_idx ON memories (tenant);

CREATE TABLE IF NOT EXISTS edges (     -- derivation graph: child derived from parent
  child  UUID REFERENCES memories(id),
  parent UUID REFERENCES memories(id),
  PRIMARY KEY (child, parent)
);

CREATE TABLE IF NOT EXISTS ledger (    -- append-only: every write attempt, accepted or not
  seq        BIGSERIAL PRIMARY KEY,
  ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
  tenant     TEXT NOT NULL,
  op         TEXT NOT NULL,            -- write | reject | quarantine | summarize | read
  memory_id  UUID,
  detail     JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_log (
  seq        BIGSERIAL PRIMARY KEY,
  episode_id TEXT, session_id TEXT, step INT,
  tool TEXT, args JSONB, tainted BOOLEAN, decision TEXT,
  memory_ids_in_context UUID[],
  turn_seq   BIGINT                     -- the user turn that produced this call
);

CREATE TABLE IF NOT EXISTS turns (     -- user requests, PII-redacted; the auditor replays them
  seq        BIGSERIAL PRIMARY KEY,
  ts         TIMESTAMPTZ NOT NULL,
  tenant     TEXT NOT NULL,
  episode_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  user_text  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_tenant_idx ON turns (tenant);
