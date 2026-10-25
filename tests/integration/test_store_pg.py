"""Store contract against real Postgres + pgvector. Skips if localhost:5432 is unreachable."""
import os
import socket
import uuid

import pytest

from store_contract import CHECKS

DSN = os.environ.get("SMRITI_PG_DSN", "host=localhost port=5432 user=postgres dbname=postgres")


def _pg_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5432), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _pg_reachable(), reason="Postgres tunnel not open")


@pytest.fixture
def pg_store():
    from smriti.store.store import PgStore

    schema = f"smriti_test_{uuid.uuid4().hex[:8]}"
    store = PgStore(DSN, schema=schema, embed_dim=3)
    store.init_schema()
    try:
        yield store
    finally:
        store.drop_schema()
        store.close()


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_pg_store(pg_store, check):
    check(pg_store)
