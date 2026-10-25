"""Static check (SPEC §7): every query in store.py that reads tenant data filters by tenant
before any vector ordering, and the ledger is never updated or deleted."""
import ast
import re
from pathlib import Path

import smriti.store.store as store_module

TENANT_DATA = re.compile(r"\b(FROM|JOIN|UPDATE)\s+(memories|ledger|edges)\b", re.I)
TENANT_FILTER = re.compile(r"\btenant\s*=\s*%\(tenant\)s")
LEDGER_MUTATION = re.compile(r"\b(UPDATE\s+ledger|DELETE\s+FROM\s+ledger)\b", re.I)


SQL_START = re.compile(r"\s*(SELECT|UPDATE|DELETE|WITH|INSERT|TRUNCATE)\b", re.I)


def sql_strings(source: str) -> list[str]:
    # Literals in the source plus the built values of module constants (f-strings are
    # split into pieces in the AST, so their full text only exists at runtime).
    literals = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    constants = [v for v in vars(store_module).values() if isinstance(v, str)]
    return [s for s in literals + constants if SQL_START.match(s)]


def execute_args(source: str) -> list[ast.expr]:
    return [
        node.args[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute" and node.args
    ]


def violations(query: str) -> list[str]:
    if not TENANT_DATA.search(query):
        return []
    found = TENANT_FILTER.search(query)
    if not found:
        return ["missing tenant filter"]
    order = re.search(r"\bORDER\s+BY\b", query, re.I)
    if order and order.start() < found.start():
        return ["ORDER BY before tenant filter"]
    return []


def store_source() -> str:
    return Path(store_module.__file__).read_text()


def test_every_query_filters_by_tenant():
    queries = sql_strings(store_source())
    assert len(queries) >= 4, "checker found too few queries; is it still parsing store.py?"
    bad = {q: violations(q) for q in queries if violations(q)}
    assert not bad, bad


def test_every_execute_runs_a_checked_constant():
    # Inline or computed SQL would bypass the check above; only named module constants
    # (or the schema DDL / sql.SQL schema statements in init/drop) may be executed.
    allowed_locals = {"ddl"}
    for arg in execute_args(store_source()):
        if isinstance(arg, ast.Name):
            value = vars(store_module).get(arg.id)
            assert isinstance(value, str) or arg.id in allowed_locals, arg.id
        else:
            text = ast.unparse(arg)
            assert text.startswith("sql.SQL('CREATE SCHEMA") or text.startswith(
                "sql.SQL('DROP SCHEMA"), text


def test_search_query_is_checked():
    assert store_module.SEARCH in sql_strings(store_source())
    assert not violations(store_module.SEARCH)


def test_ledger_is_append_only():
    assert not LEDGER_MUTATION.search(store_source())


def test_checker_catches_missing_filter():
    assert violations("SELECT text FROM memories ORDER BY embedding <=> %(q)s::vector")
    assert violations(
        "SELECT text FROM memories ORDER BY id LIMIT 1 -- tenant = %(tenant)s"
    )
    assert not violations("SELECT text FROM memories WHERE tenant = %(tenant)s ORDER BY id")
    assert not violations("SELECT seq FROM tool_log WHERE episode_id = %(episode_id)s")
