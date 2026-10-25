import pytest
from hypothesis import given, strategies as st

from smriti.gateway.labels import CrossTenantError, Label, Sensitivity, Trust, join, top

TENANT = "acme"

sources = st.sampled_from(["user", "tool:web", "tool:email", "agent", "derived"])
parent_sets = st.frozensets(st.uuids().map(str), max_size=4)


@st.composite
def labels(draw, tenant=TENANT):
    return Label(
        source=draw(sources),
        tenant=tenant,
        trust=draw(st.sampled_from(Trust)),
        sensitivity=draw(st.sampled_from(Sensitivity)),
        parents=draw(parent_sets),
    )


@given(labels(), labels())
def test_join_takes_min_trust_max_sensitivity_union_parents(a, b):
    j = join(a, b)
    assert j.trust == min(a.trust, b.trust)
    assert j.sensitivity == max(a.sensitivity, b.sensitivity)
    assert j.parents == a.parents | b.parents
    assert j.source == "derived"
    assert j.tenant == TENANT


@given(labels(), labels())
def test_join_commutative(a, b):
    assert join(a, b) == join(b, a)


@given(labels(), labels(), labels())
def test_join_associative(a, b, c):
    assert join(join(a, b), c) == join(a, join(b, c))


@given(labels())
def test_top_is_identity(a):
    assert join(top(TENANT), a) == join(a, top(TENANT))
    j = join(top(TENANT), a)
    assert (j.trust, j.sensitivity, j.parents) == (a.trust, a.sensitivity, a.parents)


@given(labels(), labels())
def test_join_never_raises_trust(a, b):
    assert join(a, b).trust <= a.trust


@given(labels(), labels(tenant="other"))
def test_join_cross_tenant_raises(a, b):
    with pytest.raises(CrossTenantError):
        join(a, b)


def test_sensitivity_is_ordered_not_string():
    # max over strings would rank "public" above "pii" and launder PII.
    assert max(Sensitivity.PUBLIC, Sensitivity.PII) == Sensitivity.PII
    assert Trust.LOW < Trust.MED < Trust.HIGH
