import pytest

from smriti.store.fake_store import FakeStore
from store_contract import CHECKS


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_fake_store(check):
    check(FakeStore())
