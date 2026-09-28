from __future__ import annotations

import pytest

from load_client import LoadClient, build_client
from proxy_client import ProxyClient

_OPT_IN_MARKERS = (
    ("weekly", WEEKLY_ANOMALY_OPT_IN_ENV),
    ("redis_chaos", REDIS_CHAOS_OPT_IN_ENV),
)

@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoadClient:
    return build_client(proxy)
