"""The proxy must not serialize concurrent provider calls on its event loop.

`_forward_and_record` makes a blocking (sync httpx) upstream call. It is run in
a worker thread so a slow provider cannot stall the event loop and queue every
other concurrent run behind it. This test fires several concurrent calls against
a deliberately slow upstream and asserts they overlap (total ≈ one delay, not N).
"""

from __future__ import annotations

import time
from uuid import uuid4

import httpx
import pytest

from src.core.config import AppConfig
from src.gateway.local_proxy import create_app

_UPSTREAM_DELAY = 0.2
_N = 5


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _cfg():
    cfg = AppConfig.load_default()
    cfg.bittensor.signing_required = False
    return cfg


def _slow_upstream(request: httpx.Request) -> httpx.Response:
    body = __import__("json").loads(request.content.decode("utf-8"))
    time.sleep(_UPSTREAM_DELAY)  # blocking — simulates a slow LLM provider
    return httpx.Response(
        200,
        json={
            "provider_id": body["provider"],
            "call_type": body.get("callType") or "unknown",
            "data": {"echo": True},
            "latency_ms": 0,
        },
    )


@pytest.mark.anyio
async def test_concurrent_calls_are_not_serialized():
    upstream = httpx.Client(transport=httpx.MockTransport(_slow_upstream), timeout=5.0)
    app = create_app(_cfg(), http_client=upstream)
    state = app.state.proxy
    state.load_hotkey()

    run_ids = [uuid4() for _ in range(_N)]
    for rid in run_ids:
        state.register_run(rid, track="MAIN", miner_hotkey="5RealMiner")

    import anyio

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        async def _one(rid):
            return await client.post(
                "/v1/call",
                json={"provider_id": "polymarket", "call_type": "get_market", "params": {}},
                headers={"X-Run-Id": str(rid)},
            )

        t0 = time.monotonic()
        async with anyio.create_task_group() as tg:
            results = []

            async def _run(rid):
                results.append(await _one(rid))

            for rid in run_ids:
                tg.start_soon(_run, rid)
        elapsed = time.monotonic() - t0

    assert all(r.status_code == 200 for r in results)
    # Serialized on the loop, this would be ~N * delay. Off-loop it overlaps and
    # stays close to a single delay. Generous ceiling to avoid CI flakiness.
    assert elapsed < _UPSTREAM_DELAY * _N * 0.6, (
        f"{_N} calls took {elapsed:.2f}s — looks serialized on the event loop"
    )
