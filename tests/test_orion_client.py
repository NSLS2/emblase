"""Tests for OrionClient — all HTTP calls are mocked with httpx.MockTransport."""

import httpx
import pytest

from emblase.compute.orion import OrionClient, OrionJob


def _make_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    """Return a mock transport that yields responses in order."""
    it = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(it)

    return httpx.MockTransport(handler)


@pytest.fixture
def client_with_mock(monkeypatch):
    """Return a helper that injects a mock httpx client into OrionClient."""

    def factory(responses: list[httpx.Response]) -> OrionClient:
        c = OrionClient(api_url="https://fake-orion", api_key="testkey", cluster="orion")
        c._client = httpx.AsyncClient(
            transport=_make_transport(responses),
            headers={"x-api-key": "testkey"},
        )
        return c

    return factory


@pytest.mark.asyncio
async def test_submit_job(client_with_mock):
    client = client_with_mock(
        [
            httpx.Response(
                200,
                json={"job_id": 42, "step_id": "batch", "errors": [], "warnings": []},
            ),
        ]
    )
    job_id = await client.submit_job(script="#!/bin/bash\nhostname")
    assert job_id == 42


@pytest.mark.asyncio
async def test_get_job_completed(client_with_mock):
    client = client_with_mock(
        [
            httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "job_id": 42,
                            "state": ["COMPLETED"],
                            "nodes": "mars5",
                            "stdout": "/tmp/slurm-42.out",
                            "stderr": "/tmp/slurm-42.out",
                        }
                    ]
                },
            ),
        ]
    )
    info = await client.get_job(42)
    assert isinstance(info, OrionJob)
    assert info.state == "COMPLETED"
    assert info.node == "mars5"
    assert info.job_id == 42


@pytest.mark.asyncio
async def test_get_job_normalises_state_list(client_with_mock):
    """State comes back as a list from the API; client should unwrap it."""
    client = client_with_mock(
        [
            httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "job_id": 7,
                            "state": ["RUNNING"],
                            "nodes": "cnode1",
                            "stdout": None,
                            "stderr": None,
                        }
                    ]
                },
            ),
        ]
    )
    info = await client.get_job(7)
    assert info.state == "RUNNING"


@pytest.mark.asyncio
async def test_cancel_job(client_with_mock):
    client = client_with_mock(
        [
            httpx.Response(200, json={}),
        ]
    )
    # Should not raise
    await client.cancel_job(42)


@pytest.mark.asyncio
async def test_submit_job_http_error(client_with_mock):
    client = client_with_mock(
        [
            httpx.Response(401, json={"detail": "Unauthorized"}),
        ]
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.submit_job(script="#!/bin/bash\nhostname")


@pytest.mark.asyncio
async def test_wait_for_job_already_complete(client_with_mock):
    client = client_with_mock(
        [
            httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "job_id": 1,
                            "state": ["COMPLETED"],
                            "nodes": "n1",
                            "stdout": None,
                            "stderr": None,
                        }
                    ]
                },
            ),
        ]
    )
    info = await client.wait_for_job(1, poll_interval=0.01, timeout=5.0)
    assert info.state == "COMPLETED"


@pytest.mark.asyncio
async def test_wait_for_job_timeout(client_with_mock):
    """If job stays PENDING, wait_for_job should raise TimeoutError."""

    # Return PENDING indefinitely
    def handler(request):
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "job_id": 1,
                        "state": ["PENDING"],
                        "nodes": None,
                        "stdout": None,
                        "stderr": None,
                    }
                ]
            },
        )

    c = OrionClient(api_url="https://fake", api_key="k", cluster="orion")
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(TimeoutError):
        await c.wait_for_job(1, poll_interval=0.01, timeout=0.05)
