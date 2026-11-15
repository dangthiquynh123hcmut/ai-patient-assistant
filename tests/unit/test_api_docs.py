"""API docs (Swagger UI, ReDoc, OpenAPI) chỉ được phục vụ khi bật PA_API_DOCS."""

from __future__ import annotations

import httpx
import pytest

from patient_assistant.interface.api.app import create_app

DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


def _client(api_docs: bool) -> httpx.AsyncClient:
    # Không chạy lifespan (ASGITransport) -> không cần container: chỉ kiểm tra route docs.
    app = create_app(api_docs=api_docs)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize("path", DOC_PATHS)
async def test_docs_are_hidden_by_default(path: str) -> None:
    async with _client(api_docs=False) as client:
        assert (await client.get(path)).status_code == 404


async def test_docs_list_every_api_when_enabled() -> None:
    async with _client(api_docs=True) as client:
        for path in DOC_PATHS:
            assert (await client.get(path)).status_code == 200, path
        paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/v1/auth/login", "/v1/chat/{conversation_id}/messages", "/v1/admin/users"} <= set(paths)
