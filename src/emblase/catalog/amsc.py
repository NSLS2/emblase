"""Client for the AmSC (American Science Cloud) OpenMetadata catalog API.

Relevant endpoints used:
    GET  /account/projects                          Connectivity check; lists projects
    GET  /search/catalog                            Full-text / filtered search
    POST /catalog/{catalog_name}/{entity_type}      Register a new artifact/collection
    GET  /catalog/{fqn}                             Fetch a single entity by FQN

Auth: ``Authorization: Bearer <token>`` header.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)

_BASE = "https://api.american-science-cloud.org/api/current"


@dataclass
class CatalogArtifact:
    """Minimal representation of a catalog entity returned by search."""

    fqn: str
    name: str
    display_name: str | None
    entity_type: str  # "artifact", "artifactCollection", "scientificWork", …
    description: str | None
    location: str | None
    parent_fqn: str | None
    # Raw dict for any extra fields
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CatalogArtifact":
        return cls(
            fqn=d.get("fqn") or d.get("fullyQualifiedName") or "",
            name=d.get("name") or "",
            display_name=d.get("displayName") or d.get("display_name"),
            entity_type=d.get("type") or d.get("entityType") or "unknown",
            description=d.get("description"),
            location=d.get("location"),
            parent_fqn=d.get("parentFqn") or d.get("parent_fqn"),
            raw=d,
        )


class AmscClient:
    """Thin async HTTP client for the AmSC catalog API.

    Designed to be used as an async context manager::

        async with AmscClient() as client:
            projects = await client.list_projects()
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        catalog_name: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = (base_url or settings.amsc_openmetadata_catalog_url or _BASE).rstrip("/")
        self.token = token or settings.amsc_openmetadata_token
        self.catalog_name = catalog_name or settings.amsc_openmetadata_catalog_name
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ── context manager ──────────────────────────────────────────────────────

    async def __aenter__(self) -> "AmscClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("AmscClient must be used as an async context manager")
        return self._client

    # ── account / connectivity ────────────────────────────────────────────────

    async def list_projects(self) -> list[dict[str, Any]]:
        """Return the list of projects for the authenticated user.

        Used as a lightweight connectivity / auth check.
        """
        resp = await self._http().get("/account/projects")
        resp.raise_for_status()
        data = resp.json()
        # API may return a list directly or a wrapped object
        if isinstance(data, list):
            return data
        return data.get("projects") or data.get("items") or []

    # ── catalog search ────────────────────────────────────────────────────────

    async def search(
        self,
        query: str = "",
        entity_types: list[str] | None = None,
        parent_fqn: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[CatalogArtifact]:
        """Search the catalog.  Returns up to *limit* results.

        Args:
            query:        Free-text search term.  Pass "" for wildcard (uses ``*``).
            entity_types: Filter by type, e.g. ["artifact", "artifactCollection"].
            parent_fqn:   Restrict to children of this FQN.
            limit:        Max results to return.
            offset:       Pagination offset.
        """
        params: dict[str, Any] = {
            "q": query or "*",  # q is required; "*" returns all
            "limit": limit,
            "offset": offset,
        }
        if entity_types:
            params["entity_types"] = entity_types
        if parent_fqn:
            params["parent_fqn"] = parent_fqn

        resp = await self._http().get("/search/catalog", params=params)
        resp.raise_for_status()
        data = resp.json()

        # Response shape: {"total_num_results": N, "results": [...], "next_cursor": "..."}
        items: list[dict[str, Any]] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("results") or data.get("items") or data.get("data") or []

        return [CatalogArtifact.from_dict(item) for item in items]

    # ── catalog registration ──────────────────────────────────────────────────

    async def register_artifact(
        self,
        name: str,
        description: str,
        location: str,
        parent_fqn: str,
        display_name: str | None = None,
        fmt: str | None = None,
        size: int | None = None,
        catalog_name: str | None = None,
    ) -> dict[str, Any]:
        """Register a new ``artifact`` entity in the catalog.

        Returns the raw response JSON (including the assigned ``fqn``).
        Raises ``httpx.HTTPStatusError`` on non-2xx.
        """
        cn = catalog_name or self.catalog_name
        if not cn:
            raise ValueError("catalog_name must be set (EMBLASE_AMSC_OPENMETADATA_CATALOG_NAME)")

        body: dict[str, Any] = {
            "type": "artifact",
            "name": name,
            "description": description,
            "location": location,
            "parent_fqn": parent_fqn,
        }
        if display_name:
            body["display_name"] = display_name
        if fmt:
            body["format"] = fmt
        if size is not None:
            body["size"] = size

        resp = await self._http().post(f"/catalog/{cn}/artifact", json=body)
        resp.raise_for_status()
        return resp.json()

    async def register_collection(
        self,
        name: str,
        parent_fqn: str,
        description: str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
    ) -> dict[str, Any]:
        """Register a new ``artifactCollection`` entity."""
        cn = catalog_name or self.catalog_name
        if not cn:
            raise ValueError("catalog_name must be set")

        body: dict[str, Any] = {
            "type": "artifactCollection",
            "name": name,
            "parent_fqn": parent_fqn,
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location

        resp = await self._http().post(f"/catalog/{cn}/artifactCollection", json=body)
        resp.raise_for_status()
        return resp.json()
