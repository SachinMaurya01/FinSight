"""API-key authentication backed by hashed keys in Postgres.

Keys are generated once, shown once, and stored only as SHA-256 digests. An
optional ``API_KEY_PEPPER`` value is mixed into the digest so a database leak
does not expose reusable keys.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import secrets
import sys
from dataclasses import dataclass

from fastapi import HTTPException, Request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import Settings
from src.config import settings as default_settings

logger = logging.getLogger(__name__)


class ApiKeyStoreError(RuntimeError):
    """Raised when the API-key store cannot be queried."""


@dataclass(frozen=True)
class AuthContext:
    """Authenticated caller identity used for limits and audit records."""

    name: str
    identity: str
    rate_limit_per_minute: int | None = None


def hash_api_key(raw_key: str, pepper: str | None = None) -> str:
    """Return the stable digest stored in Postgres for ``raw_key``."""
    material = f"{pepper}:{raw_key}" if pepper else raw_key
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _extract_api_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    authorization = request.headers.get("authorization", "")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    return None


class ApiKeyStore:
    """Postgres-backed API-key lookup and provisioning."""

    def __init__(self, settings: Settings | None = None, engine: Engine | None = None) -> None:
        self._settings = settings or default_settings
        self._engine = engine

    def _get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                self._settings.postgres_database_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 2},
            )
        return self._engine

    def verify(self, raw_key: str) -> AuthContext | None:
        """Return the caller for a valid active key, or ``None``."""
        digest = hash_api_key(raw_key, self._settings.api_key_pepper)
        try:
            with self._get_engine().begin() as connection:
                row = (
                    connection.execute(
                        text(
                            "SELECT name, request_limit_per_minute FROM api_keys "
                            "WHERE key_hash = :key_hash AND is_active = true"
                        ),
                        {"key_hash": digest},
                    )
                    .mappings()
                    .first()
                )
                if row is None:
                    return None
                connection.execute(
                    text("UPDATE api_keys SET last_used_at = now() WHERE key_hash = :key_hash"),
                    {"key_hash": digest},
                )
        except Exception as exc:
            raise ApiKeyStoreError(f"API-key store unavailable: {type(exc).__name__}") from exc

        return AuthContext(
            name=str(row["name"]),
            identity=f"key:{digest[:16]}",
            rate_limit_per_minute=row["request_limit_per_minute"],
        )

    def create_key(
        self,
        name: str,
        *,
        request_limit_per_minute: int | None = None,
    ) -> str:
        """Create a key and return the raw value exactly once."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must not be empty")
        raw_key = f"fs_{secrets.token_urlsafe(32)}"
        digest = hash_api_key(raw_key, self._settings.api_key_pepper)
        with self._get_engine().begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO api_keys (name, key_hash, request_limit_per_minute) VALUES (:name, :key_hash, :limit)"
                ),
                {"name": clean_name, "key_hash": digest, "limit": request_limit_per_minute},
            )
        return raw_key


_STORE: ApiKeyStore | None = None


def get_api_key_store(settings: Settings | None = None) -> ApiKeyStore:
    """Return the process-wide API-key store."""
    global _STORE
    if _STORE is None:
        _STORE = ApiKeyStore(settings)
    return _STORE


async def require_api_key(request: Request) -> AuthContext:
    """FastAPI dependency enforcing ``X-API-Key`` or Bearer authentication."""
    if not getattr(request.app.state, "auth_enabled", True):
        client = request.client.host if request.client else "unknown"
        return AuthContext(name="local", identity=f"ip:{client}")

    raw_key = _extract_api_key(request)
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    store: ApiKeyStore = getattr(request.app.state, "api_key_store", None) or get_api_key_store(default_settings)
    try:
        auth = store.verify(raw_key)
    except ApiKeyStoreError as exc:
        logger.error("API-key verification unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="authentication store unavailable") from exc

    if auth is None:
        raise HTTPException(
            status_code=401,
            detail="invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return auth


def main(argv: list[str] | None = None) -> int:
    """Provision API keys: ``python -m src.api.auth create --name local``."""
    parser = argparse.ArgumentParser(description="Manage FinSight API keys")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a new API key")
    create.add_argument("--name", required=True, help="human-readable key name")
    create.add_argument(
        "--request-limit-per-minute",
        type=int,
        default=None,
        help="optional per-key override for the default rate limit",
    )
    args = parser.parse_args(argv)

    if args.command == "create":
        try:
            raw_key = get_api_key_store(default_settings).create_key(
                args.name,
                request_limit_per_minute=args.request_limit_per_minute,
            )
        except Exception as exc:
            print(f"failed to create API key: {exc}", file=sys.stderr)
            return 1
        print(raw_key)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
