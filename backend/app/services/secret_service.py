"""Project secret storage service with reversible at-rest encryption."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.project import Project
from app.models.secret import ProjectSecret


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _validate_provider_key(provider: str, key_name: str) -> tuple[str, str]:
    p = _normalize_token(provider)
    k = _normalize_token(key_name)
    if not p or len(p) > 64:
        raise ValueError("provider must be between 1 and 64 characters")
    if not k or len(k) > 64:
        raise ValueError("key_name must be between 1 and 64 characters")
    return p, k


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * max(2, len(value) - 4)}{value[-2:]}"


def _master_key() -> bytes:
    seed = (settings.SECRETS_ENCRYPTION_KEY or settings.JWT_SECRET).strip()
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _encrypt_secret(raw_value: str) -> str:
    data = raw_value.encode("utf-8")
    key = _master_key()
    nonce = secrets.token_bytes(16)
    stream = _keystream(key, nonce, len(data))
    cipher = bytes(a ^ b for a, b in zip(data, stream))
    mac = hmac.new(key, nonce + cipher, hashlib.sha256).digest()
    payload = nonce + mac + cipher
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _decrypt_secret(encoded_value: str) -> str:
    payload = base64.urlsafe_b64decode(encoded_value.encode("ascii"))
    if len(payload) < 48:
        raise ValueError("Corrupt encrypted secret payload")
    nonce = payload[:16]
    mac = payload[16:48]
    cipher = payload[48:]
    key = _master_key()
    expected_mac = hmac.new(key, nonce + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected_mac):
        raise ValueError("Secret payload integrity validation failed")
    stream = _keystream(key, nonce, len(cipher))
    data = bytes(a ^ b for a, b in zip(cipher, stream))
    return data.decode("utf-8")


async def upsert_project_secret(
    db: AsyncSession,
    project_id: int,
    provider: str,
    key_name: str,
    value: str,
) -> ProjectSecret:
    """Create or update a project secret value."""
    provider_norm, key_norm = _validate_provider_key(provider, key_name)
    raw_value = value.strip()
    if not raw_value:
        raise ValueError("Secret value cannot be empty")

    project_res = await db.execute(select(Project.id).where(Project.id == project_id))
    if project_res.scalar_one_or_none() is None:
        raise ValueError(f"Project {project_id} not found")

    result = await db.execute(
        select(ProjectSecret).where(
            ProjectSecret.project_id == project_id,
            ProjectSecret.provider == provider_norm,
            ProjectSecret.key_name == key_norm,
        ).order_by(ProjectSecret.updated_at.desc(), ProjectSecret.id.desc()).limit(1)
    )
    secret_obj = result.scalar_one_or_none()
    encrypted = _encrypt_secret(raw_value)
    hint = _mask_secret(raw_value)
    if secret_obj is None:
        secret_obj = ProjectSecret(
            project_id=project_id,
            provider=provider_norm,
            key_name=key_norm,
            encrypted_value=encrypted,
            value_hint=hint,
        )
        db.add(secret_obj)
    else:
        secret_obj.encrypted_value = encrypted
        secret_obj.value_hint = hint
        secret_obj.updated_at = _utcnow()

    await db.flush()
    await db.refresh(secret_obj)
    return secret_obj


async def list_project_secrets(
    db: AsyncSession,
    project_id: int,
) -> list[ProjectSecret]:
    result = await db.execute(
        select(ProjectSecret)
        .where(ProjectSecret.project_id == project_id)
        .order_by(ProjectSecret.provider.asc(), ProjectSecret.key_name.asc())
    )
    return list(result.scalars().all())


async def delete_project_secret(
    db: AsyncSession,
    project_id: int,
    provider: str,
    key_name: str,
) -> bool:
    provider_norm, key_norm = _validate_provider_key(provider, key_name)
    result = await db.execute(
        select(ProjectSecret).where(
            ProjectSecret.project_id == project_id,
            ProjectSecret.provider == provider_norm,
            ProjectSecret.key_name == key_norm,
        )
    )
    secret_rows = list(result.scalars().all())
    if not secret_rows:
        return False
    for secret_obj in secret_rows:
        await db.delete(secret_obj)
    await db.flush()
    return True


async def get_project_secret_value(
    db: AsyncSession,
    project_id: int,
    provider: str,
    key_name: str,
    touch: bool = True,
) -> str | None:
    """Return decrypted secret value, or None when not found."""
    provider_norm, key_norm = _validate_provider_key(provider, key_name)
    result = await db.execute(
        select(ProjectSecret).where(
            ProjectSecret.project_id == project_id,
            ProjectSecret.provider == provider_norm,
            ProjectSecret.key_name == key_norm,
        ).order_by(ProjectSecret.updated_at.desc(), ProjectSecret.id.desc()).limit(1)
    )
    secret_obj = result.scalar_one_or_none()
    if not secret_obj:
        return None

    if touch:
        secret_obj.last_used_at = _utcnow()
        await db.flush()

    return _decrypt_secret(secret_obj.encrypted_value)


def serialize_secret(secret_obj: ProjectSecret) -> dict:
    return {
        "id": secret_obj.id,
        "project_id": secret_obj.project_id,
        "provider": secret_obj.provider,
        "key_name": secret_obj.key_name,
        "value_hint": secret_obj.value_hint,
        "created_at": secret_obj.created_at.isoformat() if secret_obj.created_at else None,
        "updated_at": secret_obj.updated_at.isoformat() if secret_obj.updated_at else None,
        "last_used_at": secret_obj.last_used_at.isoformat() if secret_obj.last_used_at else None,
    }
