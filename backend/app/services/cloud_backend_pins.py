"""Cloud backend pins — ``cloud:<provider>:<model>`` → a ``CloudLlmBackend``
built with the API key saved on the project.

``synth_backends.pick_backend`` only knows local backend names (ollama,
teacher, …); every route that lets a user pick a cloud model sends the pin
through here first and hands the result on as ``backend_override``. Used by
the span / relabel routes (``api/synthetic.py``) and the guided flows
(``api/flows.py``: documents → Q&A, seed documents).

Errors carry an HTTP-ish ``status_code`` so routes map them the same way
everywhere: 400 for a malformed pin / unknown provider, 402 when the project
has no key saved for the provider (the frontend renders "save a key first"
for 402).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

# Cloud provider → (secret_provider, key_name) of the project secret that
# holds its API key. Same coordinates as the gold generator's
# ``_PROVIDER_SECRET_MAP`` and the Synthetic tab's "Save key to this project",
# so a user enters a key once per project.
CLOUD_PROVIDER_SECRET_COORDS: dict[str, tuple[str, str]] = {
    "openai": ("cloud_llm_openai", "api_key"),
    "anthropic": ("cloud_llm_anthropic", "api_key"),
    "deepseek": ("cloud_llm_deepseek", "api_key"),
}


class CloudPinError(Exception):
    """A cloud pin that can't be used. ``status_code`` 400 (malformed /
    unknown provider) or 402 (no key saved for the provider)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def is_cloud_pin(pin: str | None) -> bool:
    return bool(pin) and str(pin).startswith("cloud:")


def parse_cloud_pin(pin: str | None) -> tuple[str, str] | None:
    """``"cloud:anthropic:claude-x"`` → ``("anthropic", "claude-x")``; None for
    a pin that isn't a cloud pin. Raises ``CloudPinError(400)`` when it is
    one but malformed or names an unknown provider."""
    if not is_cloud_pin(pin):
        return None
    parts = str(pin).split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise CloudPinError(
            400,
            f"Malformed cloud backend pin {pin!r}. Expected "
            f"'cloud:<provider>:<model>' (e.g. 'cloud:openai:gpt-4o-mini').",
        )
    _, provider, model = parts
    if provider not in CLOUD_PROVIDER_SECRET_COORDS:
        raise CloudPinError(
            400,
            f"Unknown cloud provider {provider!r}. Supported: "
            f"{', '.join(CLOUD_PROVIDER_SECRET_COORDS.keys())}.",
        )
    return provider, model


async def saved_cloud_key(db: AsyncSession, project_id: int, provider: str) -> str | None:
    """The project's saved API key for ``provider`` (None when none)."""
    coords = CLOUD_PROVIDER_SECRET_COORDS.get(provider)
    if coords is None:
        return None
    from app.services.secret_service import get_project_secret_value

    secret_provider, key_name = coords
    return await get_project_secret_value(db, project_id, secret_provider, key_name) or None


def missing_key_message(provider: str) -> str:
    return (
        f"No {provider} API key saved for this project. Save one on Data → "
        f"Generate (\"Save key to this project\") or under Project Settings → "
        f"Secrets, and retry."
    )


def check_command_pin(pin: str | None) -> None:
    """``command:<model>`` with a model that could parse as a CLI flag (or is
    otherwise malformed) → ``CloudPinError`` 400."""
    from app.services.synth_backends.command import validate_pin

    try:
        validate_pin(pin)
    except ValueError as exc:
        raise CloudPinError(400, str(exc)) from exc


async def resolve_cloud_backend(db: AsyncSession, project_id: int, pin: str | None) -> Any:
    """A ``CloudLlmBackend`` for a cloud pin, using the project's saved key;
    None for any other pin shape (the caller's ``pick_backend`` handles
    those). Raises ``CloudPinError`` (400 / 402)."""
    check_command_pin(pin)
    parsed = parse_cloud_pin(pin)
    if parsed is None:
        return None
    provider, model = parsed
    api_key = await saved_cloud_key(db, project_id, provider)
    if not api_key:
        raise CloudPinError(402, missing_key_message(provider))
    from app.services.synth_backends import CloudLlmBackend

    return CloudLlmBackend(provider=provider, model=model, api_key=api_key)
