"""Synthetic-data backend plugins (USER-SUCCESS Epic 2).

Each backend wraps one LLM transport (Ollama, the configured teacher
endpoint, NeMo, etc.) behind a uniform `SynthBackend` protocol so
playbooks don't have to care which model is generating the rows.

Default selection order (when caller passes `backend=None`):

  1. OllamaBackend       — local Ollama on `OLLAMA_HOST`
  2. TeacherModelBackend — whatever `TEACHER_MODEL_API_URL` points at
                            (the same dispatcher the legacy
                            ``call_teacher_model`` flow uses)

Callers can pin a specific backend by name via
`pick_backend("ollama:llama3.1:8b")` etc.
"""

from __future__ import annotations

from .base import SynthBackend, SynthBackendError, pick_backend
from .nemo import NemoBackend
from .ollama import OllamaBackend
from .teacher import TeacherModelBackend

# Order matters: pick_backend() walks this list in order when no
# explicit backend is requested. NeMo is positioned LAST so auto-pick
# for existing local-only installs (Ollama users) is unchanged —
# users opt into NeMo by pinning it via the picker.
BACKEND_REGISTRY: list[type[SynthBackend]] = [
    OllamaBackend,
    TeacherModelBackend,
    NemoBackend,
]


__all__ = [
    "BACKEND_REGISTRY",
    "NemoBackend",
    "OllamaBackend",
    "SynthBackend",
    "SynthBackendError",
    "TeacherModelBackend",
    "pick_backend",
]
