"""Example third-party training runtime plugin.

This demonstrates the runtime plugin SDK contract used by
`TRAINING_RUNTIME_PLUGIN_MODULES`.
"""

from __future__ import annotations

import asyncio

from app.services.training_runtime_service import TrainingRuntimeStartResult


def register_training_runtime_plugins(register) -> None:
    """Register an example local runtime that reuses simulated training loop."""

    def validate() -> list[str]:
        return []

    async def start(ctx) -> TrainingRuntimeStartResult:
        if ctx.simulate_runner is None:
            raise ValueError("simulate runner is unavailable for example plugin runtime")
        asyncio.create_task(ctx.simulate_runner(ctx.experiment_id, dict(ctx.config or {})))
        return TrainingRuntimeStartResult(
            message="Example plugin runtime started (simulate-backed).",
            task_id=None,
            runtime_updates={
                "backend": "simulate",
                "runtime_kind": "plugin_example_simulate",
            },
        )

    register(
        runtime_id="plugin.example_simulate",
        label="Plugin Example Simulate",
        description="Example plugin runtime that dispatches the built-in simulate loop.",
        execution_backend="local",
        validate=validate,
        start=start,
        required_dependencies=[],
        supported_modalities=["text"],
        supports_task_tracking=False,
        supports_cancellation=True,
    )
