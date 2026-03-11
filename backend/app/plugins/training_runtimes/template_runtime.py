"""Template for third-party training runtime plugins.

Copy this into your own module namespace, replace runtime_id and start logic,
then include the module path in TRAINING_RUNTIME_PLUGIN_MODULES.
"""

from __future__ import annotations

from app.services.training_runtime_service import TrainingRuntimeStartResult


PLUGIN_VERSION = "1.0.0"


def register_training_runtime_plugins(register) -> None:
    """Register runtime plugins."""

    def validate() -> list[str]:
        errors: list[str] = []
        # Example checks:
        # - external binary available
        # - env vars present
        # - GPU driver compatibility
        return errors

    async def start(ctx) -> TrainingRuntimeStartResult:
        # Implement your orchestration:
        # - submit to external scheduler
        # - start local process
        # - return task id if cancellation/tracking is supported
        #
        # This template uses the built-in simulate runner for safe scaffolding.
        if ctx.simulate_runner is None:
            raise ValueError("simulate runner unavailable in this execution context")

        import asyncio

        asyncio.create_task(ctx.simulate_runner(ctx.experiment_id, dict(ctx.config or {})))
        return TrainingRuntimeStartResult(
            message=f"Template runtime started (v{PLUGIN_VERSION}).",
            task_id=None,
            runtime_updates={
                "backend": "simulate",
                "runtime_kind": "template_runtime",
                "plugin_version": PLUGIN_VERSION,
            },
        )

    register(
        runtime_id="template.runtime-v1",
        label="Template Runtime v1",
        description="Template plugin runtime for external scheduler integration.",
        execution_backend="local",
        validate=validate,
        start=start,
        required_dependencies=[],
        supports_task_tracking=False,
        supports_cancellation=True,
    )

