"""Shared Celery job helpers for status and cancellation."""

from __future__ import annotations


def get_task_status(task_id: str) -> dict:
    from app.worker import celery_app

    if not task_id.strip():
        raise ValueError("task_id is required")

    result = celery_app.AsyncResult(task_id)
    state = str(result.state or "PENDING").lower()
    return {
        "task_id": task_id,
        "state": state,
        "ready": bool(result.ready()),
        "successful": bool(result.successful()) if result.ready() else None,
        "failed": bool(result.failed()) if result.ready() else False,
    }


def cancel_task(task_id: str, terminate: bool = False) -> dict:
    from app.worker import celery_app

    if not task_id.strip():
        raise ValueError("task_id is required")

    celery_app.control.revoke(task_id, terminate=terminate)
    return {
        "task_id": task_id,
        "status": "cancel_requested",
        "terminate": terminate,
    }
