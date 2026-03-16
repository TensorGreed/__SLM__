from typing import Any, Optional
from fastapi import HTTPException

class SLMError(HTTPException):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        stage: str,
        message: str,
        actionable_fix: Optional[str] = None,
        docs_url: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        detail = {
            "error_code": error_code,
            "stage": stage,
            "message": message,
            "actionable_fix": actionable_fix,
            "docs_url": docs_url,
            "metadata": metadata,
        }
        super().__init__(status_code=status_code, detail=detail)

class ProjectNotFoundError(SLMError):
    def __init__(self, project_id: int):
        super().__init__(
            status_code=404,
            error_code="PROJECT_NOT_FOUND",
            stage="general",
            message=f"Project {project_id} not found.",
            actionable_fix="Check the project ID and try again.",
        )

class HardwareNotFoundError(SLMError):
    def __init__(self, hardware_id: str):
        super().__init__(
            status_code=400,
            error_code="HARDWARE_NOT_FOUND",
            stage="hardware",
            message=f"Hardware profile '{hardware_id}' not found in catalog.",
            actionable_fix="Check the hardware ID or view the catalog for valid options.",
        )

class StrictExecutionError(SLMError):
    def __init__(self, stage: str, message: str):
        super().__init__(
            status_code=400,
            error_code="STRICT_EXECUTION_VIOLATION",
            stage=stage,
            message=message,
            actionable_fix="Disable STRICT_EXECUTION_MODE or provide the required environment/configuration.",
        )
