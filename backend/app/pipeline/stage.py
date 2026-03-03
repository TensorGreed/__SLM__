"""Pipeline stage definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    success: bool
    stage_name: str
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str
    description: str
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
