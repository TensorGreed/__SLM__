"""Pydantic schemas for domain pack contracts and assignment."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.domain_pack import DomainPackStatus


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


class DomainPackOverlaySpec(BaseModel):
    dataset_split: dict[str, Any] = Field(default_factory=dict)
    training_defaults: dict[str, Any] = Field(default_factory=dict)
    registry_gates: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    normalization: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    audit: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class DomainPackContract(BaseModel):
    schema_ref: str = Field(default="slm.domain-pack/v1", alias="$schema")
    pack_id: str = Field(..., min_length=3, max_length=128)
    version: str = Field(default="1.0.0", min_length=1, max_length=32)
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    owner: str = Field(default="platform", min_length=1, max_length=128)
    status: DomainPackStatus = DomainPackStatus.ACTIVE
    default_profile_id: str | None = Field(default=None, min_length=3, max_length=128)
    tags: list[str] = Field(default_factory=list)
    overlay: DomainPackOverlaySpec = Field(default_factory=DomainPackOverlaySpec)

    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }

    @field_validator("pack_id")
    @classmethod
    def normalize_pack_id(cls, value: str) -> str:
        token = _normalize_token(value)
        if not token:
            raise ValueError("pack_id cannot be empty")
        return token

    @field_validator("default_profile_id")
    @classmethod
    def normalize_default_profile_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        token = _normalize_token(value)
        if not token:
            raise ValueError("default_profile_id cannot be empty")
        return token


class DomainPackSummaryResponse(BaseModel):
    id: int
    pack_id: str
    version: str
    display_name: str
    description: str
    owner: str
    status: DomainPackStatus
    schema_ref: str
    default_profile_id: str | None = None
    is_system: bool

    model_config = {"from_attributes": True}


class DomainPackResponse(DomainPackSummaryResponse):
    contract: DomainPackContract


class DomainPackDuplicateRequest(BaseModel):
    new_pack_id: str | None = Field(default=None, min_length=3, max_length=128)
    new_version: str | None = Field(default=None, min_length=1, max_length=32)
    status: DomainPackStatus = DomainPackStatus.DRAFT

    @field_validator("new_pack_id")
    @classmethod
    def normalize_new_pack_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        token = _normalize_token(value)
        if not token:
            raise ValueError("new_pack_id cannot be empty")
        return token
