"""Dataset preparation API routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DatasetType
from app.services.data_adapter_service import (
    clear_plugin_data_adapters,
    list_data_adapter_catalog,
    load_data_adapter_plugins_from_settings,
)
from app.services.dataset_service import (
    combine_datasets,
    preview_project_data_adapter,
    profile_project_dataset,
    resolve_project_dataset_adapter_preference,
    save_project_dataset_adapter_preference,
    split_dataset,
)
from app.services.domain_profile_service import get_dataset_split_defaults
from app.services.domain_runtime_service import resolve_project_domain_runtime

router = APIRouter(prefix="/projects/{project_id}/dataset", tags=["Dataset Prep"])


class SplitRequest(BaseModel):
    train_ratio: float = Field(0.8, gt=0, lt=1)
    val_ratio: float = Field(0.1, ge=0, lt=1)
    test_ratio: float = Field(0.1, ge=0, lt=1)
    seed: int = 42
    include_types: list[DatasetType] | None = None
    chat_template: str = "llama3"
    adapter_id: str | None = None
    adapter_config: dict[str, Any] | None = None
    field_mapping: dict[str, str] | None = None
    task_profile: str | None = None

    @model_validator(mode="after")
    def validate_ratios(self):
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        return self


class ProfileRequest(BaseModel):
    dataset_type: DatasetType = DatasetType.CLEANED
    sample_size: int = Field(default=500, ge=10, le=5000)
    document_id: int | None = None
    field_mapping: dict[str, str] | None = None


class SplitEffectiveConfigRequest(BaseModel):
    train_ratio: float | None = Field(default=None, gt=0, lt=1)
    val_ratio: float | None = Field(default=None, ge=0, lt=1)
    test_ratio: float | None = Field(default=None, ge=0, lt=1)
    seed: int | None = None
    include_types: list[DatasetType] | None = None
    chat_template: str | None = None


class AdapterPreviewRequest(BaseModel):
    dataset_type: DatasetType = DatasetType.RAW
    sample_size: int = Field(default=200, ge=10, le=5000)
    adapter_id: str = "auto"
    adapter_config: dict[str, Any] | None = None
    field_mapping: dict[str, str] | None = None
    task_profile: str | None = None
    document_id: int | None = None
    preview_limit: int = Field(default=20, ge=5, le=100)


class AdapterPreferenceUpdateRequest(BaseModel):
    adapter_id: str = Field(..., min_length=1)
    adapter_config: dict[str, Any] | None = None
    field_mapping: dict[str, str] | None = None
    task_profile: str | None = None


class AdapterPreferenceAutoDetectRequest(BaseModel):
    dataset_type: DatasetType = DatasetType.RAW
    sample_size: int = Field(default=250, ge=10, le=5000)
    adapter_config: dict[str, Any] | None = None
    field_mapping: dict[str, str] | None = None
    task_profile: str | None = None
    document_id: int | None = None
    save: bool = True


def _resolve_split_config(
    *,
    profile_defaults: dict[str, float | int | str],
    provided_fields: set[str],
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    seed: int | None,
    chat_template: str | None,
) -> tuple[dict[str, float | int | str], list[str]]:
    profile_defaults_applied: list[str] = []

    default_train = 0.8
    default_val = 0.1
    default_test = 0.1
    default_seed = 42
    default_chat_template = "llama3"

    has_train = "train_ratio" in provided_fields and train_ratio is not None
    has_val = "val_ratio" in provided_fields and val_ratio is not None
    has_test = "test_ratio" in provided_fields and test_ratio is not None
    has_seed = "seed" in provided_fields and seed is not None
    has_template = "chat_template" in provided_fields and chat_template is not None

    resolved_train = float(train_ratio if has_train else profile_defaults.get("train_ratio", default_train))
    resolved_val = float(val_ratio if has_val else profile_defaults.get("val_ratio", default_val))
    resolved_test = float(test_ratio if has_test else profile_defaults.get("test_ratio", default_test))
    resolved_seed = int(seed if has_seed else profile_defaults.get("seed", default_seed))
    resolved_template = str(chat_template if has_template else profile_defaults.get("chat_template", default_chat_template))

    if not has_train and "train_ratio" in profile_defaults:
        profile_defaults_applied.append("train_ratio")
    if not has_val and "val_ratio" in profile_defaults:
        profile_defaults_applied.append("val_ratio")
    if not has_test and "test_ratio" in profile_defaults:
        profile_defaults_applied.append("test_ratio")
    if not has_seed and "seed" in profile_defaults:
        profile_defaults_applied.append("seed")
    if not has_template and "chat_template" in profile_defaults:
        profile_defaults_applied.append("chat_template")

    resolved = {
        "train_ratio": resolved_train,
        "val_ratio": resolved_val,
        "test_ratio": resolved_test,
        "seed": resolved_seed,
        "chat_template": resolved_template,
    }
    return resolved, sorted(set(profile_defaults_applied))


@router.post("/split/effective-config")
async def split_effective_config(
    project_id: int,
    req: SplitEffectiveConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview effective split config after domain runtime defaults are applied."""
    try:
        runtime = await resolve_project_domain_runtime(db, project_id)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)
    effective_contract = runtime.get("effective_contract")
    profile_defaults = get_dataset_split_defaults(effective_contract)
    provided = set(req.model_fields_set)

    resolved, profile_defaults_applied = _resolve_split_config(
        profile_defaults=profile_defaults,
        provided_fields=provided,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        test_ratio=req.test_ratio,
        seed=req.seed,
        chat_template=req.chat_template,
    )

    if abs((float(resolved["train_ratio"]) + float(resolved["val_ratio"]) + float(resolved["test_ratio"])) - 1.0) > 1e-6:
        raise HTTPException(400, "train_ratio + val_ratio + test_ratio must equal 1.0")

    return {
        "domain_pack_applied": runtime.get("domain_pack_applied"),
        "domain_pack_source": runtime.get("domain_pack_source"),
        "domain_profile_applied": runtime.get("domain_profile_applied"),
        "domain_profile_source": runtime.get("domain_profile_source"),
        "profile_split_defaults": profile_defaults or None,
        "resolved_split_config": resolved,
        "profile_defaults_applied": profile_defaults_applied,
        "include_types": [t.value for t in req.include_types] if req.include_types else None,
    }


@router.post("/split")
async def split(
    project_id: int,
    req: SplitRequest,
    db: AsyncSession = Depends(get_db),
):
    """Split combined data into train/validation/test JSONL datasets."""
    try:
        runtime = await resolve_project_domain_runtime(db, project_id)
        effective_contract = runtime.get("effective_contract")
        profile_defaults = get_dataset_split_defaults(effective_contract)
        provided = set(req.model_fields_set)

        resolved, profile_defaults_applied = _resolve_split_config(
            profile_defaults=profile_defaults,
            provided_fields=provided,
            train_ratio=req.train_ratio,
            val_ratio=req.val_ratio,
            test_ratio=req.test_ratio,
            seed=req.seed,
            chat_template=req.chat_template,
        )

        if abs((float(resolved["train_ratio"]) + float(resolved["val_ratio"]) + float(resolved["test_ratio"])) - 1.0) > 1e-6:
            raise HTTPException(400, "train_ratio + val_ratio + test_ratio must equal 1.0")

        explicit_adapter_fields = {"adapter_id", "adapter_config", "field_mapping", "task_profile"}
        has_explicit_adapter = any(field in provided for field in explicit_adapter_fields)
        if has_explicit_adapter:
            adapter_id = str(req.adapter_id or "default-canonical").strip() or "default-canonical"
            adapter_config = dict(req.adapter_config or {})
            field_mapping = dict(req.field_mapping or {})
            task_profile = str(req.task_profile or "").strip() or None
            adapter_source = "request"
        else:
            preset = await resolve_project_dataset_adapter_preference(db, project_id)
            adapter_id = str(preset.get("adapter_id") or "default-canonical")
            adapter_config = dict(preset.get("adapter_config") or {})
            field_mapping = dict(preset.get("field_mapping") or {})
            task_profile = str(preset.get("task_profile") or "").strip() or None
            adapter_source = str(preset.get("source") or "default")

        manifest = await split_dataset(
            db=db,
            project_id=project_id,
            train_ratio=float(resolved["train_ratio"]),
            val_ratio=float(resolved["val_ratio"]),
            test_ratio=float(resolved["test_ratio"]),
            seed=int(resolved["seed"]),
            include_types=[t.value for t in req.include_types] if req.include_types else None,
            chat_template=str(resolved["chat_template"]),
            adapter_id=adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=task_profile,
        )
        manifest["domain_pack_applied"] = runtime.get("domain_pack_applied")
        manifest["domain_pack_source"] = runtime.get("domain_pack_source")
        manifest["domain_profile_applied"] = runtime.get("domain_profile_applied")
        manifest["domain_profile_source"] = runtime.get("domain_profile_source")
        manifest["profile_split_defaults"] = profile_defaults or None
        manifest["resolved_split_config"] = resolved
        manifest["profile_defaults_applied"] = sorted(set(profile_defaults_applied))
        manifest["adapter_preference_source"] = adapter_source
        return manifest
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.get("/preview")
async def preview(
    project_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    chat_template: str = "llama3",
    include_types: list[DatasetType] | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Preview combined dataset entries before splitting."""
    entries = await combine_datasets(db, project_id, include_types, chat_template)
    return {
        "total": len(entries),
        "preview": entries[:limit],
        "chat_template": chat_template,
        "included_types": [t.value for t in include_types] if include_types else None,
    }


@router.get("/adapters/catalog")
async def adapter_catalog():
    """List available dataset adapters and plugin load status."""
    return list_data_adapter_catalog()


@router.get("/adapter-preference")
async def get_adapter_preference(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get resolved adapter preference (project override, pack default, or platform default)."""
    try:
        return await resolve_project_dataset_adapter_preference(db, project_id)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.put("/adapter-preference")
async def set_adapter_preference(
    project_id: int,
    req: AdapterPreferenceUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Persist project-level adapter preference used as split default."""
    try:
        return await save_project_dataset_adapter_preference(
            db,
            project_id,
            adapter_id=req.adapter_id,
            adapter_config=req.adapter_config,
            field_mapping=req.field_mapping,
            task_profile=req.task_profile,
        )
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Project "):
            raise HTTPException(404, detail)
        raise HTTPException(400, detail)


@router.post("/adapter-preference/auto-detect")
async def auto_detect_adapter_preference(
    project_id: int,
    req: AdapterPreferenceAutoDetectRequest,
    db: AsyncSession = Depends(get_db),
):
    """Auto-detect best adapter from project sample and optionally persist it."""
    try:
        preview = await preview_project_data_adapter(
            db=db,
            project_id=project_id,
            dataset_type=req.dataset_type,
            sample_size=req.sample_size,
            adapter_id="auto",
            adapter_config=req.adapter_config,
            field_mapping=req.field_mapping,
            task_profile=req.task_profile,
            document_id=req.document_id,
            preview_limit=20,
        )
        resolved_adapter_id = str(preview.get("resolved_adapter_id") or "default-canonical")
        resolved_task_profile = str(
            req.task_profile
            or preview.get("resolved_task_profile")
            or ""
        ).strip() or None
        if req.save:
            preference = await save_project_dataset_adapter_preference(
                db,
                project_id,
                adapter_id=resolved_adapter_id,
                adapter_config=req.adapter_config,
                field_mapping=req.field_mapping,
                task_profile=resolved_task_profile,
            )
        else:
            preference = {
                "project_id": project_id,
                "source": "auto_detect_preview",
                "adapter_id": resolved_adapter_id,
                "adapter_config": dict(req.adapter_config or {}),
                "field_mapping": dict(req.field_mapping or {}),
                "task_profile": resolved_task_profile,
            }
        return {
            "preference": preference,
            "preview": preview,
            "saved": bool(req.save),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/adapters/reload")
async def reload_adapter_plugins():
    """Reload dataset adapter plugin modules configured in settings."""
    clear_plugin_data_adapters()
    load_result = load_data_adapter_plugins_from_settings(force_reload=True)
    return {
        "reload": load_result,
        "catalog": list_data_adapter_catalog(),
    }


@router.post("/adapters/preview")
async def adapter_preview(
    project_id: int,
    req: AdapterPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview adapter mapping output and validation for sampled project records."""
    try:
        return await preview_project_data_adapter(
            db=db,
            project_id=project_id,
            dataset_type=req.dataset_type,
            sample_size=req.sample_size,
            adapter_id=req.adapter_id,
            adapter_config=req.adapter_config,
            field_mapping=req.field_mapping,
            task_profile=req.task_profile,
            document_id=req.document_id,
            preview_limit=req.preview_limit,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/profile")
async def profile(
    project_id: int,
    req: ProfileRequest,
    db: AsyncSession = Depends(get_db),
):
    """Inspect schema/normalization coverage for a project dataset."""
    try:
        return await profile_project_dataset(
            db=db,
            project_id=project_id,
            dataset_type=req.dataset_type,
            sample_size=req.sample_size,
            document_id=req.document_id,
            field_mapping=req.field_mapping,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
