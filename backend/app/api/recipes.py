"""Recipe catalog + shape-sniffer API routes.

A Recipe answers "what kind of model do I want to train?" — task
shape, adapter, scoring mode, gold-set template, suggested base
model. The catalog endpoints expose the built-in recipe set; the
sniff endpoint takes a list of column headers (from a freshly
uploaded CSV/JSONL) and returns ranked recipe suggestions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from app.services import recipe_service

router = APIRouter(prefix="/recipes", tags=["Recipes"])


class SniffHeadersRequest(BaseModel):
    headers: list[str] = Field(
        default_factory=list,
        description="Column headers from the source file. Order does not matter.",
    )


@router.get("")
async def list_recipes():
    """List all built-in recipes plus catalog metadata."""
    return recipe_service.list_recipe_catalog()


@router.get("/{recipe_id}")
async def get_recipe(recipe_id: str):
    recipe = recipe_service.get_recipe(recipe_id)
    if recipe is None:
        raise HTTPException(404, f"Recipe '{recipe_id}' not found")
    return recipe.model_dump()


@router.post("/sniff")
async def sniff_recipe_from_headers(request: SniffHeadersRequest):
    """Rank recipes by how well each one matches a list of column headers."""
    suggestions = recipe_service.sniff_recipe_from_headers(request.headers)
    return {
        "headers": request.headers,
        "suggestions": suggestions,
        "top_recipe_id": suggestions[0]["recipe_id"] if suggestions else None,
    }
