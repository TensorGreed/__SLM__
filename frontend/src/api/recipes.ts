/**
 * Typed wrappers around the Theme 2 recipe registry + shape sniffer.
 *
 * Endpoints:
 *   GET  /api/recipes              — catalog + metadata
 *   GET  /api/recipes/{id}         — single recipe
 *   POST /api/recipes/sniff        — rank recipes by column headers
 *
 * A Recipe answers "what kind of model do I want to train?" — task
 * shape, adapter, scoring mode, gold-set template, suggested base
 * model. The picker shows the top-ranked recipes after dataset
 * introspection so a fresh project lands on a sniffed task shape
 * before touching the dense config surface.
 */

import api from './client';

export interface ShapeColumnSpec {
    name_patterns: string[];
    column_role: string;
    required: boolean;
}

export interface ShapeSignatureSpec {
    columns: ShapeColumnSpec[];
    base_confidence: number;
}

export interface GoldFieldSpec {
    name: string;
    required: boolean;
    description: string;
}

export interface GoldTemplate {
    shape_label: string;
    min_rows_recommended: number;
    fields: GoldFieldSpec[];
    example_row: Record<string, unknown>;
}

export interface Recipe {
    id: string;
    name: string;
    headline: string;
    description: string;
    icon: string;

    task_profile: string;
    adapter_id: string;
    scoring_mode: 'field_match' | 'span_set' | string;

    default_input_column: string;
    default_output_column: string;

    suggested_base_model: string;
    alt_base_models: string[];

    target_profile: string;
    training_plan_profile: string;
    eval_pack_id: string;

    gold_template: GoldTemplate;

    sample_eval_prompts: string[];
    data_acquisition_hints: string[];

    shape_signatures: ShapeSignatureSpec[];

    catalog_source: string;
    catalog_version: string;
    is_builtin: boolean;
}

export interface RecipeCatalogResponse {
    catalog_version: string;
    catalog_source: string;
    recipe_count: number;
    recipes: Recipe[];
}

export interface RecipeSuggestion {
    recipe_id: string;
    recipe_name: string;
    icon: string;
    confidence: number;
    /** Map of column_role → matched header (or null). Roles vary per recipe:
     * input/output/label/rationale/auxiliary. */
    matched_columns: Record<string, string | null>;
    signature_index: number;
    /** True when this is the generic-sft floor fallback rather than a real match. */
    fallback?: boolean;
}

export interface SniffResponse {
    headers: string[];
    suggestions: RecipeSuggestion[];
    top_recipe_id: string | null;
}

export async function listRecipes(): Promise<RecipeCatalogResponse> {
    const res = await api.get<RecipeCatalogResponse>('/recipes');
    return res.data;
}

export async function getRecipe(recipeId: string): Promise<Recipe> {
    const res = await api.get<Recipe>(`/recipes/${encodeURIComponent(recipeId)}`);
    return res.data;
}

export async function sniffRecipeFromHeaders(headers: string[]): Promise<SniffResponse> {
    const res = await api.post<SniffResponse>('/recipes/sniff', { headers });
    return res.data;
}
