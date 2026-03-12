import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';

import api from '../api/client';
import { useProjectStore } from '../stores/projectStore';
import type {
    PipelineRecipeApplyResponse,
    PipelineRecipeCatalogResponse,
    PipelineRecipeRunListResponse,
    PipelineRecipeRunRecord,
    PipelineRecipeRunResponse,
    PipelineRecipeResolveResponse,
    PipelineRecipeStateResponse,
    PipelineRecipeSummary,
} from '../types';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectRecipesPage.css';

function extractErrorMessage(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) {
            return detail;
        }
    }
    if (error instanceof Error) {
        return error.message;
    }
    return 'Request failed';
}

function normalizeStringList(value: unknown): string[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value
        .map((item) => (typeof item === 'string' ? item.trim() : ''))
        .filter((item) => item.length > 0);
}

function parseObjectJson(raw: string): Record<string, unknown> | null {
    const trimmed = raw.trim();
    if (!trimmed) {
        return {};
    }
    const parsed = JSON.parse(trimmed) as unknown;
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        throw new Error('Overrides JSON must be an object.');
    }
    return parsed as Record<string, unknown>;
}

function stateRecipeId(state: Record<string, unknown> | null | undefined): string {
    const raw = state?.active_recipe_id;
    return typeof raw === 'string' ? raw : '';
}

function stateUpdatedAt(state: Record<string, unknown> | null | undefined): string {
    const raw = state?.updated_at;
    return typeof raw === 'string' ? raw : '';
}

export default function ProjectRecipesPage() {
    const navigate = useNavigate();
    const { projectId, refreshPipelineStatus } = useOutletContext<ProjectWorkspaceContextValue>();
    const fetchProject = useProjectStore((state) => state.fetchProject);

    const [catalog, setCatalog] = useState<PipelineRecipeCatalogResponse | null>(null);
    const [recipeState, setRecipeState] = useState<PipelineRecipeStateResponse | null>(null);
    const [selectedRecipeId, setSelectedRecipeId] = useState('');
    const [overridesJson, setOverridesJson] = useState('{}');
    const [includePreflight, setIncludePreflight] = useState(true);
    const [enforcePreflightOk, setEnforcePreflightOk] = useState(false);
    const [markActive, setMarkActive] = useState(true);
    const [runAsync, setRunAsync] = useState(true);
    const [runBackend, setRunBackend] = useState('celery');
    const [runMaxRetries, setRunMaxRetries] = useState(0);
    const [resumeNodeId, setResumeNodeId] = useState('');
    const [resolveResult, setResolveResult] = useState<PipelineRecipeResolveResponse | null>(null);
    const [applyResult, setApplyResult] = useState<PipelineRecipeApplyResponse | null>(null);
    const [recipeRuns, setRecipeRuns] = useState<PipelineRecipeRunRecord[]>([]);
    const [selectedRecipeRunId, setSelectedRecipeRunId] = useState('');
    const [selectedRecipeRun, setSelectedRecipeRun] = useState<PipelineRecipeRunRecord | null>(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [isLoadingCatalog, setIsLoadingCatalog] = useState(false);
    const [isLoadingRuns, setIsLoadingRuns] = useState(false);
    const [isResolving, setIsResolving] = useState(false);
    const [isApplying, setIsApplying] = useState(false);
    const [isRunningRecipe, setIsRunningRecipe] = useState(false);
    const [isControllingRun, setIsControllingRun] = useState(false);

    const loadCatalog = useCallback(async () => {
        setIsLoadingCatalog(true);
        setErrorMessage('');
        try {
            const res = await api.get<PipelineRecipeCatalogResponse>(`/projects/${projectId}/pipeline/recipes`);
            setCatalog(res.data);
        } catch (error) {
            setCatalog(null);
            setErrorMessage(`Failed to load recipes: ${extractErrorMessage(error)}`);
        } finally {
            setIsLoadingCatalog(false);
        }
    }, [projectId]);

    const loadState = useCallback(async () => {
        try {
            const res = await api.get<PipelineRecipeStateResponse>(`/projects/${projectId}/pipeline/recipes/state`);
            setRecipeState(res.data);
        } catch {
            setRecipeState(null);
        }
    }, [projectId]);

    const loadRuns = useCallback(async () => {
        setIsLoadingRuns(true);
        try {
            const res = await api.get<PipelineRecipeRunListResponse>(`/projects/${projectId}/pipeline/recipes/runs`, {
                params: { limit: 20 },
            });
            const rows = Array.isArray(res.data.runs) ? res.data.runs : [];
            setRecipeRuns(rows);
            if (!selectedRecipeRunId && rows.length > 0) {
                setSelectedRecipeRunId(rows[0].recipe_run_id);
                setSelectedRecipeRun(rows[0]);
            } else if (selectedRecipeRunId) {
                const current = rows.find((item) => item.recipe_run_id === selectedRecipeRunId) || null;
                if (current) {
                    setSelectedRecipeRun(current);
                }
            }
        } catch {
            setRecipeRuns([]);
            if (!selectedRecipeRunId) {
                setSelectedRecipeRun(null);
            }
        } finally {
            setIsLoadingRuns(false);
        }
    }, [projectId, selectedRecipeRunId]);

    const loadRunDetail = useCallback(
        async (recipeRunId: string) => {
            if (!recipeRunId) {
                setSelectedRecipeRun(null);
                return;
            }
            try {
                const res = await api.get<PipelineRecipeRunRecord>(
                    `/projects/${projectId}/pipeline/recipes/runs/${recipeRunId}`,
                );
                setSelectedRecipeRun(res.data);
            } catch {
                setSelectedRecipeRun(null);
            }
        },
        [projectId],
    );

    useEffect(() => {
        void Promise.all([loadCatalog(), loadState(), loadRuns()]);
    }, [loadCatalog, loadState, loadRuns]);

    useEffect(() => {
        if (!catalog || selectedRecipeId) {
            return;
        }
        const activeRecipe = stateRecipeId(catalog.active_state);
        const recommendedRecipe =
            typeof catalog.recommended_recipe_id === 'string' ? catalog.recommended_recipe_id : '';
        const fallback =
            activeRecipe
            || recommendedRecipe
            || catalog.default_recipe_id
            || (catalog.recipes.length > 0 ? catalog.recipes[0].recipe_id : '');
        if (fallback) {
            setSelectedRecipeId(fallback);
        }
    }, [catalog, selectedRecipeId]);

    useEffect(() => {
        void loadRunDetail(selectedRecipeRunId);
    }, [selectedRecipeRunId, loadRunDetail]);

    useEffect(() => {
        const active = recipeRuns.some((item) => {
            const status = String(item.workflow_status || item.status || '').toLowerCase();
            return status === 'queued' || status === 'pending' || status === 'running';
        });
        if (!active) {
            return undefined;
        }
        const timer = window.setInterval(() => {
            void loadRuns();
            if (selectedRecipeRunId) {
                void loadRunDetail(selectedRecipeRunId);
            }
        }, 4000);
        return () => window.clearInterval(timer);
    }, [recipeRuns, loadRuns, loadRunDetail, selectedRecipeRunId]);

    const selectedRecipe = useMemo<PipelineRecipeSummary | null>(() => {
        if (!catalog) {
            return null;
        }
        return catalog.recipes.find((item) => item.recipe_id === selectedRecipeId) || null;
    }, [catalog, selectedRecipeId]);

    const activeState = recipeState?.state || catalog?.active_state || null;
    const warnings = normalizeStringList((applyResult || resolveResult)?.warnings);
    const preflight = ((applyResult || resolveResult)?.preflight || null) as Record<string, unknown> | null;
    const preflightOk = typeof preflight?.ok === 'boolean' ? preflight.ok : null;
    const resolvedPayload = (applyResult || resolveResult)?.resolved || null;
    const selectedRunStatus = String(
        selectedRecipeRun?.workflow_status || selectedRecipeRun?.status || '',
    ).toLowerCase();
    const canCancelSelectedRun = selectedRunStatus === 'queued' || selectedRunStatus === 'pending' || selectedRunStatus === 'running';

    const handleResolve = async () => {
        if (!selectedRecipeId) {
            setErrorMessage('Select a recipe first.');
            return;
        }

        let overrides: Record<string, unknown> | null = null;
        try {
            overrides = parseObjectJson(overridesJson);
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
            return;
        }

        setIsResolving(true);
        setErrorMessage('');
        setStatusMessage('');
        try {
            const res = await api.post<PipelineRecipeResolveResponse>(
                `/projects/${projectId}/pipeline/recipes/resolve`,
                {
                    recipe_id: selectedRecipeId,
                    overrides,
                    include_preflight: includePreflight,
                },
            );
            setResolveResult(res.data);
            setApplyResult(null);
            setStatusMessage(`Resolved recipe ${selectedRecipeId}.`);
        } catch (error) {
            setErrorMessage(`Resolve failed: ${extractErrorMessage(error)}`);
        } finally {
            setIsResolving(false);
        }
    };

    const handleApply = async () => {
        if (!selectedRecipeId) {
            setErrorMessage('Select a recipe first.');
            return;
        }

        let overrides: Record<string, unknown> | null = null;
        try {
            overrides = parseObjectJson(overridesJson);
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
            return;
        }

        setIsApplying(true);
        setErrorMessage('');
        setStatusMessage('');
        try {
            const res = await api.post<PipelineRecipeApplyResponse>(
                `/projects/${projectId}/pipeline/recipes/apply`,
                {
                    recipe_id: selectedRecipeId,
                    overrides,
                    include_preflight: includePreflight,
                    enforce_preflight_ok: enforcePreflightOk,
                    mark_active: markActive,
                },
            );
            setApplyResult(res.data);
            setResolveResult(null);
            setStatusMessage(`Applied recipe ${selectedRecipeId}.`);
            await Promise.all([loadCatalog(), loadState(), fetchProject(projectId), refreshPipelineStatus()]);
        } catch (error) {
            setErrorMessage(`Apply failed: ${extractErrorMessage(error)}`);
        } finally {
            setIsApplying(false);
        }
    };

    const handleRunRecipe = async () => {
        if (!selectedRecipeId) {
            setErrorMessage('Select a recipe first.');
            return;
        }

        let overrides: Record<string, unknown> | null = null;
        try {
            overrides = parseObjectJson(overridesJson);
        } catch (error) {
            setErrorMessage(extractErrorMessage(error));
            return;
        }

        setIsRunningRecipe(true);
        setErrorMessage('');
        setStatusMessage('');
        try {
            const res = await api.post<PipelineRecipeRunResponse>(
                `/projects/${projectId}/pipeline/recipes/run`,
                {
                    recipe_id: selectedRecipeId,
                    overrides,
                    include_preflight: includePreflight,
                    enforce_preflight_ok: enforcePreflightOk,
                    mark_active: markActive,
                    execution_backend: runBackend,
                    max_retries: Number.isFinite(runMaxRetries) ? Math.max(0, Math.min(5, runMaxRetries)) : 0,
                    stop_on_blocked: true,
                    stop_on_failure: true,
                    async_run: runAsync,
                    config: {},
                },
            );
            const body = res.data;
            const record = body.execution;
            setSelectedRecipeRunId(record.recipe_run_id);
            setSelectedRecipeRun(record);
            setStatusMessage(
                body.queued
                    ? `Recipe run queued (${record.recipe_run_id}) on ${runBackend}.`
                    : `Recipe run completed (${record.recipe_run_id}) with status ${record.workflow_status}.`,
            );
            await Promise.all([loadCatalog(), loadState(), loadRuns(), fetchProject(projectId), refreshPipelineStatus()]);
            if (record.recipe_run_id) {
                await loadRunDetail(record.recipe_run_id);
            }
        } catch (error) {
            setErrorMessage(`Run failed: ${extractErrorMessage(error)}`);
        } finally {
            setIsRunningRecipe(false);
        }
    };

    const handleCancelRun = async () => {
        if (!selectedRecipeRunId) {
            return;
        }
        setIsControllingRun(true);
        setErrorMessage('');
        setStatusMessage('');
        try {
            await api.post(`/projects/${projectId}/pipeline/recipes/runs/${selectedRecipeRunId}/cancel`);
            setStatusMessage(`Cancel requested for ${selectedRecipeRunId}.`);
            await Promise.all([loadRuns(), loadState()]);
            await loadRunDetail(selectedRecipeRunId);
        } catch (error) {
            setErrorMessage(`Cancel failed: ${extractErrorMessage(error)}`);
        } finally {
            setIsControllingRun(false);
        }
    };

    const handleRetryRun = async () => {
        if (!selectedRecipeRunId) {
            return;
        }
        setIsControllingRun(true);
        setErrorMessage('');
        setStatusMessage('');
        try {
            const res = await api.post<PipelineRecipeRunResponse>(
                `/projects/${projectId}/pipeline/recipes/runs/${selectedRecipeRunId}/retry`,
                {
                    execution_backend: runBackend,
                    max_retries: Number.isFinite(runMaxRetries) ? Math.max(0, Math.min(5, runMaxRetries)) : 0,
                    async_run: runAsync,
                    config: {},
                },
            );
            const nextRunId = res.data.recipe_run_id;
            setSelectedRecipeRunId(nextRunId);
            setStatusMessage(`Retry launched as ${nextRunId}.`);
            await Promise.all([loadRuns(), loadState()]);
            await loadRunDetail(nextRunId);
        } catch (error) {
            setErrorMessage(`Retry failed: ${extractErrorMessage(error)}`);
        } finally {
            setIsControllingRun(false);
        }
    };

    const handleResumeRun = async () => {
        if (!selectedRecipeRunId) {
            return;
        }
        setIsControllingRun(true);
        setErrorMessage('');
        setStatusMessage('');
        try {
            const res = await api.post<PipelineRecipeRunResponse>(
                `/projects/${projectId}/pipeline/recipes/runs/${selectedRecipeRunId}/resume`,
                {
                    execution_backend: runBackend,
                    max_retries: Number.isFinite(runMaxRetries) ? Math.max(0, Math.min(5, runMaxRetries)) : 0,
                    async_run: runAsync,
                    resume_from_node_id: resumeNodeId.trim() || undefined,
                    config: {},
                },
            );
            const nextRunId = res.data.recipe_run_id;
            setSelectedRecipeRunId(nextRunId);
            setStatusMessage(`Resume launched as ${nextRunId}.`);
            await Promise.all([loadRuns(), loadState()]);
            await loadRunDetail(nextRunId);
        } catch (error) {
            setErrorMessage(`Resume failed: ${extractErrorMessage(error)}`);
        } finally {
            setIsControllingRun(false);
        }
    };

    return (
        <div className="pipeline-recipes-page workspace-page">
            <div className="card pipeline-recipes-header">
                <div>
                    <h3>Pipeline Recipes</h3>
                    <p className="pipeline-recipes-subtitle">
                        Apply end-to-end blueprints that wire domain defaults, workflow template, training recipe,
                        dataset adapter preset, and evaluation pack.
                    </p>
                </div>
                <div className="pipeline-recipes-header-actions">
                    <button className="btn btn-secondary" onClick={() => void loadCatalog()} disabled={isLoadingCatalog}>
                        {isLoadingCatalog ? 'Refreshing...' : 'Refresh'}
                    </button>
                    <button
                        className="btn btn-ghost"
                        onClick={() => navigate(`/project/${projectId}/workflow`)}
                    >
                        Open Workflow Graph
                    </button>
                    <button
                        className="btn btn-ghost"
                        onClick={() => navigate(`/project/${projectId}/domain/packs`)}
                    >
                        Open Domain Packs
                    </button>
                </div>
            </div>

            <div className="pipeline-recipes-layout">
                <section className="card pipeline-recipes-catalog">
                    <h4>Recipe Catalog</h4>
                    <div className="pipeline-recipes-list">
                        {catalog?.recipes.map((recipe) => {
                            const active = recipe.recipe_id === selectedRecipeId;
                            const selected = stateRecipeId(activeState) === recipe.recipe_id;
                            const recommended = catalog.recommended_recipe_id === recipe.recipe_id;
                            return (
                                <button
                                    key={recipe.recipe_id}
                                    className={`pipeline-recipe-card ${active ? 'active' : ''}`}
                                    onClick={() => setSelectedRecipeId(recipe.recipe_id)}
                                >
                                    <div className="pipeline-recipe-card-head">
                                        <strong>{recipe.display_name}</strong>
                                        <div className="pipeline-recipe-badges">
                                            {recommended && <span className="badge badge-success">Recommended</span>}
                                            {selected && <span className="badge badge-accent">Active</span>}
                                        </div>
                                    </div>
                                    <div className="pipeline-recipe-card-id">{recipe.recipe_id}</div>
                                    <div className="pipeline-recipe-card-desc">{recipe.description}</div>
                                    <div className="pipeline-recipe-card-meta">
                                        <span>{recipe.category}</span>
                                        <span>v{recipe.version}</span>
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                    {catalog?.training_recipe_ids && (
                        <p className="pipeline-recipes-footnote">
                            Available training recipe ids: {catalog.training_recipe_ids.join(', ')}
                        </p>
                    )}
                    {catalog?.recommendation_context && (
                        <p className="pipeline-recipes-footnote">
                            Recommendation context:
                            {' '}
                            task_profile=
                            {String(catalog.recommendation_context.task_profile || 'auto')}
                            {' '}
                            • plan=
                            {String(catalog.recommendation_context.preferred_plan_profile || 'balanced')}
                            {' '}
                            • prefer_fast=
                            {String(catalog.recommendation_context.prefer_fast ?? 'auto')}
                        </p>
                    )}
                </section>

                <section className="card pipeline-recipes-controls">
                    <h4>Resolve or Apply</h4>

                    <label className="pipeline-recipes-label">Selected Recipe</label>
                    <select
                        className="input"
                        value={selectedRecipeId}
                        onChange={(event) => setSelectedRecipeId(event.target.value)}
                    >
                        {catalog?.recipes.map((recipe) => (
                            <option key={recipe.recipe_id} value={recipe.recipe_id}>
                                {recipe.display_name} ({recipe.recipe_id})
                            </option>
                        ))}
                    </select>

                    <label className="pipeline-recipes-label">Overrides JSON</label>
                    <textarea
                        className="input pipeline-recipes-json"
                        value={overridesJson}
                        onChange={(event) => setOverridesJson(event.target.value)}
                        spellCheck={false}
                    />

                    <div className="pipeline-recipes-toggle-grid">
                        <label className="pipeline-recipes-toggle">
                            <input
                                type="checkbox"
                                checked={includePreflight}
                                onChange={(event) => setIncludePreflight(event.target.checked)}
                            />
                            Include preflight
                        </label>
                        <label className="pipeline-recipes-toggle">
                            <input
                                type="checkbox"
                                checked={enforcePreflightOk}
                                onChange={(event) => setEnforcePreflightOk(event.target.checked)}
                            />
                            Enforce preflight pass on apply
                        </label>
                        <label className="pipeline-recipes-toggle">
                            <input
                                type="checkbox"
                                checked={markActive}
                                onChange={(event) => setMarkActive(event.target.checked)}
                            />
                            Mark recipe as active
                        </label>
                    </div>

                    <div className="pipeline-recipes-run-options">
                        <div>
                            <label className="pipeline-recipes-label">Run Backend</label>
                            <select
                                className="input"
                                value={runBackend}
                                onChange={(event) => setRunBackend(event.target.value)}
                            >
                                <option value="celery">celery</option>
                                <option value="local">local</option>
                                <option value="external">external</option>
                            </select>
                        </div>
                        <div>
                            <label className="pipeline-recipes-label">Max Retries</label>
                            <input
                                className="input"
                                type="number"
                                min={0}
                                max={5}
                                value={runMaxRetries}
                                onChange={(event) => {
                                    const parsed = Number.parseInt(event.target.value, 10);
                                    setRunMaxRetries(Number.isFinite(parsed) ? parsed : 0);
                                }}
                            />
                        </div>
                        <label className="pipeline-recipes-toggle">
                            <input
                                type="checkbox"
                                checked={runAsync}
                                onChange={(event) => setRunAsync(event.target.checked)}
                            />
                            Run async
                        </label>
                    </div>

                    <div className="pipeline-recipes-actions">
                        <button
                            className="btn btn-secondary"
                            onClick={() => void handleResolve()}
                            disabled={!selectedRecipeId || isResolving || isApplying}
                        >
                            {isResolving ? 'Resolving...' : 'Resolve Blueprint'}
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={() => void handleApply()}
                            disabled={!selectedRecipeId || isApplying || isResolving || isRunningRecipe}
                        >
                            {isApplying ? 'Applying...' : 'Apply Blueprint'}
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={() => void handleRunRecipe()}
                            disabled={!selectedRecipeId || isApplying || isResolving || isRunningRecipe}
                        >
                            {isRunningRecipe ? 'Running...' : 'Apply + Run Workflow'}
                        </button>
                    </div>

                    {selectedRecipe && (
                        <div className="pipeline-recipes-selected-meta">
                            <div>
                                <span className="pipeline-recipes-meta-label">Selected</span>
                                <strong>{selectedRecipe.display_name}</strong>
                            </div>
                            <div>
                                <span className="pipeline-recipes-meta-label">Recipe ID</span>
                                <code>{selectedRecipe.recipe_id}</code>
                            </div>
                        </div>
                    )}
                </section>
            </div>

            <section className="card pipeline-recipes-results">
                <div className="pipeline-recipes-results-head">
                    <h4>Recipe State and Resolution</h4>
                    {preflightOk !== null && (
                        <span className={`badge ${preflightOk ? 'badge-success' : 'badge-error'}`}>
                            Preflight {preflightOk ? 'PASS' : 'FAIL'}
                        </span>
                    )}
                </div>

                {statusMessage && <div className="pipeline-recipes-status">{statusMessage}</div>}
                {errorMessage && <div className="pipeline-recipes-error">{errorMessage}</div>}
                {warnings.length > 0 && (
                    <div className="pipeline-recipes-warning">
                        {warnings.map((warning) => (
                            <div key={warning}>{warning}</div>
                        ))}
                    </div>
                )}

                <div className="pipeline-recipes-state-grid">
                    <div>
                        <span className="pipeline-recipes-meta-label">Active recipe</span>
                        <strong>{stateRecipeId(activeState) || 'none'}</strong>
                    </div>
                    <div>
                        <span className="pipeline-recipes-meta-label">Last updated</span>
                        <strong>
                            {stateUpdatedAt(activeState)
                                ? new Date(stateUpdatedAt(activeState)).toLocaleString()
                                : 'n/a'}
                        </strong>
                    </div>
                </div>

                <div className="pipeline-recipes-runs">
                    <div className="pipeline-recipes-runs-head">
                        <h5>Recipe Runs</h5>
                        <button className="btn btn-ghost" onClick={() => void loadRuns()} disabled={isLoadingRuns}>
                            {isLoadingRuns ? 'Refreshing...' : 'Refresh Runs'}
                        </button>
                    </div>

                    {recipeRuns.length === 0 ? (
                        <p className="pipeline-recipes-empty">No recipe runs yet.</p>
                    ) : (
                        <div className="pipeline-recipes-runs-grid">
                            <div className="pipeline-recipes-runs-list">
                                {recipeRuns.map((run) => {
                                    const isActive = run.recipe_run_id === selectedRecipeRunId;
                                    const status = String(run.workflow_status || run.status || '').toLowerCase();
                                    return (
                                        <button
                                            key={run.recipe_run_id}
                                            className={`pipeline-recipe-run-card ${isActive ? 'active' : ''}`}
                                            onClick={() => setSelectedRecipeRunId(run.recipe_run_id)}
                                        >
                                            <div className="pipeline-recipe-run-card-head">
                                                <strong>{run.recipe_id}</strong>
                                                <span className="pipeline-recipe-run-status">{status || 'unknown'}</span>
                                            </div>
                                            <div className="pipeline-recipe-run-id">{run.recipe_run_id}</div>
                                            <div className="pipeline-recipe-run-meta">
                                                workflow #{run.workflow_run_id}
                                            </div>
                                        </button>
                                    );
                                })}
                            </div>
                            <div className="pipeline-recipes-run-detail">
                                {selectedRecipeRun ? (
                                    <>
                                        <div className="pipeline-recipes-run-detail-head">
                                            <strong>{selectedRecipeRun.recipe_run_id}</strong>
                                            <span
                                                className={`badge ${
                                                    selectedRunStatus === 'completed'
                                                        ? 'badge-success'
                                                        : selectedRunStatus === 'failed' || selectedRunStatus === 'blocked'
                                                          ? 'badge-error'
                                                          : 'badge'
                                                }`}
                                            >
                                                {selectedRunStatus || 'unknown'}
                                            </span>
                                        </div>
                                        <div className="pipeline-recipes-run-detail-grid">
                                            <div>
                                                <span className="pipeline-recipes-meta-label">Workflow Run</span>
                                                <strong>#{selectedRecipeRun.workflow_run_id}</strong>
                                            </div>
                                            <div>
                                                <span className="pipeline-recipes-meta-label">Backend</span>
                                                <strong>{selectedRecipeRun.execution_backend || 'n/a'}</strong>
                                            </div>
                                            <div>
                                                <span className="pipeline-recipes-meta-label">Started</span>
                                                <strong>
                                                    {selectedRecipeRun.started_at
                                                        ? new Date(selectedRecipeRun.started_at).toLocaleString()
                                                        : 'n/a'}
                                                </strong>
                                            </div>
                                            <div>
                                                <span className="pipeline-recipes-meta-label">Finished</span>
                                                <strong>
                                                    {selectedRecipeRun.finished_at
                                                        ? new Date(selectedRecipeRun.finished_at).toLocaleString()
                                                        : 'running'}
                                                </strong>
                                            </div>
                                        </div>
                                        <div className="pipeline-recipes-run-controls">
                                            <button
                                                className="btn btn-secondary"
                                                onClick={() => void handleRetryRun()}
                                                disabled={isControllingRun || isRunningRecipe || !selectedRecipeRunId}
                                            >
                                                {isControllingRun ? 'Working...' : 'Retry'}
                                            </button>
                                            <button
                                                className="btn btn-secondary"
                                                onClick={() => void handleCancelRun()}
                                                disabled={!canCancelSelectedRun || isControllingRun || isRunningRecipe}
                                            >
                                                Cancel
                                            </button>
                                            <input
                                                className="input"
                                                value={resumeNodeId}
                                                onChange={(event) => setResumeNodeId(event.target.value)}
                                                placeholder="Resume from node id (optional)"
                                            />
                                            <button
                                                className="btn btn-secondary"
                                                onClick={() => void handleResumeRun()}
                                                disabled={isControllingRun || isRunningRecipe || !selectedRecipeRunId}
                                            >
                                                Resume
                                            </button>
                                        </div>
                                        {selectedRecipeRun.workflow_run && (
                                            <div className="pipeline-recipes-run-links">
                                                <button
                                                    className="btn btn-ghost"
                                                    onClick={() => navigate(`/project/${projectId}/workflow`)}
                                                >
                                                    Open Workflow Run Monitor
                                                </button>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <p className="pipeline-recipes-empty">Select a run to inspect details.</p>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {resolvedPayload ? (
                    <pre className="pipeline-recipes-json-view">
                        {JSON.stringify(resolvedPayload, null, 2)}
                    </pre>
                ) : (
                    <p className="pipeline-recipes-empty">
                        Resolve or apply a recipe to preview concrete wiring details.
                    </p>
                )}
            </section>
        </div>
    );
}
