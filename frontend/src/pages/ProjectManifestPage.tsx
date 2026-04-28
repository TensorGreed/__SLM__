/**
 * ProjectManifestPage — Pipeline-as-Code import / validate / diff / apply
 * surface (priority.md P24).
 *
 * The page mounts at two routes:
 *   - `/project/:id/manifest` — diff and apply against an existing project.
 *   - `/manifest/import`      — apply a manifest end-to-end into a brand
 *                                new project (no project_id in path).
 *
 * Backend endpoints used:
 *   - POST /api/manifest/validate
 *   - POST /api/projects/{id}/manifest/diff
 *   - POST /api/projects/{id}/manifest/apply
 *   - POST /api/manifest/apply  (new project)
 *   - GET  /api/projects/{id}/manifest/export?format=json (summary card)
 *   - GET  /api/projects/{id}/manifest/export?format=yaml (export button)
 */

import {
    type ChangeEvent,
    type DragEvent,
    useCallback,
    useMemo,
    useRef,
    useState,
} from 'react';
import { useNavigate, useOutletContext, useParams } from 'react-router-dom';

import api from '../api/client';
import ManifestExportButton from '../components/manifest/ManifestExportButton';
import ManifestSummaryCard from '../components/manifest/ManifestSummaryCard';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import type {
    ManifestApplyAction,
    ManifestApplyPlan,
    ManifestApplyResult,
    ManifestValidationResult,
} from '../types/manifest';

import './ProjectManifestPage.css';

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Request failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    if (detail && typeof detail === 'object') {
        const payload = detail as { reason?: string; message?: string };
        if (payload.message) return String(payload.message);
        if (payload.reason) return String(payload.reason);
    }
    return e?.message || fallback;
}

function operationVariant(operation: string): string {
    switch (operation) {
        case 'create':
            return 'badge-success';
        case 'update':
            return 'badge-warning';
        case 'delete':
            return 'badge-danger';
        case 'noop':
        default:
            return 'badge-info';
    }
}

function groupActionsByTarget(actions: ManifestApplyAction[]): Record<string, ManifestApplyAction[]> {
    const groups: Record<string, ManifestApplyAction[]> = {};
    for (const action of actions) {
        const key = action.target || 'other';
        if (!groups[key]) groups[key] = [];
        groups[key].push(action);
    }
    return groups;
}

export default function ProjectManifestPage() {
    const params = useParams();
    const navigate = useNavigate();
    const workspace = useOutletContext<ProjectWorkspaceContextValue | null>();
    const routeProjectId = params.id ? Number.parseInt(params.id, 10) : null;
    const projectId = workspace?.projectId ?? (Number.isFinite(routeProjectId) ? routeProjectId : null);
    const projectName = workspace?.project?.name ?? '';
    const isNewProjectFlow = projectId == null;

    const [manifestText, setManifestText] = useState('');
    const [fileName, setFileName] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [planOnly, setPlanOnly] = useState(false);

    const [validation, setValidation] = useState<ManifestValidationResult | null>(null);
    const [diffPlan, setDiffPlan] = useState<ManifestApplyPlan | null>(null);
    const [applyResult, setApplyResult] = useState<ManifestApplyResult | null>(null);

    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const [validating, setValidating] = useState(false);
    const [diffing, setDiffing] = useState(false);
    const [applying, setApplying] = useState(false);

    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const validationErrors = validation?.errors ?? [];
    const validationWarnings = validation?.warnings ?? [];
    const hasValidationErrors = validationErrors.length > 0;

    const applyDisabled = useMemo(() => {
        if (applying) return true;
        if (!manifestText.trim()) return true;
        if (validation && !validation.ok) return true;
        return false;
    }, [applying, manifestText, validation]);

    const clearResults = () => {
        setValidation(null);
        setDiffPlan(null);
        setApplyResult(null);
        setStatusMessage(null);
        setErrorMessage(null);
    };

    const ingestFile = useCallback((file: File | null) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            const text = typeof reader.result === 'string' ? reader.result : '';
            setManifestText(text);
            setFileName(file.name);
            clearResults();
        };
        reader.onerror = () => {
            setErrorMessage('Could not read the selected file.');
        };
        reader.readAsText(file);
    }, []);

    const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0] || null;
        ingestFile(file);
        event.target.value = '';
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setDragActive(false);
        const file = event.dataTransfer?.files?.[0] || null;
        ingestFile(file);
    };

    const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        if (!dragActive) setDragActive(true);
    };

    const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setDragActive(false);
    };

    const handleValidate = useCallback(async () => {
        if (!manifestText.trim()) {
            setErrorMessage('Paste or drop a manifest first.');
            return;
        }
        setErrorMessage(null);
        setStatusMessage(null);
        setValidating(true);
        try {
            const response = await api.post<ManifestValidationResult>('/manifest/validate', {
                manifest_yaml: manifestText,
            });
            setValidation(response.data);
            setStatusMessage(
                response.data.ok
                    ? 'Manifest is valid against the schema and project catalogs.'
                    : `Manifest has ${response.data.errors.length} error(s); fix them before applying.`,
            );
        } catch (err) {
            setErrorMessage(extractErrorMessage(err, 'Validation failed.'));
        } finally {
            setValidating(false);
        }
    }, [manifestText]);

    const handleDiff = useCallback(async () => {
        if (isNewProjectFlow || projectId == null) {
            setErrorMessage('Diff is only available for an existing project.');
            return;
        }
        if (!manifestText.trim()) {
            setErrorMessage('Paste or drop a manifest first.');
            return;
        }
        setErrorMessage(null);
        setStatusMessage(null);
        setDiffing(true);
        try {
            const response = await api.post<ManifestApplyPlan>(
                `/projects/${projectId}/manifest/diff`,
                { manifest_yaml: manifestText },
            );
            setDiffPlan(response.data);
            const summary = response.data.summary || {};
            const counts = Object.entries(summary)
                .map(([k, v]) => `${v} ${k}`)
                .join(', ');
            setStatusMessage(counts ? `Diff ready: ${counts}.` : 'Diff ready.');
        } catch (err) {
            setErrorMessage(extractErrorMessage(err, 'Diff failed.'));
        } finally {
            setDiffing(false);
        }
    }, [isNewProjectFlow, projectId, manifestText]);

    const handleApply = useCallback(async () => {
        if (!manifestText.trim()) {
            setErrorMessage('Paste or drop a manifest first.');
            return;
        }
        setErrorMessage(null);
        setStatusMessage(null);
        setApplying(true);
        try {
            const url = isNewProjectFlow
                ? '/manifest/apply'
                : `/projects/${projectId}/manifest/apply`;
            const response = await api.post<ManifestApplyResult>(url, {
                manifest_yaml: manifestText,
                plan_only: planOnly,
            });
            setApplyResult(response.data);
            setValidation(response.data.validation);
            setDiffPlan(response.data.plan);
            if (planOnly) {
                setStatusMessage('Plan-only run completed — no changes were written.');
                return;
            }
            if (response.data.validation && !response.data.validation.ok) {
                setStatusMessage('Apply blocked by validation errors. Nothing was written.');
                return;
            }
            const newProjectId = response.data.project_id;
            setStatusMessage(
                `Applied ${response.data.applied_actions.length} action(s) to project ${response.data.project_name}.`,
            );
            if (newProjectId) {
                navigate(`/project/${newProjectId}/pipeline/data`);
            }
        } catch (err) {
            setErrorMessage(extractErrorMessage(err, 'Apply failed.'));
        } finally {
            setApplying(false);
        }
    }, [isNewProjectFlow, projectId, manifestText, planOnly, navigate]);

    const groupedActions = diffPlan ? groupActionsByTarget(diffPlan.actions) : null;

    return (
        <div className="workspace-page manifest-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Pipeline as Code</h2>
                    <p className="workspace-page-subtitle">
                        Import, validate, diff, and apply a <code>brewslm.yaml</code> manifest. Strict
                        schema with structured errors — no silent fallbacks.
                    </p>
                </div>
                {!isNewProjectFlow && projectId != null && projectName && (
                    <ManifestExportButton projectId={projectId} projectName={projectName} />
                )}
            </section>

            {!isNewProjectFlow && projectId != null && projectName && (
                <ManifestSummaryCard projectId={projectId} projectName={projectName} />
            )}

            {errorMessage && (
                <div className="manifest-status is-error" role="alert">
                    {errorMessage}
                </div>
            )}
            {statusMessage && (
                <div className="manifest-status is-info">{statusMessage}</div>
            )}

            <section className="card">
                <h3 style={{ marginTop: 0 }}>Manifest source</h3>
                <div
                    className={`manifest-dropzone ${dragActive ? 'is-active' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => fileInputRef.current?.click()}
                    role="button"
                    tabIndex={0}
                    aria-label="Manifest file drop zone"
                >
                    <div>Drop a <strong>.yaml</strong>, <strong>.yml</strong>, or <strong>.json</strong> manifest here, or click to choose a file.</div>
                    {fileName && (
                        <div style={{ marginTop: 6, fontSize: 'var(--font-size-xs)', color: 'var(--text-tertiary)' }}>
                            Loaded: <code>{fileName}</code>
                        </div>
                    )}
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".yaml,.yml,.json,application/x-yaml,application/json,text/yaml,text/plain"
                        onChange={handleFileInputChange}
                        aria-label="Manifest file input"
                    />
                </div>
                <textarea
                    className="manifest-textarea"
                    placeholder="…or paste your manifest YAML/JSON here."
                    value={manifestText}
                    onChange={(event) => {
                        setManifestText(event.target.value);
                        clearResults();
                    }}
                    aria-label="Manifest text"
                    style={{ marginTop: 'var(--space-sm)' }}
                />

                <div className="manifest-actions" style={{ marginTop: 'var(--space-sm)' }}>
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => void handleValidate()}
                        disabled={validating || !manifestText.trim()}
                    >
                        {validating ? 'Validating…' : 'Validate'}
                    </button>
                    {!isNewProjectFlow && (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => void handleDiff()}
                            disabled={diffing || !manifestText.trim()}
                        >
                            {diffing ? 'Diffing…' : 'Diff against project'}
                        </button>
                    )}
                    <label className="manifest-toggle">
                        <input
                            type="checkbox"
                            checked={planOnly}
                            onChange={(event) => setPlanOnly(event.target.checked)}
                        />
                        Plan-only preview (do not write)
                    </label>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void handleApply()}
                        disabled={applyDisabled}
                        title={hasValidationErrors ? 'Validation errors block apply.' : 'Apply this manifest'}
                    >
                        {applying ? 'Applying…' : isNewProjectFlow ? 'Apply (create project)' : 'Apply'}
                    </button>
                </div>
            </section>

            {validation && (
                <section className="card">
                    <h3 style={{ marginTop: 0 }}>Validation</h3>
                    {validation.ok && validationWarnings.length === 0 ? (
                        <div className="manifest-validation-ok" role="status">
                            Manifest is clean — no errors, no warnings.
                        </div>
                    ) : (
                        <div className="manifest-issues">
                            {validationErrors.map((issue, idx) => (
                                <div
                                    key={`err-${idx}-${issue.code}`}
                                    className="manifest-issue-card is-error"
                                >
                                    <div className="issue-head">
                                        <span className="badge badge-danger">error</span>
                                        <span className="issue-code">{issue.code}</span>
                                        {issue.field && <span className="issue-field">{issue.field}</span>}
                                    </div>
                                    <div>{issue.message}</div>
                                    {issue.actionable_fix && (
                                        <div className="issue-fix">Fix: {issue.actionable_fix}</div>
                                    )}
                                </div>
                            ))}
                            {validationWarnings.map((issue, idx) => (
                                <div
                                    key={`warn-${idx}-${issue.code}`}
                                    className="manifest-issue-card is-warning"
                                >
                                    <div className="issue-head">
                                        <span className="badge badge-warning">warning</span>
                                        <span className="issue-code">{issue.code}</span>
                                        {issue.field && <span className="issue-field">{issue.field}</span>}
                                    </div>
                                    <div>{issue.message}</div>
                                    {issue.actionable_fix && (
                                        <div className="issue-fix">Fix: {issue.actionable_fix}</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            )}

            {diffPlan && groupedActions && (
                <section className="card manifest-diff">
                    <h3 style={{ marginTop: 0 }}>Diff plan</h3>
                    <div className="manifest-diff-summary">
                        <span className="badge badge-info">project: {diffPlan.project_name}</span>
                        {Object.entries(diffPlan.summary || {}).map(([key, value]) => (
                            <span key={key} className={`badge ${operationVariant(key)}`}>
                                {value} {key}
                            </span>
                        ))}
                    </div>
                    {diffPlan.warnings && diffPlan.warnings.length > 0 && (
                        <div className="manifest-issues">
                            {diffPlan.warnings.map((warning, idx) => (
                                <div key={`plan-warn-${idx}`} className="manifest-issue-card is-warning">
                                    <div className="issue-head">
                                        <span className="badge badge-warning">warning</span>
                                    </div>
                                    <div>{warning}</div>
                                </div>
                            ))}
                        </div>
                    )}
                    {Object.entries(groupedActions).map(([target, actions]) => (
                        <div key={target} className="manifest-diff-group">
                            <h4>{target} ({actions.length})</h4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                {actions.map((action, idx) => (
                                    <div
                                        key={`${target}-${action.operation}-${action.name ?? idx}`}
                                        className="manifest-diff-action"
                                    >
                                        <div className="action-head">
                                            <span className={`badge ${operationVariant(action.operation)}`}>
                                                {action.operation}
                                            </span>
                                            <span className="action-name">
                                                {action.name || target}
                                            </span>
                                            {action.reason && (
                                                <span className="action-reason">— {action.reason}</span>
                                            )}
                                        </div>
                                        {action.fields_changed && action.fields_changed.length > 0 && (
                                            <div className="action-fields">
                                                fields_changed: {action.fields_changed.join(', ')}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </section>
            )}

            {applyResult && (
                <section className="card">
                    <h3 style={{ marginTop: 0 }}>Apply result</h3>
                    <div className="manifest-diff-summary">
                        <span className="badge badge-info">
                            project_id: {applyResult.project_id}
                        </span>
                        <span className={`badge ${applyResult.plan_only ? 'badge-info' : 'badge-success'}`}>
                            {applyResult.plan_only ? 'plan-only' : 'applied'}
                        </span>
                        <span className="badge badge-info">
                            {applyResult.applied_actions.length} action(s) written
                        </span>
                    </div>
                </section>
            )}
        </div>
    );
}
