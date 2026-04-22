/**
 * Autopilot Planner page (priority.md P6).
 *
 * Two-step flow built on P3:
 *   1. "Preview Plan" → POST /autopilot/repair-preview  → returns plan_token + config diff.
 *   2. "Apply Plan"   → POST /autopilot/repair-apply    → executes with the stored token.
 *
 * The page also exposes a "Run directly" shortcut that skips the persisted
 * preview and calls /autopilot/v2/orchestrate/run in one step — equivalent to
 * `brewslm autopilot run`, kept for automation flows that don't need the
 * explicit confirmation step.
 *
 * Strict-mode refusals (P5) come back as reason codes in the response
 * guardrails; the page renders them as prominent risk cards so operators know
 * exactly which auto-repair was blocked.
 */

import { type FormEvent, useMemo, useState } from 'react';
import { useOutletContext } from 'react-router-dom';

import api from '../api/client';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectAutopilotPage.css';

type PlanProfile = 'safe' | 'balanced' | 'max_quality' | 'fastest' | 'best_quality';

type TargetDeviceHint = '' | 'mobile' | 'laptop' | 'server';

interface RepairEntry {
    kind?: string;
    applied?: boolean;
    succeeded?: boolean;
    strict_mode_blocked?: boolean;
    original_intent?: string;
    rewritten_intent?: string | null;
    proposed_intent?: string;
    source?: string | null;
    from_profile?: string;
    to_profile?: string;
    from_target_profile_id?: string;
    to_target_profile_id?: string | null;
    reason?: string | null;
    reason_code?: string;
    [key: string]: unknown;
}

interface DecisionLogRow {
    step?: string;
    status?: string;
    summary?: string;
    changed?: boolean;
    blocker?: boolean;
    why?: string;
    before?: Record<string, unknown>;
    after?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
}

interface ConfigDiff {
    summary?: string;
    would_create_experiment?: boolean;
    would_start_training?: boolean;
    selected_profile?: string | null;
    effective_target_profile_id?: string | null;
    resolved_target_device?: string | null;
    repairs_planned?: RepairEntry[];
    safe_config_preview?: Record<string, unknown>;
    preflight_ok?: boolean;
    guardrails?: {
        blockers?: string[];
        warnings?: string[];
        reason_codes?: string[];
        can_run?: boolean;
    };
    decision_log_preview?: DecisionLogRow[];
    strict_mode?: boolean;
}

interface PreviewRecord {
    id: number;
    plan_token: string;
    project_id: number;
    intent?: string | null;
    expires_at?: string | null;
    created_at?: string | null;
    applied_at?: string | null;
    applied_run_id?: string | null;
    applied_by?: string | null;
    applied_reason?: string | null;
    state_hash: string;
    config_diff?: ConfigDiff;
}

interface PreviewResponse {
    preview: PreviewRecord;
    config_diff: ConfigDiff;
    dry_run_response: OrchestrateResponse;
    state_hash: string;
}

interface OrchestrateResponse {
    project_id: number;
    run_id?: string;
    dry_run?: boolean;
    strict_mode?: boolean;
    intent?: string;
    effective_target_profile_id?: string;
    resolved_target_device?: string;
    selected_profile?: string | null;
    guardrails?: Record<string, unknown>;
    readiness?: Record<string, unknown>;
    decision_log?: DecisionLogRow[];
    repairs?: Record<string, RepairEntry | null>;
    plan_v2?: Record<string, unknown>;
    experiment?: Record<string, unknown> | null;
    started?: boolean;
    start_result?: Record<string, unknown> | null;
    start_error?: string | null;
}

interface ApplyResponse {
    ok: boolean;
    preview: PreviewRecord;
    response: OrchestrateResponse;
}

type ApiErrorShape = {
    response?: {
        status?: number;
        data?: { detail?: unknown };
    };
    message?: string;
};

function extractErrorMessage(err: unknown): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') {
        const payload = detail as { reason?: string; message?: string };
        if (payload.message) return String(payload.message);
        if (payload.reason) return `${payload.reason}`;
    }
    return e?.message || 'Request failed.';
}

function summarizeRepair(entry: RepairEntry): string {
    if (entry.strict_mode_blocked) {
        return `Refused by strict mode (${entry.reason_code || 'STRICT_MODE'})`;
    }
    if (entry.applied || entry.succeeded) {
        return 'Applied';
    }
    return entry.reason || 'Skipped';
}

function friendlyRepairKind(kind: string): string {
    switch (kind) {
        case 'intent_rewrite':
            return 'Intent rewrite';
        case 'dataset_auto_prepare':
            return 'Dataset auto-prepare';
        case 'target_fallback':
            return 'Target fallback';
        case 'profile_autotune':
            return 'Profile autotune';
        default:
            return kind;
    }
}

function renderKeyValue(entry: [string, unknown]): string {
    const [key, value] = entry;
    if (value === null || value === undefined) return `${key}: —`;
    if (typeof value === 'object') return `${key}: ${JSON.stringify(value)}`;
    return `${key}: ${String(value)}`;
}

export default function ProjectAutopilotPage() {
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();

    // Form state
    const [intent, setIntent] = useState('Train a concise support assistant with JSON output.');
    const [intentRewrite, setIntentRewrite] = useState('');
    const [targetProfile, setTargetProfile] = useState('edge_gpu');
    const [targetDevice, setTargetDevice] = useState<TargetDeviceHint>('');
    const [planProfile, setPlanProfile] = useState<PlanProfile>('balanced');
    const [baseModel, setBaseModel] = useState('');
    const [availableVramGb, setAvailableVramGb] = useState('');
    const [primaryLanguage, setPrimaryLanguage] = useState('english');
    const [runName, setRunName] = useState('');
    const [description, setDescription] = useState('');
    const [autoApplyRewrite, setAutoApplyRewrite] = useState(true);
    const [autoPrepareData, setAutoPrepareData] = useState(true);
    const [allowTargetFallback, setAllowTargetFallback] = useState(true);
    const [allowProfileAutotune, setAllowProfileAutotune] = useState(true);
    const [force, setForce] = useState(false);

    // Flow state
    const [isPreviewing, setIsPreviewing] = useState(false);
    const [isApplying, setIsApplying] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const [preview, setPreview] = useState<PreviewRecord | null>(null);
    const [configDiff, setConfigDiff] = useState<ConfigDiff | null>(null);
    const [dryRunResponse, setDryRunResponse] = useState<OrchestrateResponse | null>(null);
    const [applyResult, setApplyResult] = useState<ApplyResponse | null>(null);

    const buildOrchestrateBody = () => {
        const body: Record<string, unknown> = {
            intent,
            target_profile_id: targetProfile,
            plan_profile: planProfile,
            primary_language: primaryLanguage,
            auto_apply_rewrite: autoApplyRewrite,
            auto_prepare_data: autoPrepareData,
            allow_target_fallback: allowTargetFallback,
            allow_profile_autotune: allowProfileAutotune,
        };
        if (targetDevice) body.target_device = targetDevice;
        if (intentRewrite.trim()) body.intent_rewrite = intentRewrite.trim();
        if (baseModel.trim()) body.base_model = baseModel.trim();
        if (runName.trim()) body.run_name = runName.trim();
        if (description.trim()) body.description = description.trim();
        const vram = Number.parseFloat(availableVramGb);
        if (Number.isFinite(vram) && vram > 0) body.available_vram_gb = vram;
        return body;
    };

    const clearResults = () => {
        setPreview(null);
        setConfigDiff(null);
        setDryRunResponse(null);
        setApplyResult(null);
    };

    const handlePreview = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!intent.trim()) {
            setErrorMessage('Intent is required.');
            return;
        }
        setErrorMessage('');
        setStatusMessage('');
        setIsPreviewing(true);
        clearResults();
        try {
            const body = { ...buildOrchestrateBody(), project_id: projectId };
            const res = await api.post<PreviewResponse>('/autopilot/repair-preview', body);
            setPreview(res.data.preview);
            setConfigDiff(res.data.config_diff);
            setDryRunResponse(res.data.dry_run_response);
            setStatusMessage(
                `Preview ready (token ${res.data.preview.plan_token.slice(0, 8)}…). `
                    + `Review the diff and click Apply when ready.`,
            );
        } catch (err) {
            setErrorMessage(extractErrorMessage(err));
        } finally {
            setIsPreviewing(false);
        }
    };

    const handleApply = async () => {
        if (!preview) return;
        setErrorMessage('');
        setStatusMessage('');
        setIsApplying(true);
        try {
            const res = await api.post<ApplyResponse>('/autopilot/repair-apply', {
                plan_token: preview.plan_token,
                force,
            });
            setApplyResult(res.data);
            setPreview(res.data.preview);
            setStatusMessage(
                res.data.response.started
                    ? `Plan applied. Training started (run_id ${res.data.response.run_id}).`
                    : `Plan applied but training did not start (${res.data.response.start_error || 'unknown reason'}).`,
            );
        } catch (err) {
            setErrorMessage(extractErrorMessage(err));
        } finally {
            setIsApplying(false);
        }
    };

    const handleRunDirectly = async () => {
        if (!intent.trim()) {
            setErrorMessage('Intent is required.');
            return;
        }
        setErrorMessage('');
        setStatusMessage('');
        setIsRunning(true);
        clearResults();
        try {
            const body = buildOrchestrateBody();
            body.dry_run = false;
            const res = await api.post<OrchestrateResponse>(
                `/projects/${projectId}/training/autopilot/v2/orchestrate/run`,
                body,
            );
            setDryRunResponse(res.data);
            setStatusMessage(
                res.data.started
                    ? `Autopilot run completed (run_id ${res.data.run_id}, started).`
                    : `Autopilot run completed (run_id ${res.data.run_id}) — training did not start.`,
            );
        } catch (err) {
            setErrorMessage(extractErrorMessage(err));
        } finally {
            setIsRunning(false);
        }
    };

    const guardrails = configDiff?.guardrails || {};
    const blockers = Array.isArray(guardrails.blockers) ? guardrails.blockers : [];
    const warnings = Array.isArray(guardrails.warnings) ? guardrails.warnings : [];
    const reasonCodes = Array.isArray(guardrails.reason_codes) ? guardrails.reason_codes : [];
    const strictRefusals = reasonCodes.filter((code) => code.startsWith('STRICT_MODE_REFUSED_'));

    const canRun = Boolean(guardrails.can_run ?? configDiff?.would_create_experiment);

    const applyDisabled = useMemo(
        () => !preview || isApplying || (!force && !canRun) || Boolean(preview.applied_at),
        [preview, isApplying, force, canRun],
    );

    const repairs = configDiff?.repairs_planned || [];
    const decisionPreview = configDiff?.decision_log_preview || [];
    const safeConfig = configDiff?.safe_config_preview || {};

    const postApplyDecisions = applyResult?.response?.decision_log || [];

    return (
        <div className="workspace-page autopilot-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Autopilot Planner</h2>
                    <p className="workspace-page-subtitle">
                        Preview a full autopilot plan before running it. See every auto-repair,
                        every blocker, and apply only when you're satisfied.
                    </p>
                </div>
                <div className="autopilot-header-badges">
                    {configDiff?.strict_mode ? (
                        <span className="badge badge-warning">Strict mode</span>
                    ) : null}
                    {preview?.plan_token ? (
                        <span className="badge badge-info" title={preview.plan_token}>
                            plan {preview.plan_token.slice(0, 8)}…
                        </span>
                    ) : null}
                </div>
            </section>

            {errorMessage ? (
                <div className="card autopilot-alert autopilot-alert-error">{errorMessage}</div>
            ) : null}
            {statusMessage ? (
                <div className="card autopilot-alert autopilot-alert-info">{statusMessage}</div>
            ) : null}

            <div className="autopilot-grid">
                <form className="card autopilot-form" onSubmit={handlePreview}>
                    <h3>Intent</h3>
                    <label className="autopilot-field">
                        <span>Plain-language intent</span>
                        <textarea
                            value={intent}
                            onChange={(e) => setIntent(e.target.value)}
                            rows={3}
                            placeholder="Describe what you want the model to do."
                            aria-label="Autopilot intent"
                        />
                    </label>
                    <label className="autopilot-field">
                        <span>Intent rewrite (optional)</span>
                        <textarea
                            value={intentRewrite}
                            onChange={(e) => setIntentRewrite(e.target.value)}
                            rows={2}
                            placeholder="Leave blank unless you want to override autopilot's rewrite."
                        />
                    </label>

                    <h3>Target</h3>
                    <div className="autopilot-field-row">
                        <label className="autopilot-field">
                            <span>Target profile</span>
                            <input
                                value={targetProfile}
                                onChange={(e) => setTargetProfile(e.target.value)}
                                placeholder="e.g. edge_gpu"
                            />
                        </label>
                        <label className="autopilot-field">
                            <span>Target device</span>
                            <select
                                value={targetDevice}
                                onChange={(e) => setTargetDevice(e.target.value as TargetDeviceHint)}
                            >
                                <option value="">auto</option>
                                <option value="mobile">mobile</option>
                                <option value="laptop">laptop</option>
                                <option value="server">server</option>
                            </select>
                        </label>
                    </div>

                    <h3>Plan</h3>
                    <div className="autopilot-field-row">
                        <label className="autopilot-field">
                            <span>Plan profile</span>
                            <select
                                value={planProfile}
                                onChange={(e) => setPlanProfile(e.target.value as PlanProfile)}
                            >
                                <option value="safe">safe</option>
                                <option value="balanced">balanced</option>
                                <option value="max_quality">max_quality</option>
                                <option value="fastest">fastest</option>
                                <option value="best_quality">best_quality</option>
                            </select>
                        </label>
                        <label className="autopilot-field">
                            <span>Primary language</span>
                            <input
                                value={primaryLanguage}
                                onChange={(e) => setPrimaryLanguage(e.target.value)}
                            />
                        </label>
                    </div>
                    <div className="autopilot-field-row">
                        <label className="autopilot-field">
                            <span>Base model (override)</span>
                            <input
                                value={baseModel}
                                onChange={(e) => setBaseModel(e.target.value)}
                                placeholder="e.g. microsoft/phi-2"
                            />
                        </label>
                        <label className="autopilot-field">
                            <span>Available VRAM (GB)</span>
                            <input
                                value={availableVramGb}
                                onChange={(e) => setAvailableVramGb(e.target.value)}
                                inputMode="decimal"
                                placeholder="optional"
                            />
                        </label>
                    </div>
                    <div className="autopilot-field-row">
                        <label className="autopilot-field">
                            <span>Run name</span>
                            <input value={runName} onChange={(e) => setRunName(e.target.value)} />
                        </label>
                        <label className="autopilot-field">
                            <span>Description</span>
                            <input value={description} onChange={(e) => setDescription(e.target.value)} />
                        </label>
                    </div>

                    <h3>Auto-repairs</h3>
                    <div className="autopilot-toggles">
                        <label>
                            <input
                                type="checkbox"
                                checked={autoApplyRewrite}
                                onChange={(e) => setAutoApplyRewrite(e.target.checked)}
                            />
                            Apply autopilot intent rewrites
                        </label>
                        <label>
                            <input
                                type="checkbox"
                                checked={autoPrepareData}
                                onChange={(e) => setAutoPrepareData(e.target.checked)}
                            />
                            Auto-prepare dataset
                        </label>
                        <label>
                            <input
                                type="checkbox"
                                checked={allowTargetFallback}
                                onChange={(e) => setAllowTargetFallback(e.target.checked)}
                            />
                            Allow target fallback
                        </label>
                        <label>
                            <input
                                type="checkbox"
                                checked={allowProfileAutotune}
                                onChange={(e) => setAllowProfileAutotune(e.target.checked)}
                            />
                            Allow profile autotune
                        </label>
                    </div>

                    <div className="autopilot-actions">
                        <button type="submit" className="btn btn-primary" disabled={isPreviewing}>
                            {isPreviewing ? 'Previewing…' : 'Preview Plan'}
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleRunDirectly}
                            disabled={isRunning}
                            title="Skip the preview and run immediately."
                        >
                            {isRunning ? 'Running…' : 'Run Directly'}
                        </button>
                    </div>
                </form>

                <div className="autopilot-result">
                    {!preview && !dryRunResponse && !applyResult ? (
                        <div className="card autopilot-empty">
                            <h3>No plan yet</h3>
                            <p>
                                Fill in the intent and click <strong>Preview Plan</strong> to see
                                what autopilot would do. Nothing mutates until you click
                                <strong> Apply</strong>.
                            </p>
                        </div>
                    ) : null}

                    {preview ? (
                        <section className="card autopilot-summary">
                            <div className="autopilot-summary-head">
                                <div>
                                    <span className="autopilot-summary-kicker">Plan token</span>
                                    <code className="autopilot-summary-token">{preview.plan_token}</code>
                                </div>
                                <div className="autopilot-summary-meta">
                                    {preview.expires_at ? (
                                        <span>Expires {new Date(preview.expires_at).toLocaleString()}</span>
                                    ) : null}
                                    <span title={preview.state_hash}>
                                        state {preview.state_hash.slice(0, 10)}…
                                    </span>
                                </div>
                            </div>
                            {configDiff?.summary ? (
                                <p className="autopilot-summary-text">{configDiff.summary}</p>
                            ) : null}
                            <div className="autopilot-summary-apply">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={force}
                                        onChange={(e) => setForce(e.target.checked)}
                                    />
                                    Force apply (skip state drift check)
                                </label>
                                <button
                                    className="btn btn-primary"
                                    type="button"
                                    onClick={handleApply}
                                    disabled={applyDisabled}
                                    title={canRun ? 'Execute this plan' : 'Plan has blockers; Force apply only'}
                                >
                                    {isApplying
                                        ? 'Applying…'
                                        : preview.applied_at
                                          ? 'Already applied'
                                          : 'Apply Plan'}
                                </button>
                            </div>
                        </section>
                    ) : null}

                    {configDiff ? (
                        <section className="card autopilot-guardrails">
                            <h3>Guardrails</h3>
                            <div className="autopilot-guardrails-row">
                                <span className={`badge ${canRun ? 'badge-success' : 'badge-danger'}`}>
                                    {canRun ? 'can_run: true' : 'can_run: false'}
                                </span>
                                {configDiff.selected_profile ? (
                                    <span className="badge badge-info">{`profile: ${configDiff.selected_profile}`}</span>
                                ) : null}
                                {configDiff.effective_target_profile_id ? (
                                    <span className="badge badge-info">
                                        {`target: ${configDiff.effective_target_profile_id}`}
                                    </span>
                                ) : null}
                                {configDiff.preflight_ok ? (
                                    <span className="badge badge-success">preflight ok</span>
                                ) : (
                                    <span className="badge badge-warning">preflight pending</span>
                                )}
                            </div>

                            {strictRefusals.length > 0 ? (
                                <div className="autopilot-risk-cards">
                                    {strictRefusals.map((code) => (
                                        <div key={code} className="autopilot-risk-card autopilot-risk-strict">
                                            <div className="autopilot-risk-title">Strict mode refusal</div>
                                            <div className="autopilot-risk-code">{code}</div>
                                        </div>
                                    ))}
                                </div>
                            ) : null}

                            {blockers.length > 0 ? (
                                <div className="autopilot-risk-cards">
                                    {blockers.map((msg, idx) => (
                                        <div
                                            key={`${idx}-${msg.slice(0, 20)}`}
                                            className="autopilot-risk-card autopilot-risk-blocker"
                                        >
                                            <div className="autopilot-risk-title">Blocker</div>
                                            <div>{msg}</div>
                                        </div>
                                    ))}
                                </div>
                            ) : null}

                            {warnings.length > 0 ? (
                                <div className="autopilot-risk-cards">
                                    {warnings.map((msg, idx) => (
                                        <div
                                            key={`${idx}-${msg.slice(0, 20)}`}
                                            className="autopilot-risk-card autopilot-risk-warning"
                                        >
                                            <div className="autopilot-risk-title">Warning</div>
                                            <div>{msg}</div>
                                        </div>
                                    ))}
                                </div>
                            ) : null}

                            {reasonCodes.length > 0 ? (
                                <div className="autopilot-reason-codes">
                                    {reasonCodes.map((code) => (
                                        <span key={code} className="autopilot-reason-code">
                                            {code}
                                        </span>
                                    ))}
                                </div>
                            ) : null}
                        </section>
                    ) : null}

                    {repairs.length > 0 ? (
                        <section className="card autopilot-repairs">
                            <h3>Auto-repairs planned</h3>
                            <table className="autopilot-repairs-table">
                                <thead>
                                    <tr>
                                        <th>Kind</th>
                                        <th>Status</th>
                                        <th>Before</th>
                                        <th>After</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {repairs.map((entry, idx) => (
                                        <tr key={`${entry.kind}-${idx}`}>
                                            <td>
                                                <strong>{friendlyRepairKind(entry.kind || 'unknown')}</strong>
                                            </td>
                                            <td>{summarizeRepair(entry)}</td>
                                            <td>
                                                {entry.from_profile
                                                    || entry.from_target_profile_id
                                                    || (entry.original_intent
                                                        ? `"${entry.original_intent}"`
                                                        : '—')}
                                            </td>
                                            <td>
                                                {entry.to_profile
                                                    || entry.to_target_profile_id
                                                    || (entry.rewritten_intent
                                                        ? `"${entry.rewritten_intent}"`
                                                        : entry.proposed_intent
                                                          ? `"${entry.proposed_intent}"`
                                                          : '—')}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </section>
                    ) : null}

                    {Object.keys(safeConfig).length > 0 ? (
                        <section className="card autopilot-diff">
                            <h3>Safe config preview</h3>
                            <div className="autopilot-diff-grid">
                                <div className="autopilot-diff-column">
                                    <h4>Before</h4>
                                    <pre className="autopilot-diff-code">— (project defaults)</pre>
                                </div>
                                <div className="autopilot-diff-column">
                                    <h4>After</h4>
                                    <pre className="autopilot-diff-code">
                                        {Object.entries(safeConfig)
                                            .filter(([key]) => !key.startsWith('_'))
                                            .map(renderKeyValue)
                                            .join('\n')}
                                    </pre>
                                </div>
                            </div>
                        </section>
                    ) : null}

                    {decisionPreview.length > 0 ? (
                        <section className="card autopilot-decision-preview">
                            <h3>Decision log preview</h3>
                            <ol className="autopilot-decision-list">
                                {decisionPreview.map((row, idx) => (
                                    <li
                                        key={`${idx}-${row.step || ''}`}
                                        className={row.blocker ? 'is-blocker' : ''}
                                    >
                                        <span className={`autopilot-decision-status status-${row.status || 'info'}`}>
                                            {String(row.status || 'info').toUpperCase()}
                                        </span>
                                        <span className="autopilot-decision-step">{row.step || 'step'}</span>
                                        <span className="autopilot-decision-summary">{row.summary || ''}</span>
                                    </li>
                                ))}
                            </ol>
                        </section>
                    ) : null}

                    {applyResult || dryRunResponse ? (
                        <section className="card autopilot-run-result">
                            <h3>Run result</h3>
                            <div className="autopilot-run-meta">
                                <span className="badge badge-info">
                                    {`run_id ${applyResult?.response?.run_id || dryRunResponse?.run_id || '—'}`}
                                </span>
                                <span
                                    className={`badge ${
                                        applyResult?.response?.started || dryRunResponse?.started
                                            ? 'badge-success'
                                            : 'badge-warning'
                                    }`}
                                >
                                    {(applyResult?.response?.started ?? dryRunResponse?.started)
                                        ? 'training started'
                                        : 'not started'}
                                </span>
                            </div>
                            {postApplyDecisions.length > 0 ? (
                                <ol className="autopilot-decision-list compact">
                                    {postApplyDecisions.map((row, idx) => (
                                        <li key={`post-${idx}`} className={row.blocker ? 'is-blocker' : ''}>
                                            <span
                                                className={`autopilot-decision-status status-${row.status || 'info'}`}
                                            >
                                                {String(row.status || 'info').toUpperCase()}
                                            </span>
                                            <span className="autopilot-decision-step">{row.step || 'step'}</span>
                                            <span className="autopilot-decision-summary">{row.summary || ''}</span>
                                        </li>
                                    ))}
                                </ol>
                            ) : null}
                        </section>
                    ) : null}
                </div>
            </div>
        </div>
    );
}
