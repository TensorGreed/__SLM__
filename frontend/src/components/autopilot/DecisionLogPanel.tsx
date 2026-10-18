/**
 * Autopilot decision log — a Settings section (``?open=decisions``).
 *
 * Lists every step the Autopilot recorded for the project from
 * ``/autopilot/decisions``, grouped by run, with filters for stage, status,
 * action, reason code and run id. It used to be a floating drawer on every
 * workspace page (priority.md P6); since it is an audit tool, not a daily
 * one, it lives with the other long-tail tools under Settings.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ClipboardList, RefreshCw } from 'lucide-react';

import api from '../../api/client';
import EmptyState from '../shared/EmptyState';
import LoadingState from '../shared/LoadingState';
import './DecisionLogPanel.css';

export interface DecisionLogPanelProps {
    projectId: number;
}

interface DecisionRow {
    id: number;
    run_id: string;
    project_id: number | null;
    sequence: number;
    stage: string;
    status: string;
    action: string;
    reason_code: string | null;
    confidence: number | null;
    rationale: string | null;
    summary: string | null;
    actor: string;
    changed: boolean;
    safe: boolean;
    blocker: boolean;
    dry_run: boolean;
    intent: string | null;
    payload: Record<string, unknown>;
    created_at: string | null;
}

interface DecisionListResponse {
    items: DecisionRow[];
    limit: number;
    offset: number;
    returned: number;
}

const STAGE_HINTS = [
    'strict_mode_policy',
    'runtime_readiness',
    'initial_planning',
    'intent_rewrite',
    'dataset_auto_prepare',
    'target_fallback',
    'profile_autotune',
    'start_training',
    'final_guardrails',
    'rollback',
];

const STATUS_CHOICES = ['', 'applied', 'completed', 'blocked', 'failed', 'skipped', 'info', 'active'];
const ACTION_CHOICES = ['', 'applied', 'blocked', 'warned', 'skipped', 'rolled_back', 'info'];

function errorText(err: unknown): string {
    const e = err as { response?: { data?: { detail?: unknown } }; message?: string };
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') return JSON.stringify(detail);
    return e?.message || 'Failed to load decisions.';
}

export default function DecisionLogPanel({ projectId }: DecisionLogPanelProps) {
    const [rows, setRows] = useState<DecisionRow[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState('');
    const [stage, setStage] = useState('');
    const [status, setStatus] = useState('');
    const [action, setAction] = useState('');
    const [reasonCode, setReasonCode] = useState('');
    const [runId, setRunId] = useState('');
    const [expandedId, setExpandedId] = useState<number | null>(null);

    const fetchDecisions = useCallback(async () => {
        setErrorMessage('');
        setIsLoading(true);
        try {
            const params: Record<string, string | number> = { project_id: projectId, limit: 200 };
            if (stage.trim()) params.stage = stage.trim();
            if (status.trim()) params.status = status.trim();
            if (action.trim()) params.action = action.trim();
            if (reasonCode.trim()) params.reason_code = reasonCode.trim();
            if (runId.trim()) params.run_id = runId.trim();
            const res = await api.get<DecisionListResponse>('/autopilot/decisions', { params });
            setRows(res.data.items || []);
        } catch (err) {
            setErrorMessage(errorText(err));
        } finally {
            setIsLoading(false);
        }
    }, [projectId, stage, status, action, reasonCode, runId]);

    useEffect(() => {
        void fetchDecisions();
    }, [fetchDecisions]);

    const groupedByRun = useMemo(() => {
        const groups = new Map<string, DecisionRow[]>();
        for (const row of rows) {
            const key = row.run_id || '';
            const bucket = groups.get(key) || [];
            bucket.push(row);
            groups.set(key, bucket);
        }
        // Insertion order = created_at desc from the API.
        return Array.from(groups.entries());
    }, [rows]);

    const filtered = Boolean(stage || status || action || reasonCode || runId);

    return (
        <section className="decision-log" data-testid="decision-log">
            <header className="decision-log__head">
                <div className="decision-log__title">
                    <ClipboardList size={16} aria-hidden="true" />
                    <span>Autopilot decisions</span>
                </div>
                <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => void fetchDecisions()}
                    disabled={isLoading}
                    data-testid="decision-log-refresh"
                >
                    <RefreshCw size={14} aria-hidden="true" />
                    Refresh
                </button>
            </header>
            <p className="u-hint u-m0">
                Every step the Autopilot took on this project, newest run first. Each row says what it did, why, and what it changed.
            </p>

            <div className="decision-log__filters">
                <label>
                    <span>Stage</span>
                    <input
                        list="decision-log-stage-hints"
                        value={stage}
                        onChange={(e) => setStage(e.target.value)}
                        placeholder="e.g. start_training"
                        data-testid="decision-log-stage"
                    />
                    <datalist id="decision-log-stage-hints">
                        {STAGE_HINTS.map((hint) => (
                            <option key={hint} value={hint} />
                        ))}
                    </datalist>
                </label>
                <label>
                    <span>Status</span>
                    <select value={status} onChange={(e) => setStatus(e.target.value)} data-testid="decision-log-status">
                        {STATUS_CHOICES.map((choice) => (
                            <option key={choice} value={choice}>
                                {choice || 'any'}
                            </option>
                        ))}
                    </select>
                </label>
                <label>
                    <span>Action</span>
                    <select value={action} onChange={(e) => setAction(e.target.value)}>
                        {ACTION_CHOICES.map((choice) => (
                            <option key={choice} value={choice}>
                                {choice || 'any'}
                            </option>
                        ))}
                    </select>
                </label>
                <label>
                    <span>Reason code</span>
                    <input
                        value={reasonCode}
                        onChange={(e) => setReasonCode(e.target.value)}
                        placeholder="e.g. STRICT_MODE_REFUSED_TARGET_FALLBACK"
                    />
                </label>
                <label>
                    <span>Run id</span>
                    <input value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="Filter by autopilot run id" />
                </label>
            </div>

            {errorMessage ? (
                <div className="decision-log__error" role="alert">
                    {errorMessage}
                </div>
            ) : null}

            {isLoading ? (
                <LoadingState label="Loading decisions" lines={4} />
            ) : rows.length === 0 ? (
                <EmptyState
                    size="compact"
                    icon={<ClipboardList size={20} />}
                    title={filtered ? 'No decisions match these filters' : 'No Autopilot decisions yet'}
                    description={
                        filtered
                            ? 'Clear a filter to see more.'
                            : 'Run the Autopilot Planner (above) and every step it takes lands here.'
                    }
                />
            ) : (
                <div className="decision-log__runs">
                    {groupedByRun.map(([runKey, groupRows]) => (
                        <section key={runKey || 'no-run'} className="decision-log__run" data-testid="decision-log-run">
                            <header className="decision-log__run-head">
                                <span className="decision-log__run-id" title={runKey || ''}>
                                    {runKey ? `run ${runKey.slice(0, 8)}…` : 'no run id'}
                                </span>
                                <span>
                                    {groupRows.length} step{groupRows.length === 1 ? '' : 's'}
                                </span>
                            </header>
                            <ol className="decision-log__list">
                                {groupRows.map((row) => {
                                    const expanded = expandedId === row.id;
                                    return (
                                        <li
                                            key={row.id}
                                            className={`decision-log__item${row.blocker ? ' decision-log__item--blocker' : ''}`}
                                        >
                                            <button
                                                type="button"
                                                className="decision-log__row"
                                                onClick={() => setExpandedId((prev) => (prev === row.id ? null : row.id))}
                                                aria-label={`${expanded ? 'Collapse' : 'Expand'} step ${row.sequence}: ${row.stage}`}
                                                data-testid={`decision-log-row-${row.id}`}
                                            >
                                                <span className={`decision-log__status decision-log__status--${row.status || 'info'}`}>
                                                    {String(row.status || 'info').toUpperCase()}
                                                </span>
                                                <span className="decision-log__stage">{row.stage}</span>
                                                <span className="decision-log__summary">{row.summary || row.rationale || ''}</span>
                                            </button>
                                            {expanded ? (
                                                <div className="decision-log__detail" data-testid={`decision-log-detail-${row.id}`}>
                                                    <div>
                                                        <strong>action</strong>: {row.action}
                                                    </div>
                                                    {row.reason_code ? (
                                                        <div>
                                                            <strong>reason_code</strong>: {row.reason_code}
                                                        </div>
                                                    ) : null}
                                                    <div>
                                                        <strong>actor</strong>: {row.actor}
                                                        {' · '}
                                                        <strong>seq</strong>: {row.sequence}
                                                        {' · '}
                                                        <strong>dry_run</strong>: {row.dry_run ? 'yes' : 'no'}
                                                    </div>
                                                    {row.created_at ? (
                                                        <div>
                                                            <strong>ts</strong>: {new Date(row.created_at).toLocaleString()}
                                                        </div>
                                                    ) : null}
                                                    {row.rationale ? (
                                                        <div>
                                                            <strong>rationale</strong>: {row.rationale}
                                                        </div>
                                                    ) : null}
                                                    {Object.keys(row.payload || {}).length > 0 ? (
                                                        <pre className="decision-log__payload">{JSON.stringify(row.payload, null, 2)}</pre>
                                                    ) : null}
                                                </div>
                                            ) : null}
                                        </li>
                                    );
                                })}
                            </ol>
                        </section>
                    ))}
                </div>
            )}
        </section>
    );
}
