/**
 * Persistent Autopilot Decision Log drawer (priority.md P6).
 *
 * Mounted at workspace level so it's available on every pipeline/workspace
 * page. Fetches from `/autopilot/decisions` (P1) and supports filtering by
 * stage/status/reason_code. Opens via a floating action button in the
 * bottom-right of the viewport.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ClipboardList, RefreshCw, X } from 'lucide-react';

import api from '../../api/client';
import './DecisionLogDrawer.css';

export interface DecisionLogDrawerProps {
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

export default function DecisionLogDrawer({ projectId }: DecisionLogDrawerProps) {
    const [open, setOpen] = useState(false);
    const [rows, setRows] = useState<DecisionRow[]>([]);
    const [isLoading, setIsLoading] = useState(false);
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
            const params: Record<string, string | number> = {
                project_id: projectId,
                limit: 200,
            };
            if (stage.trim()) params.stage = stage.trim();
            if (status.trim()) params.status = status.trim();
            if (action.trim()) params.action = action.trim();
            if (reasonCode.trim()) params.reason_code = reasonCode.trim();
            if (runId.trim()) params.run_id = runId.trim();
            const res = await api.get<DecisionListResponse>('/autopilot/decisions', { params });
            setRows(res.data.items || []);
        } catch (err) {
            const e = err as { response?: { data?: { detail?: unknown } }; message?: string };
            const detail = e?.response?.data?.detail;
            if (typeof detail === 'string') {
                setErrorMessage(detail);
            } else if (detail && typeof detail === 'object') {
                setErrorMessage(JSON.stringify(detail));
            } else {
                setErrorMessage(e?.message || 'Failed to load decisions.');
            }
        } finally {
            setIsLoading(false);
        }
    }, [projectId, stage, status, action, reasonCode, runId]);

    useEffect(() => {
        if (open) {
            void fetchDecisions();
        }
    }, [open, fetchDecisions]);

    const toggleExpand = (id: number) => {
        setExpandedId((prev) => (prev === id ? null : id));
    };

    const groupedByRun = useMemo(() => {
        const groups = new Map<string, DecisionRow[]>();
        for (const row of rows) {
            const key = row.run_id || '';
            const bucket = groups.get(key) || [];
            bucket.push(row);
            groups.set(key, bucket);
        }
        // Preserve insertion order (which is created_at desc from API).
        return Array.from(groups.entries());
    }, [rows]);

    return (
        <>
            <button
                type="button"
                className="decision-log-fab"
                onClick={() => setOpen((prev) => !prev)}
                aria-label="Toggle Autopilot decision log"
                title="Autopilot decision log"
            >
                <ClipboardList size={18} />
                <span>Decisions</span>
            </button>

            {open ? (
                <aside
                    className="decision-log-drawer"
                    role="complementary"
                    aria-label="Autopilot decision log"
                >
                    <header className="decision-log-header">
                        <div className="decision-log-title">
                            <ClipboardList size={16} />
                            <span>Autopilot decisions</span>
                        </div>
                        <div className="decision-log-actions">
                            <button
                                type="button"
                                className="icon-btn"
                                onClick={() => void fetchDecisions()}
                                disabled={isLoading}
                                aria-label="Refresh decision log"
                                title="Refresh"
                            >
                                <RefreshCw size={14} />
                            </button>
                            <button
                                type="button"
                                className="icon-btn"
                                onClick={() => setOpen(false)}
                                aria-label="Close decision log"
                            >
                                <X size={14} />
                            </button>
                        </div>
                    </header>

                    <div className="decision-log-filters">
                        <label>
                            <span>Stage</span>
                            <input
                                list="decision-log-stage-hints"
                                value={stage}
                                onChange={(e) => setStage(e.target.value)}
                                placeholder="e.g. start_training"
                            />
                            <datalist id="decision-log-stage-hints">
                                {STAGE_HINTS.map((hint) => (
                                    <option key={hint} value={hint} />
                                ))}
                            </datalist>
                        </label>
                        <label>
                            <span>Status</span>
                            <select value={status} onChange={(e) => setStatus(e.target.value)}>
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
                            <input
                                value={runId}
                                onChange={(e) => setRunId(e.target.value)}
                                placeholder="Filter by autopilot run id"
                            />
                        </label>
                    </div>

                    {errorMessage ? (
                        <div className="decision-log-error">{errorMessage}</div>
                    ) : null}

                    <div className="decision-log-body">
                        {isLoading ? (
                            <div className="decision-log-empty">Loading…</div>
                        ) : rows.length === 0 ? (
                            <div className="decision-log-empty">
                                No decisions match the current filters.
                            </div>
                        ) : (
                            groupedByRun.map(([runKey, groupRows]) => (
                                <section key={runKey || 'no-run'} className="decision-log-run">
                                    <header className="decision-log-run-head">
                                        <span className="decision-log-run-id" title={runKey || ''}>
                                            {runKey ? `run ${runKey.slice(0, 8)}…` : 'no run id'}
                                        </span>
                                        <span className="decision-log-run-count">
                                            {groupRows.length} step{groupRows.length === 1 ? '' : 's'}
                                        </span>
                                    </header>
                                    <ol className="decision-log-list">
                                        {groupRows.map((row) => {
                                            const expanded = expandedId === row.id;
                                            const statusClass = `status-${row.status || 'info'}`;
                                            return (
                                                <li
                                                    key={row.id}
                                                    className={`decision-log-item ${
                                                        row.blocker ? 'is-blocker' : ''
                                                    }`}
                                                >
                                                    <button
                                                        type="button"
                                                        className="decision-log-row"
                                                        onClick={() => toggleExpand(row.id)}
                                                        aria-expanded={expanded}
                                                    >
                                                        <span className={`decision-log-status ${statusClass}`}>
                                                            {String(row.status || 'info').toUpperCase()}
                                                        </span>
                                                        <span className="decision-log-stage">{row.stage}</span>
                                                        <span className="decision-log-summary">
                                                            {row.summary || row.rationale || ''}
                                                        </span>
                                                    </button>
                                                    {expanded ? (
                                                        <div className="decision-log-detail">
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
                                                                <strong>dry_run</strong>:{' '}
                                                                {row.dry_run ? 'yes' : 'no'}
                                                            </div>
                                                            {row.created_at ? (
                                                                <div>
                                                                    <strong>ts</strong>:{' '}
                                                                    {new Date(row.created_at).toLocaleString()}
                                                                </div>
                                                            ) : null}
                                                            {row.rationale ? (
                                                                <div>
                                                                    <strong>rationale</strong>: {row.rationale}
                                                                </div>
                                                            ) : null}
                                                            {Object.keys(row.payload || {}).length > 0 ? (
                                                                <pre className="decision-log-payload">
                                                                    {JSON.stringify(row.payload, null, 2)}
                                                                </pre>
                                                            ) : null}
                                                        </div>
                                                    ) : null}
                                                </li>
                                            );
                                        })}
                                    </ol>
                                </section>
                            ))
                        )}
                    </div>
                </aside>
            ) : null}
        </>
    );
}
