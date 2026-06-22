/**
 * DataHealthReportPanel — D1+D2 of the data-quality arc.
 *
 * Single aggregated view of every data-quality signal the platform
 * computes across ingestion / cleaning / shape vs recipe / class
 * balance. Each row:
 *
 *   - **Traffic-light severity badge** — ok / warn / block, matching
 *     the Coach Mode + trainability-forecast palette so the same red
 *     means the same thing across panels.
 *   - **Plain-English summary** — the actual problem, in words a
 *     non-technical user can act on. Sits above the technical
 *     headline (which is still rendered for users who want the
 *     numbers).
 *   - **"Why this matters" expander** — consequence at training
 *     time if not fixed. Closed by default; one-click expand. The
 *     point is to teach, not to wall users with text.
 *   - **Action chip** (D1 informational only — D3/D4 wire auto-fix).
 *     Names the next step; clicking navigates to the relevant tab
 *     when ``target`` is set.
 *
 * Grouped by phase (Ingestion → Cleaning → Shape → Balance) so the
 * panel reads as a left-to-right pipeline. Empty groups (e.g. no
 * classification recipe → empty Balance) are skipped.
 *
 * Top-level overall badge ("All green" / "Warnings to address" /
 * "Blockers — won't train cleanly") collapses the panel into a
 * one-liner readers can scan from the tab strip.
 *
 * Backed by GET /api/projects/{id}/data-health.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import api from '../../api/client';
import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import ErrorPanel from '../shared/ErrorPanel';
import AutofixPreviewModal from './AutofixPreviewModal';
import './DataHealthReportPanel.css';

type Severity = 'ok' | 'warn' | 'block';

interface SuggestedAction {
    kind?: string;
    label?: string;
    target?: string;
}

// A single leaked-row example carried in a signal's context (the
// leakage signals populate this; the renderer is generic on its
// presence). Every field is optional + defensively read — a malformed
// payload must degrade, not throw (matches the gap-panel guard).
interface LeakageExample {
    source?: string;
    split?: string;
    match_kind?: string;
    jaccard?: number;
    excerpt?: string;
    matched_excerpt?: string;
}

function readExamples(context: Record<string, unknown>): LeakageExample[] {
    const raw = (context as { examples?: unknown }).examples;
    if (!Array.isArray(raw)) return [];
    return raw.filter((e): e is LeakageExample => !!e && typeof e === 'object');
}

interface HealthSignal {
    id: string;
    severity: Severity;
    headline: string;
    plain_english: string;
    why_it_matters: string;
    suggested_action: SuggestedAction | null;
    context: Record<string, unknown>;
    // D3 — when set, the safe auto-fix engine can resolve this
    // signal in one click. The panel renders an "Auto-fix" button
    // that POSTs to /data-health/autofix with this kind as the
    // payload, then refreshes the report.
    autofix_kind?: string | null;
}

interface AutofixResult {
    fix_kind: string;
    applied_count: number;
    summary: string;
    details: Record<string, unknown>;
}

interface HealthGroup {
    id: string;
    title: string;
    subtitle: string;
    signals: HealthSignal[];
}

interface HealthReport {
    project_id: number;
    computed_at: string;
    overall: Severity;
    severity_summary: { ok: number; warn: number; block: number };
    total_signals: number;
    groups: HealthGroup[];
}

interface DataHealthReportPanelProps {
    projectId: number;
    // Optional in-page action interceptor. When a signal's action is
    // clicked, the host gets first refusal: return ``true`` to handle it
    // locally (e.g. scroll to a form on the SAME page the panel is
    // mounted on, or trigger a re-prepare) and suppress the navigate.
    // Returning ``false``/omitting falls back to the normal target→URL
    // navigation. This fixes the dead "Re-split …" leakage buttons —
    // their target is ``dataprep``, but the panel renders INSIDE the
    // dataprep tab, so a navigate is a no-op. See DatasetPrepPanel.
    onSignalAction?: (signalId: string, target?: string) => boolean;
}

// Severity → display metadata. Single source of truth for badge
// label + ARIA description so the same vocab lands across every
// signal row + the top-line summary.
const SEVERITY_META: Record<Severity, { label: string; icon: string }> = {
    ok: { label: 'OK', icon: '✓' },
    warn: { label: 'Warning', icon: '!' },
    block: { label: 'Blocker', icon: '✕' },
};

const OVERALL_HEADLINE: Record<Severity, string> = {
    ok: 'All clear — no data-health issues detected.',
    warn: 'Warnings to address before training.',
    block: 'Blockers — training won\'t produce reliable results until these are fixed.',
};

// Map navigate target → URL. D1 only uses navigate; D3/D4 will add
// the auto_fix kind. Keep this list tight + explicit so a new target
// is one obvious place to touch.
function targetUrl(projectId: number, target: string | undefined): string | null {
    if (!target) return null;
    switch (target) {
        case 'data':
            return `/project/${projectId}/pipeline/data`;
        case 'cleaning':
            return `/project/${projectId}/pipeline/cleaning`;
        case 'goldset':
            return `/project/${projectId}/pipeline/goldset`;
        case 'dataprep':
            return `/project/${projectId}/pipeline/dataprep`;
        case 'synthetic':
            return `/project/${projectId}/pipeline/synthetic`;
        case 'recipe-picker':
            return `/project/${projectId}/recipe-picker`;
        case 'ingest-error-list':
            return `/project/${projectId}/pipeline/data#errors`;
        default:
            return null;
    }
}

// Layman label per autofix kind — what the user sees on the button.
// Keep these terse and active-voice ("Drop", "Dedupe", "Redact") so
// the button reads as the verb the click does.
const AUTOFIX_LABEL: Record<string, string> = {
    drop_failed_docs: 'Drop failed docs',
    dedupe_duplicate_docs: 'Dedupe duplicates',
    redact_pii: 'Redact PII',
    canonicalise_labels: 'Merge label variants',
    // Phase 4 — five new "rewrite cleaned text" fixes.
    near_duplicate_dedup: 'Dedupe near-duplicates',
    normalize_whitespace: 'Normalize whitespace',
    strip_html: 'Strip HTML',
    length_cap: 'Cap length',
    normalize_schema: 'Normalize gold schema',
};

export default function DataHealthReportPanel({ projectId, onSignalAction }: DataHealthReportPanelProps) {
    const navigate = useNavigate();
    const [data, setData] = useState<HealthReport | null>(null);
    const [loading, setLoading] = useState(false);
    // Diagnostics Intervention B — load-failure rendered via shared
    // <ErrorPanel> with troubleshooting_id + remediation copy.
    const [error, setError] = useState<ErrorEnvelope | null>(null);
    const [expanded, setExpanded] = useState<Set<string>>(new Set());
    const [lastFix, setLastFix] = useState<AutofixResult | null>(null);
    // D3.2 — preview-then-apply: the panel opens AutofixPreviewModal
    // here; the modal handles preview + apply and surfaces the
    // result via onApplied. No more window.confirm() — every
    // destructive transform shows its per-item diff first.
    const [previewing, setPreviewing] = useState<{ kind: string; label: string } | null>(null);

    const fetch = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await api.get<HealthReport>(`/projects/${projectId}/data-health`);
            setData(res.data);
        } catch (err) {
            setError(parseErrorEnvelope(err));
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetch();
    }, [fetch]);

    const toggleExpanded = (id: string) => {
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const openPreview = useCallback((kind: string) => {
        const label = AUTOFIX_LABEL[kind] || kind;
        setLastFix(null);
        setPreviewing({ kind, label });
    }, []);

    const handleApplied = useCallback(async (result: AutofixResult) => {
        setLastFix(result);
        // Refresh the report so the just-fixed signal's severity
        // updates (typically ok now).
        await fetch();
    }, [fetch]);

    if (loading && !data) {
        return (
            <div className="data-health data-health--loading" data-testid="data-health">
                Loading Data Health Report…
            </div>
        );
    }

    if (error) {
        return (
            <div className="data-health data-health--error" data-testid="data-health">
                <ErrorPanel
                    envelope={error}
                    onDismiss={() => setError(null)}
                    testIdPrefix="data-health-load-error"
                    actions={
                        <button
                            type="button"
                            className="btn btn-link"
                            onClick={() => void fetch()}
                            data-testid="data-health-retry"
                        >
                            Retry
                        </button>
                    }
                />
            </div>
        );
    }

    if (!data) return null;

    const populatedGroups = data.groups.filter((g) => g.signals.length > 0);

    return (
        <section className="data-health" data-testid="data-health" data-overall={data.overall}>
            <header className={`data-health__head data-health__head--${data.overall}`}>
                <div className="data-health__head-line">
                    <span
                        className={`data-health__overall-badge data-health__overall-badge--${data.overall}`}
                        data-testid="data-health-overall-badge"
                    >
                        {SEVERITY_META[data.overall].icon} {SEVERITY_META[data.overall].label}
                    </span>
                    <h3 className="data-health__title">Data Health Report</h3>
                </div>
                <p className="data-health__headline">{OVERALL_HEADLINE[data.overall]}</p>
                <p className="data-health__counts">
                    {data.severity_summary.block > 0 && (
                        <>
                            <strong>{data.severity_summary.block}</strong> blocker{data.severity_summary.block === 1 ? '' : 's'}
                        </>
                    )}
                    {data.severity_summary.block > 0 && data.severity_summary.warn > 0 && ' · '}
                    {data.severity_summary.warn > 0 && (
                        <>
                            <strong>{data.severity_summary.warn}</strong> warning{data.severity_summary.warn === 1 ? '' : 's'}
                        </>
                    )}
                    {(data.severity_summary.block > 0 || data.severity_summary.warn > 0) && data.severity_summary.ok > 0 && ' · '}
                    {data.severity_summary.ok > 0 && (
                        <>
                            <strong>{data.severity_summary.ok}</strong> ok
                        </>
                    )}
                </p>
            </header>

            {lastFix && (
                <div
                    className="data-health__fix-toast"
                    data-testid="data-health-fix-toast"
                    role="status"
                >
                    <strong>{lastFix.summary}</strong>
                    <button
                        type="button"
                        className="data-health__fix-toast-dismiss"
                        onClick={() => setLastFix(null)}
                        aria-label="Dismiss fix summary"
                    >
                        ×
                    </button>
                </div>
            )}

            {populatedGroups.length === 0 ? (
                <p className="data-health__empty">No data-quality signals yet. Upload some documents to get started.</p>
            ) : (
                <div className="data-health__groups">
                    {populatedGroups.map((group) => (
                        <div
                            key={group.id}
                            className="data-health__group"
                            data-testid={`data-health-group-${group.id}`}
                        >
                            <header className="data-health__group-head">
                                <h4 className="data-health__group-title">{group.title}</h4>
                                <span className="data-health__group-subtitle">{group.subtitle}</span>
                            </header>
                            <ul className="data-health__signal-list">
                                {group.signals.map((sig) => {
                                    const isExpanded = expanded.has(sig.id);
                                    const url = targetUrl(projectId, sig.suggested_action?.target);
                                    return (
                                        <li
                                            key={sig.id}
                                            className={`data-health__signal data-health__signal--${sig.severity}`}
                                            data-testid={`data-health-signal-${sig.id}`}
                                            data-severity={sig.severity}
                                        >
                                            <div className="data-health__signal-head">
                                                <span
                                                    className={`data-health__sev-badge data-health__sev-badge--${sig.severity}`}
                                                    aria-label={SEVERITY_META[sig.severity].label}
                                                >
                                                    {SEVERITY_META[sig.severity].icon}
                                                </span>
                                                <div className="data-health__signal-body">
                                                    {sig.plain_english && (
                                                        <p className="data-health__plain-english">
                                                            {sig.plain_english}
                                                        </p>
                                                    )}
                                                    <p className="data-health__headline-line">
                                                        {sig.headline}
                                                    </p>
                                                    {sig.why_it_matters && (
                                                        <button
                                                            type="button"
                                                            className="data-health__why-toggle"
                                                            onClick={() => toggleExpanded(sig.id)}
                                                            data-testid={`data-health-why-${sig.id}`}
                                                            aria-label={
                                                                isExpanded
                                                                    ? `Hide why this matters for ${sig.id}`
                                                                    : `Show why this matters for ${sig.id}`
                                                            }
                                                        >
                                                            {isExpanded ? '− Why this matters' : '+ Why this matters'}
                                                        </button>
                                                    )}
                                                    {isExpanded && sig.why_it_matters && (
                                                        <p
                                                            className="data-health__why"
                                                            data-testid={`data-health-why-text-${sig.id}`}
                                                        >
                                                            {sig.why_it_matters}
                                                        </p>
                                                    )}
                                                    {(() => {
                                                        // Drill-down: when a signal carries
                                                        // leaked-row examples, let the user expand
                                                        // exactly which rows leaked, from which
                                                        // split, and what they matched.
                                                        const examples = readExamples(sig.context);
                                                        if (examples.length === 0) return null;
                                                        const exKey = `${sig.id}::examples`;
                                                        const exOpen = expanded.has(exKey);
                                                        return (
                                                            <div className="data-health__examples">
                                                                <button
                                                                    type="button"
                                                                    className="data-health__why-toggle"
                                                                    onClick={() => toggleExpanded(exKey)}
                                                                    data-testid={`data-health-examples-toggle-${sig.id}`}
                                                                    aria-label={
                                                                        exOpen
                                                                            ? `Hide leaked rows for ${sig.id}`
                                                                            : `Show ${examples.length} leaked rows for ${sig.id}`
                                                                    }
                                                                >
                                                                    {exOpen
                                                                        ? '− Hide leaked rows'
                                                                        : `+ Show leaked rows (${examples.length})`}
                                                                </button>
                                                                {exOpen && (
                                                                    <table
                                                                        className="data-health__examples-table"
                                                                        data-testid={`data-health-examples-${sig.id}`}
                                                                    >
                                                                        <thead>
                                                                            <tr>
                                                                                <th>Held-out split</th>
                                                                                <th>Match</th>
                                                                                <th>Leaked row</th>
                                                                                <th>Matched source row</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody>
                                                                            {examples.map((ex, i) => (
                                                                                <tr key={i} data-testid={`data-health-example-row-${sig.id}-${i}`}>
                                                                                    <td>
                                                                                        {String(ex.split ?? '?')}
                                                                                    </td>
                                                                                    <td>
                                                                                        {ex.match_kind === 'exact'
                                                                                            ? 'exact'
                                                                                            : `~${typeof ex.jaccard === 'number' ? ex.jaccard.toFixed(2) : '?'}`}
                                                                                    </td>
                                                                                    <td className="data-health__examples-text">
                                                                                        {String(ex.excerpt ?? '')}
                                                                                    </td>
                                                                                    <td className="data-health__examples-text">
                                                                                        {ex.source ? `${ex.source}: ` : ''}
                                                                                        {String(ex.matched_excerpt ?? '')}
                                                                                    </td>
                                                                                </tr>
                                                                            ))}
                                                                        </tbody>
                                                                    </table>
                                                                )}
                                                            </div>
                                                        );
                                                    })()}
                                                </div>
                                                <div className="data-health__actions">
                                                    {sig.autofix_kind && (
                                                        <button
                                                            type="button"
                                                            className="data-health__autofix btn btn-sm"
                                                            onClick={() => openPreview(sig.autofix_kind as string)}
                                                            data-testid={`data-health-autofix-${sig.id}`}
                                                            title="Preview the diff before applying"
                                                        >
                                                            Preview: {AUTOFIX_LABEL[sig.autofix_kind] || sig.autofix_kind}
                                                        </button>
                                                    )}
                                                    {sig.suggested_action?.label && (
                                                        <button
                                                            type="button"
                                                            className="data-health__action btn btn-sm"
                                                            onClick={() => {
                                                                // Host gets first refusal (in-page handling);
                                                                // otherwise fall back to target→URL navigation.
                                                                if (onSignalAction?.(sig.id, sig.suggested_action?.target)) {
                                                                    return;
                                                                }
                                                                if (url) navigate(url);
                                                            }}
                                                            disabled={!url && !onSignalAction}
                                                            data-testid={`data-health-action-${sig.id}`}
                                                            title={
                                                                url || onSignalAction
                                                                    ? `Navigate to ${sig.suggested_action.target}`
                                                                    : 'Action not yet wired up'
                                                            }
                                                        >
                                                            {sig.suggested_action.label}
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        </li>
                                    );
                                })}
                            </ul>
                        </div>
                    ))}
                </div>
            )}

            <footer className="data-health__foot">
                Computed at {new Date(data.computed_at).toLocaleTimeString()}.{' '}
                <button
                    type="button"
                    className="btn btn-link"
                    onClick={() => void fetch()}
                    data-testid="data-health-refresh"
                >
                    Refresh
                </button>
            </footer>

            {previewing && (
                <AutofixPreviewModal
                    projectId={projectId}
                    fixKind={previewing.kind}
                    fixLabel={previewing.label}
                    onClose={() => setPreviewing(null)}
                    onApplied={(r) => void handleApplied(r)}
                />
            )}
        </section>
    );
}
