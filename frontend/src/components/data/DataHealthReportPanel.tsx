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
import './DataHealthReportPanel.css';

type Severity = 'ok' | 'warn' | 'block';

interface SuggestedAction {
    kind?: string;
    label?: string;
    target?: string;
}

interface HealthSignal {
    id: string;
    severity: Severity;
    headline: string;
    plain_english: string;
    why_it_matters: string;
    suggested_action: SuggestedAction | null;
    context: Record<string, unknown>;
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

export default function DataHealthReportPanel({ projectId }: DataHealthReportPanelProps) {
    const navigate = useNavigate();
    const [data, setData] = useState<HealthReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [expanded, setExpanded] = useState<Set<string>>(new Set());

    const fetch = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const res = await api.get<HealthReport>(`/projects/${projectId}/data-health`);
            setData(res.data);
        } catch (err) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Failed to load Data Health Report.');
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
                {error}{' '}
                <button type="button" className="btn btn-link" onClick={() => void fetch()}>
                    Retry
                </button>
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
                                                </div>
                                                {sig.suggested_action?.label && (
                                                    <button
                                                        type="button"
                                                        className="data-health__action btn btn-sm"
                                                        onClick={() => {
                                                            if (url) navigate(url);
                                                        }}
                                                        disabled={!url}
                                                        data-testid={`data-health-action-${sig.id}`}
                                                        title={
                                                            url
                                                                ? `Navigate to ${sig.suggested_action.target}`
                                                                : 'Action not yet wired up'
                                                        }
                                                    >
                                                        {sig.suggested_action.label}
                                                    </button>
                                                )}
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
        </section>
    );
}
