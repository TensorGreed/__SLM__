import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    ExternalLink,
    RefreshCw,
    ScanSearch,
    ShieldAlert,
    Sparkles,
} from 'lucide-react';

import { getDataStudioQualitySafety } from '../../api/dataStudio';
import type {
    DataStudioQualitySafety,
    DataStudioQualitySafetyCheck,
    DataStudioQualitySafetyGroup,
} from '../../api/dataStudio';
import './DataStudioQualitySafetyPanel.css';

interface DataStudioQualitySafetyPanelProps {
    projectId: number;
    onOpenTarget: (target: string) => void;
}

const QUALITY_VERDICT_COPY: Record<DataStudioQualitySafety['verdict'], { label: string; detail: string }> = {
    blocked: {
        label: 'Blocked',
        detail: 'Fix high-risk safety or leakage blockers before preparing another training dataset.',
    },
    attention: {
        label: 'Needs review',
        detail: 'Deterministic scans found warnings to inspect before the next dataset prep run.',
    },
    ready: {
        label: 'Clear',
        detail: 'Deterministic quality, safety, review, and leakage checks are clear for the scanned sample.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function labelForToken(value: string | undefined | null): string {
    if (!value) return 'Unknown';
    return value.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function statusClass(status: string): string {
    if (status === 'blocked') return 'blocked';
    if (status === 'attention') return 'attention';
    return 'ready';
}

function checkIcon(check: DataStudioQualitySafetyCheck) {
    if (check.status === 'ready') {
        return <CheckCircle2 size={16} aria-hidden="true" />;
    }
    return <AlertTriangle size={16} aria-hidden="true" />;
}

function CheckCard({
    check,
    onOpenTarget,
}: {
    check: DataStudioQualitySafetyCheck;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <article className={`data-studio-quality__check data-studio-quality__check--${statusClass(check.status)}`}>
            <div className="data-studio-quality__check-head">
                <span>{checkIcon(check)}</span>
                <div>
                    <strong>{check.label}</strong>
                    <small>
                        {check.workflow_owner}
                        {' · '}
                        {labelForToken(check.category)}
                    </small>
                </div>
                <b>{formatNumber(check.count)}</b>
            </div>
            <p>{check.message}</p>
            {check.evidence.length > 0 ? (
                <ul>
                    {check.evidence.slice(0, 3).map((item) => (
                        <li key={item}>{item}</li>
                    ))}
                </ul>
            ) : null}
            <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget(check.target_tab)}>
                <ExternalLink size={15} aria-hidden="true" />
                {check.action_label}
            </button>
        </article>
    );
}

function GroupList({
    title,
    groups,
    onOpenTarget,
}: {
    title: string;
    groups: DataStudioQualitySafetyGroup[];
    onOpenTarget: (target: string) => void;
}) {
    return (
        <div className="data-studio-quality__group">
            <h4>{title}</h4>
            {groups.length > 0 ? (
                <div className="data-studio-quality__group-list">
                    {groups.slice(0, 6).map((group) => (
                        <button
                            type="button"
                            className="data-studio-quality__group-row"
                            key={group.key}
                            onClick={() => onOpenTarget(group.target_tab)}
                        >
                            <span>
                                <strong>{group.label}</strong>
                                <small>
                                    {formatNumber(group.blocker_count)}
                                    {' blockers · '}
                                    {formatNumber(group.warning_count)}
                                    {' warnings'}
                                </small>
                            </span>
                            <b>{formatNumber(group.total)}</b>
                        </button>
                    ))}
                </div>
            ) : (
                <p className="data-studio-quality__empty">No grouped findings yet.</p>
            )}
        </div>
    );
}

export default function DataStudioQualitySafetyPanel({
    projectId,
    onOpenTarget,
}: DataStudioQualitySafetyPanelProps) {
    const [quality, setQuality] = useState<DataStudioQualitySafety | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadQuality = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioQualitySafety(projectId);
            setQuality(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load quality and safety scans.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadQuality();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topChecks = useMemo(
        () => {
            const checks = (quality?.checks ?? []).filter((check) => !check.domain_authored);
            const nonReady = checks.filter((check) => check.status !== 'ready');
            return (nonReady.length > 0 ? nonReady : checks).slice(0, 6);
        },
        [quality],
    );
    const domainAuthoredChecks = useMemo(
        () => (quality?.checks ?? []).filter((check) => check.domain_authored).slice(0, 6),
        [quality],
    );

    if (loading && !quality) {
        return (
            <section className="data-studio-quality data-studio-quality--loading">
                <span>Loading quality and safety scans...</span>
            </section>
        );
    }

    if (error && !quality) {
        return (
            <section className="data-studio-quality data-studio-quality--error">
                <div>
                    <h3>Quality & Safety Scan Center</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadQuality()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!quality) {
        return null;
    }

    const verdict = QUALITY_VERDICT_COPY[quality.verdict];

    return (
        <section
            className={`data-studio-quality data-studio-quality--${quality.verdict}`}
            data-testid="data-studio-quality-safety"
        >
            <div className="data-studio-quality__header">
                <div>
                    <p className="data-studio-quality__eyebrow">Quality & Safety</p>
                    <h3>Quality & Safety Scan Center</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-quality__actions">
                    <span className={`data-studio-quality__verdict data-studio-quality__verdict--${quality.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-quality__refresh"
                        onClick={() => void loadQuality()}
                        aria-label="Refresh quality and safety scans"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-quality__metrics" aria-label="Quality and safety metrics">
                <div className="data-studio-quality__metric">
                    <ScanSearch size={18} aria-hidden="true" />
                    <span>Rows scanned</span>
                    <strong>{formatNumber(quality.summary.scanned_rows)}</strong>
                </div>
                <div className="data-studio-quality__metric">
                    <ShieldAlert size={18} aria-hidden="true" />
                    <span>PII/PCI signals</span>
                    <strong>{formatNumber(quality.summary.pii_pci_signal_count)}</strong>
                </div>
                <div className="data-studio-quality__metric">
                    <AlertTriangle size={18} aria-hidden="true" />
                    <span>Blockers</span>
                    <strong>{formatNumber(quality.summary.blocker_count)}</strong>
                </div>
                <div className="data-studio-quality__metric">
                    <CheckCircle2 size={18} aria-hidden="true" />
                    <span>Ready checks</span>
                    <strong>{formatNumber(quality.summary.info_count)}</strong>
                </div>
            </div>

            <div className="data-studio-quality__signals">
                <span>{quality.domain.label}</span>
                <span>{formatPercent(quality.domain.confidence)} domain confidence</span>
                <span>{formatNumber(quality.summary.duplicate_signal_count)} duplicate signals</span>
                <span>{formatNumber(quality.summary.leakage_overlap_count)} leakage overlaps</span>
                <span>{formatNumber(quality.summary.domain_authored_check_count)} domain checks</span>
                <span>{quality.read_only ? 'Read-only scan' : 'Can mutate'}</span>
            </div>

            <div className="data-studio-quality__entrypoints">
                {quality.entry_points.slice(0, 5).map((entry) => (
                    <button
                        type="button"
                        className="btn btn-secondary"
                        key={`${entry.label}:${entry.target_tab}`}
                        onClick={() => onOpenTarget(entry.target_tab)}
                    >
                        <ExternalLink size={15} aria-hidden="true" />
                        {entry.label}
                    </button>
                ))}
                {quality.assist.available ? (
                    <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => onOpenTarget(quality.assist.target_tab)}
                    >
                        <Sparkles size={15} aria-hidden="true" />
                        Explain with Ollama
                    </button>
                ) : null}
            </div>

            <div className="data-studio-quality__body">
                <div className="data-studio-quality__checks">
                    <h4>Deterministic scan results</h4>
                    {topChecks.length > 0 ? (
                        <div className="data-studio-quality__check-list">
                            {topChecks.map((check) => (
                                <CheckCard check={check} key={check.id} onOpenTarget={onOpenTarget} />
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-quality__empty">No scan checks have been produced yet.</p>
                    )}
                </div>

                <div className="data-studio-quality__groups">
                    {quality.domain_authored?.available ? (
                        <div className="data-studio-quality__domain-authored">
                            <div>
                                <h4>Domain-authored previews</h4>
                                <p>
                                    {formatNumber(quality.domain_authored.check_count)}
                                    {' checks from '}
                                    {quality.domain_authored.applied_profile_id || quality.domain_authored.applied_pack_id || 'applied domain setup'}
                                </p>
                            </div>
                            {domainAuthoredChecks.length > 0 ? (
                                <div className="data-studio-quality__check-list">
                                    {domainAuthoredChecks.map((check) => (
                                        <CheckCard check={check} key={check.id} onOpenTarget={onOpenTarget} />
                                    ))}
                                </div>
                            ) : (
                                <p className="data-studio-quality__empty">
                                    Applied domain setup has no previewable quality checks yet.
                                </p>
                            )}
                        </div>
                    ) : (
                        <div className="data-studio-quality__domain-authored data-studio-quality__domain-authored--empty">
                            <h4>Domain-authored previews</h4>
                            <p>Apply a specific Domain Profile or Pack to preview domain-owned checks here.</p>
                            <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget('domain')}>
                                <ExternalLink size={15} aria-hidden="true" />
                                Open Domain Managers
                            </button>
                        </div>
                    )}
                    <GroupList
                        title="By workflow owner"
                        groups={quality.findings_by_owner}
                        onOpenTarget={onOpenTarget}
                    />
                    <GroupList
                        title="By source"
                        groups={quality.findings_by_source}
                        onOpenTarget={onOpenTarget}
                    />
                </div>
            </div>

            <details className="data-studio-quality__details">
                <summary>Power details</summary>
                <pre>{compactJson(quality)}</pre>
            </details>
        </section>
    );
}
