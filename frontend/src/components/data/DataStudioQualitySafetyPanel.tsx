import { useEffect, useId, useMemo, useState } from 'react';
import {
    AlertTriangle,
    ChevronDown,
    ChevronUp,
    CheckCircle2,
    Eye,
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

function triageForCheck(check: DataStudioQualitySafetyCheck): 'fix-now' | 'review-soon' | 'looks-good' {
    if (check.status === 'blocked' || check.severity === 'blocker') {
        return 'fix-now';
    }
    if (check.status === 'attention' || check.severity === 'warning') {
        return 'review-soon';
    }
    return 'looks-good';
}

function checkIcon(check: DataStudioQualitySafetyCheck) {
    if (check.status === 'ready') {
        return <CheckCircle2 size={16} aria-hidden="true" />;
    }
    return <AlertTriangle size={16} aria-hidden="true" />;
}

function whyBeforeSft(check: DataStudioQualitySafetyCheck): string {
    const fingerprint = `${check.id} ${check.category} ${check.label}`.toLowerCase();

    if (check.status === 'ready') {
        if (check.domain_authored) {
            return 'This domain rule is currently satisfied, so it does not add risk to the next SFT handoff.';
        }
        return 'This built-in check is clear in the scanned sample, so it does not block SFT readiness right now.';
    }
    if (fingerprint.includes('pii') || fingerprint.includes('pci') || fingerprint.includes('privacy') || fingerprint.includes('redaction')) {
        return 'Sensitive values can be memorized by a fine-tuned model, so they should be removed, masked, or reviewed before training.';
    }
    if (fingerprint.includes('leakage') || fingerprint.includes('split')) {
        return 'Leakage across train, validation, and test splits makes evaluation look stronger than the model really is.';
    }
    if (fingerprint.includes('duplicate')) {
        return 'Repeated rows can overweight a few examples and make the SLM less reliable on real user inputs.';
    }
    if (fingerprint.includes('required') || fingerprint.includes('field') || fingerprint.includes('mapping') || fingerprint.includes('coverage')) {
        return 'Missing contract fields can create malformed instruction pairs that teach the adapter the wrong shape.';
    }
    if (fingerprint.includes('synthetic') || fingerprint.includes('review') || fingerprint.includes('gate')) {
        return 'Unreviewed rows should stay out of SFT until a person or trusted workflow confirms they are trainable.';
    }
    if (fingerprint.includes('citation') || fingerprint.includes('context')) {
        return 'Grounding gaps can train answers that sound confident without the context your domain expects.';
    }
    if (fingerprint.includes('forbidden') || fingerprint.includes('policy') || fingerprint.includes('domain-authored')) {
        return 'Domain-authored rules capture local risks and style requirements that generic checks may miss.';
    }
    if (fingerprint.includes('quality') || fingerprint.includes('empty') || fingerprint.includes('low')) {
        return 'Low-quality examples add noise to SFT and can make the model copy weak formatting or incomplete answers.';
    }
    return 'Fixing this before SFT improves trust in the prepared dataset and reduces avoidable training noise.';
}

function DrilldownPreview({
    check,
    onOpenTarget,
}: {
    check: DataStudioQualitySafetyCheck;
    onOpenTarget: (target: string) => void;
}) {
    const drilldown = check.drilldown;
    if (!drilldown) {
        return null;
    }

    return (
        <div className="data-studio-quality__drilldown">
            <div className="data-studio-quality__drilldown-head">
                <div>
                    <strong>Affected-row preview</strong>
                    <span>
                        {drilldown.read_only ? 'Read-only' : 'Editable'}
                        {' · '}
                        {drilldown.redacted ? 'Redacted sample context' : 'Sample context'}
                    </span>
                </div>
                <b>{formatNumber(drilldown.total_affected)} affected</b>
            </div>

            <p className="data-studio-quality__drilldown-action">
                <strong>Destination action</strong>
                <span>{drilldown.action.description}</span>
            </p>

            {drilldown.source_counts.length > 0 ? (
                <div className="data-studio-quality__source-counts" aria-label={`${check.label} source counts`}>
                    {drilldown.source_counts.map((source) => (
                        <button
                            type="button"
                            key={`${check.id}:${source.source}:${source.target_tab}`}
                            onClick={() => onOpenTarget(source.target_tab)}
                        >
                            <span>{source.source}</span>
                            <b>{formatNumber(source.count)}</b>
                        </button>
                    ))}
                </div>
            ) : null}

            {drilldown.rows.length > 0 ? (
                <div className="data-studio-quality__preview-rows">
                    {drilldown.rows.map((row) => (
                        <article
                            className="data-studio-quality__preview-row"
                            key={`${check.id}:${row.source}:${row.source_type}:${row.row_index}`}
                        >
                            <small>
                                {row.source}
                                {' · '}
                                {labelForToken(row.source_type)}
                                {' · row '}
                                {formatNumber(row.row_index + 1)}
                                {row.file_name ? ` · ${row.file_name}` : ''}
                            </small>
                            <p>{row.redacted_text || drilldown.empty_message}</p>
                            {row.fields.length > 0 ? (
                                <dl>
                                    {row.fields.map((field) => (
                                        <div key={`${row.source}:${row.row_index}:${field.field}`}>
                                            <dt>{field.field}</dt>
                                            <dd>{field.value}</dd>
                                        </div>
                                    ))}
                                </dl>
                            ) : null}
                        </article>
                    ))}
                </div>
            ) : (
                <p className="data-studio-quality__empty">{drilldown.empty_message}</p>
            )}

            <button
                type="button"
                className="btn btn-secondary"
                onClick={() => onOpenTarget(drilldown.action.target_tab)}
            >
                <ExternalLink size={15} aria-hidden="true" />
                {drilldown.action.label}
            </button>
        </div>
    );
}

function CheckCard({
    check,
    onOpenTarget,
}: {
    check: DataStudioQualitySafetyCheck;
    onOpenTarget: (target: string) => void;
}) {
    const ruleType = check.domain_authored ? 'Domain-authored' : 'Built-in deterministic';
    const [previewOpen, setPreviewOpen] = useState(false);
    const previewId = useId();

    return (
        <article
            className={[
                'data-studio-quality__check',
                `data-studio-quality__check--${statusClass(check.status)}`,
                check.domain_authored ? 'data-studio-quality__check--domain-authored' : 'data-studio-quality__check--built-in',
            ].join(' ')}
        >
            <div className="data-studio-quality__check-head">
                <span>{checkIcon(check)}</span>
                <div>
                    <strong>{check.label}</strong>
                    <div className="data-studio-quality__check-meta">
                        <small>
                            {check.workflow_owner}
                            {' · '}
                            {labelForToken(check.category)}
                        </small>
                        <em>{ruleType}</em>
                        {check.read_only_preview ? <em>Preview only</em> : null}
                    </div>
                </div>
                <b>{formatNumber(check.count)}</b>
            </div>
            <p>{check.message}</p>
            <p className="data-studio-quality__why">
                <strong>Why before SFT</strong>
                <span>{whyBeforeSft(check)}</span>
            </p>
            {check.evidence.length > 0 ? (
                <ul>
                    {check.evidence.slice(0, 3).map((item) => (
                        <li key={item}>{item}</li>
                    ))}
                </ul>
            ) : null}
            <div className="data-studio-quality__check-actions">
                {check.drilldown ? (
                    <button
                        type="button"
                        className="btn btn-ghost"
                        aria-expanded={previewOpen}
                        aria-controls={previewId}
                        onClick={() => setPreviewOpen((current) => !current)}
                    >
                        <Eye size={15} aria-hidden="true" />
                        {previewOpen ? 'Hide preview' : 'Preview rows'}
                        {previewOpen ? <ChevronUp size={15} aria-hidden="true" /> : <ChevronDown size={15} aria-hidden="true" />}
                    </button>
                ) : null}
                <button type="button" className="btn btn-secondary" onClick={() => onOpenTarget(check.target_tab)}>
                    <ExternalLink size={15} aria-hidden="true" />
                    {check.action_label}
                </button>
            </div>
            {previewOpen && check.drilldown ? (
                <div id={previewId}>
                    <DrilldownPreview check={check} onOpenTarget={onOpenTarget} />
                </div>
            ) : null}
        </article>
    );
}

function TriageLane({
    id,
    title,
    description,
    emptyMessage,
    checks,
    onOpenTarget,
}: {
    id: 'fix-now' | 'review-soon' | 'looks-good';
    title: string;
    description: string;
    emptyMessage: string;
    checks: DataStudioQualitySafetyCheck[];
    onOpenTarget: (target: string) => void;
}) {
    const Icon = id === 'looks-good' ? CheckCircle2 : id === 'fix-now' ? ShieldAlert : AlertTriangle;

    return (
        <section className={`data-studio-quality__triage-lane data-studio-quality__triage-lane--${id}`}>
            <div className="data-studio-quality__triage-head">
                <span>
                    <Icon size={16} aria-hidden="true" />
                </span>
                <div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                <b>{formatNumber(checks.length)}</b>
            </div>
            {checks.length > 0 ? (
                <div className="data-studio-quality__check-list">
                    {checks.slice(0, 4).map((check) => (
                        <CheckCard check={check} key={check.id} onOpenTarget={onOpenTarget} />
                    ))}
                </div>
            ) : (
                <p className="data-studio-quality__empty">{emptyMessage}</p>
            )}
        </section>
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
    const triageLanes = useMemo(() => {
        const checks = quality?.checks ?? [];
        return [
            {
                id: 'fix-now' as const,
                title: 'Fix now',
                description: 'Blockers that can leak private data, contaminate splits, or break trainable examples.',
                emptyMessage: 'No blockers found in the scanned sample.',
                checks: checks.filter((check) => triageForCheck(check) === 'fix-now'),
            },
            {
                id: 'review-soon' as const,
                title: 'Review soon',
                description: 'Warnings that may still train, but deserve review before the next prepare run.',
                emptyMessage: 'No warning-level findings need review right now.',
                checks: checks.filter((check) => triageForCheck(check) === 'review-soon'),
            },
            {
                id: 'looks-good' as const,
                title: 'Looks good',
                description: 'Checks that are clear for the scanned sample and can support SFT readiness.',
                emptyMessage: 'No passing checks have been reported yet.',
                checks: checks.filter((check) => triageForCheck(check) === 'looks-good'),
            },
        ];
    }, [quality]);

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

            <div className="data-studio-quality__triage" aria-label="Quality and safety triage">
                {triageLanes.map((lane) => (
                    <TriageLane
                        key={lane.id}
                        id={lane.id}
                        title={lane.title}
                        description={lane.description}
                        emptyMessage={lane.emptyMessage}
                        checks={lane.checks}
                        onOpenTarget={onOpenTarget}
                    />
                ))}
            </div>

            <div className="data-studio-quality__body">
                <div className="data-studio-quality__checks">
                    <h4>Built-in deterministic checks</h4>
                    <p className="data-studio-quality__section-note">
                        Built-in checks run first and catch common SFT risks before any domain-specific rules are layered on.
                    </p>
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
                                <p>
                                    These checks are visually tagged as Domain-authored and remain read-only previews until a destination workflow confirms changes.
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
