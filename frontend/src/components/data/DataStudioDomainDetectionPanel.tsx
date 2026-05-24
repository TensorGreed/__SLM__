import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    Compass,
    RefreshCw,
    ShieldCheck,
} from 'lucide-react';

import {
    getDataStudioDomainDetection,
} from '../../api/dataStudio';
import type { DataStudioDomainDetection } from '../../api/dataStudio';
import './DataStudioDomainDetectionPanel.css';

interface DataStudioDomainDetectionPanelProps {
    projectId: number;
}

const DOMAIN_VERDICT_COPY: Record<DataStudioDomainDetection['verdict'], { label: string; detail: string }> = {
    unknown: {
        label: 'Unknown',
        detail: 'Add representative rows or assign a domain profile to confirm domain behavior.',
    },
    attention: {
        label: 'Needs attention',
        detail: 'BrewSLM found domain signals that should be confirmed before training.',
    },
    confirmed: {
        label: 'Confirmed',
        detail: 'Applied domain settings line up with the current project evidence.',
    },
};

const DATASET_TYPE_LABELS: Record<string, string> = {
    raw: 'Raw',
    cleaned: 'Cleaned',
    gold_dev: 'Gold dev',
    gold_test: 'Gold test',
    synthetic: 'Synthetic',
    train: 'Train',
    validation: 'Validation',
    test: 'Test',
};

function formatPercent(value: number | undefined): string {
    const normalized = Number.isFinite(Number(value)) ? Number(value) : 0;
    return `${Math.round(normalized * 100)}%`;
}

function labelForDatasetType(type: string | undefined): string {
    if (!type) return 'No source';
    return DATASET_TYPE_LABELS[type] || type.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function issueIcon(severity: string) {
    if (severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

export default function DataStudioDomainDetectionPanel({
    projectId,
}: DataStudioDomainDetectionPanelProps) {
    const [domain, setDomain] = useState<DataStudioDomainDetection | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadDomain = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioDomainDetection(projectId);
            setDomain(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load domain detection.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadDomain();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topEvidence = useMemo(
        () => domain?.evidence.slice(0, 4) ?? [],
        [domain],
    );
    const topIssues = useMemo(
        () => domain?.issues.slice(0, 3) ?? [],
        [domain],
    );
    const topRisks = useMemo(
        () => domain?.risks.slice(0, 3) ?? [],
        [domain],
    );

    if (loading && !domain) {
        return (
            <section className="data-studio-domain data-studio-domain--loading">
                <span>Loading domain detection...</span>
            </section>
        );
    }

    if (error && !domain) {
        return (
            <section className="data-studio-domain data-studio-domain--error">
                <div>
                    <h3>Domain detection</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadDomain()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!domain) {
        return null;
    }

    const verdict = DOMAIN_VERDICT_COPY[domain.verdict];
    const detected = domain.detected_domain;
    const appliedProfile = domain.applied.profile_display_name || domain.applied.profile_id || 'Generic domain';
    const appliedPack = domain.applied.pack_display_name || domain.applied.pack_id || 'General pack';
    const sourceLabel = domain.source
        ? `${labelForDatasetType(domain.source.dataset_type)}${domain.source.document_name ? ` · ${domain.source.document_name}` : ''}`
        : 'No source sample yet';

    return (
        <section
            className={`data-studio-domain data-studio-domain--${domain.verdict}`}
            data-testid="data-studio-domain"
        >
            <div className="data-studio-domain__header">
                <div>
                    <p className="data-studio-domain__eyebrow">Domain</p>
                    <h3>Domain detection</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-domain__actions">
                    <span className={`data-studio-domain__verdict data-studio-domain__verdict--${domain.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-domain__refresh"
                        onClick={() => void loadDomain()}
                        aria-label="Refresh domain detection"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-domain__metrics" aria-label="Domain detection metrics">
                <div className="data-studio-domain__metric">
                    <Compass size={18} aria-hidden="true" />
                    <span>Detected domain</span>
                    <strong>{detected.label}</strong>
                </div>
                <div className="data-studio-domain__metric">
                    <span>Confidence</span>
                    <strong>{formatPercent(detected.confidence)}</strong>
                </div>
                <div className="data-studio-domain__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Applied profile</span>
                    <strong>{appliedProfile}</strong>
                </div>
                <div className="data-studio-domain__metric">
                    <span>Applied pack</span>
                    <strong>{appliedPack}</strong>
                </div>
            </div>

            <div className="data-studio-domain__source">
                <span>{sourceLabel}</span>
                <small>
                    {detected.source}
                    {domain.source?.sampled_records ? ` · ${domain.source.sampled_records} sampled` : ''}
                </small>
            </div>

            <div className="data-studio-domain__body">
                <div className="data-studio-domain__evidence">
                    <h4>Why BrewSLM thinks this</h4>
                    {topEvidence.length > 0 ? (
                        <div className="data-studio-domain__evidence-list">
                            {topEvidence.map((item) => (
                                <div className="data-studio-domain__evidence-row" key={item.id}>
                                    <strong>{item.title}</strong>
                                    <small>{item.message}</small>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="data-studio-domain__empty">
                            Evidence appears after BrewSLM can inspect representative source rows.
                        </p>
                    )}

                    {topIssues.length > 0 ? (
                        <ul className="data-studio-domain__issues">
                            {topIssues.map((issue) => (
                                <li key={issue.id} className={`data-studio-domain__issue data-studio-domain__issue--${issue.severity}`}>
                                    <span>{issueIcon(issue.severity)}</span>
                                    <div>
                                        <strong>{issue.title}</strong>
                                        <small>{issue.message}</small>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : null}
                </div>

                <div className="data-studio-domain__guidance">
                    <h4>Domain guidance</h4>
                    {domain.suggested_actions.length > 0 ? (
                        <ul className="data-studio-domain__actions-list">
                            {domain.suggested_actions.slice(0, 4).map((action) => (
                                <li key={action.id}>{action.label}</li>
                            ))}
                        </ul>
                    ) : (
                        <p className="data-studio-domain__empty">
                            Domain-specific recommendations will appear after confidence improves.
                        </p>
                    )}

                    {topRisks.length > 0 ? (
                        <div className="data-studio-domain__risks">
                            {topRisks.map((risk) => (
                                <div key={risk.id} className={`data-studio-domain__risk data-studio-domain__risk--${risk.severity}`}>
                                    <strong>{risk.title}</strong>
                                    <small>{risk.message}</small>
                                </div>
                            ))}
                        </div>
                    ) : null}
                </div>
            </div>

            <details className="data-studio-domain__details">
                <summary>Power details</summary>
                <pre>
                    {compactJson({
                        detected_domain: domain.detected_domain,
                        applied: domain.applied,
                        recipe: domain.recipe,
                        source: domain.source,
                        evidence: domain.evidence,
                        power_details: domain.power_details,
                    })}
                </pre>
            </details>
        </section>
    );
}
