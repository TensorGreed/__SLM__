/**
 * Coach rail panel for DataStudio checks with blocked / attention / ready verdicts.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    ExternalLink,
    Gauge,
    ListChecks,
    RefreshCw,
    Route,
    ShieldCheck,
} from 'lucide-react';

import { getDataStudioCoachRail } from '../../api/dataStudio';
import type {
    DataStudioCoachAction,
    DataStudioCoachCheck,
    DataStudioCoachRail,
} from '../../api/dataStudio';
import './DataStudioCoachRailPanel.css';

interface DataStudioCoachRailPanelProps {
    projectId: number;
    onOpenTarget: (target: string, sectionId?: string | null) => void;
    onCoachLoaded?: (coach: DataStudioCoachRail | null) => void;
}

const COACH_VERDICT_COPY: Record<DataStudioCoachRail['verdict'], { label: string; detail: string }> = {
    blocked: {
        label: 'Blocked',
        detail: 'Start with the highest-priority blocker before preparing or training.',
    },
    attention: {
        label: 'Needs attention',
        detail: 'The main path is visible, with a few checks to resolve or review.',
    },
    ready: {
        label: 'Ready',
        detail: 'The current data path is clear for the next downstream action.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function labelForToken(value: string | undefined | null): string {
    if (!value) return 'Unknown';
    return value.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function statusClass(status: string): string {
    if (status === 'ready') return 'ready';
    if (status === 'blocked') return 'blocked';
    if (status === 'empty') return 'empty';
    return 'attention';
}

function statusIcon(status: string) {
    if (status === 'ready') {
        return <CheckCircle2 size={16} aria-hidden="true" />;
    }
    return <AlertTriangle size={16} aria-hidden="true" />;
}

function ActionCard({
    action,
    onOpenTarget,
}: {
    action: DataStudioCoachAction;
    onOpenTarget: (target: string, sectionId?: string | null) => void;
}) {
    return (
        <article className={`data-studio-coach__action data-studio-coach__action--${action.priority}`}>
            <div>
                <p>Next best action</p>
                <h4>{action.title}</h4>
                <span>{action.section_label}</span>
            </div>
            <p>{action.message}</p>
            <button type="button" className="btn btn-primary" onClick={() => onOpenTarget(action.target_tab, action.section_id)}>
                <ExternalLink size={15} aria-hidden="true" />
                {action.action_label}
            </button>
        </article>
    );
}

function CheckButton({
    check,
    onOpenTarget,
}: {
    check: DataStudioCoachCheck;
    onOpenTarget: (target: string, sectionId?: string | null) => void;
}) {
    const totalOpen = check.blocker_count + check.warning_count;
    return (
        <button
            type="button"
            className={`data-studio-coach__check data-studio-coach__check--${statusClass(check.status)}`}
            onClick={() => onOpenTarget(check.target_tab, check.id)}
        >
            <span>{statusIcon(check.status)}</span>
            <span>
                <strong>{check.label}</strong>
                <small>{check.message}</small>
            </span>
            <b>{totalOpen > 0 ? formatNumber(totalOpen) : labelForToken(check.status)}</b>
        </button>
    );
}

function StepRow({
    step,
    onOpenTarget,
}: {
    step: DataStudioCoachAction;
    onOpenTarget: (target: string, sectionId?: string | null) => void;
}) {
    return (
        <button
            type="button"
            className={`data-studio-coach__step data-studio-coach__step--${step.severity}`}
            onClick={() => onOpenTarget(step.target_tab, step.section_id)}
        >
            <span>{step.severity === 'info' ? <CheckCircle2 size={15} aria-hidden="true" /> : <AlertTriangle size={15} aria-hidden="true" />}</span>
            <span>
                <strong>{step.title}</strong>
                <small>{step.section_label}</small>
            </span>
            <b>{step.action_label}</b>
        </button>
    );
}

export default function DataStudioCoachRailPanel({
    projectId,
    onOpenTarget,
    onCoachLoaded,
}: DataStudioCoachRailPanelProps) {
    const [coach, setCoach] = useState<DataStudioCoachRail | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadCoach = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioCoachRail(projectId);
            setCoach(data);
            onCoachLoaded?.(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Data Studio coach.');
            if (!coach) {
                onCoachLoaded?.(null);
            }
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadCoach();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topSteps = useMemo(
        () => coach?.next_steps.slice(0, 4) ?? [],
        [coach],
    );
    const checks = useMemo(
        () => coach?.checks ?? [],
        [coach],
    );

    if (loading && !coach) {
        return (
            <section className="data-studio-coach data-studio-coach--loading">
                <span>Loading Data Studio coach...</span>
            </section>
        );
    }

    if (error && !coach) {
        return (
            <section className="data-studio-coach data-studio-coach--error">
                <div>
                    <h3>Coach</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadCoach()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!coach) {
        return null;
    }

    const verdict = COACH_VERDICT_COPY[coach.verdict];

    return (
        <section
            className={`data-studio-coach data-studio-coach--${coach.verdict}`}
            data-testid="data-studio-coach-rail"
        >
            <div className="data-studio-coach__header">
                <div>
                    <p className="data-studio-coach__eyebrow">Coach</p>
                    <h3>Data Studio Coach</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-coach__actions">
                    <span className={`data-studio-coach__verdict data-studio-coach__verdict--${coach.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-coach__refresh"
                        onClick={() => void loadCoach()}
                        aria-label="Refresh Data Studio coach"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-coach__metrics" aria-label="Coach metrics">
                <div className="data-studio-coach__metric">
                    <AlertTriangle size={18} aria-hidden="true" />
                    <span>Blockers</span>
                    <strong>{formatNumber(coach.summary.blocker_count)}</strong>
                </div>
                <div className="data-studio-coach__metric">
                    <ListChecks size={18} aria-hidden="true" />
                    <span>Warnings</span>
                    <strong>{formatNumber(coach.summary.warning_count)}</strong>
                </div>
                <div className="data-studio-coach__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Ready checks</span>
                    <strong>{formatNumber(coach.summary.ready_section_count)}</strong>
                </div>
                <div className="data-studio-coach__metric">
                    <Gauge size={18} aria-hidden="true" />
                    <span>Sections</span>
                    <strong>{formatNumber(coach.summary.section_count)}</strong>
                </div>
            </div>

            <div className="data-studio-coach__body">
                <ActionCard action={coach.next_action} onOpenTarget={onOpenTarget} />

                <div className="data-studio-coach__steps">
                    <h4>Next steps</h4>
                    {topSteps.map((step) => (
                        <StepRow step={step} key={step.id} onOpenTarget={onOpenTarget} />
                    ))}
                </div>
            </div>

            <div className="data-studio-coach__checks">
                <div className="data-studio-coach__checks-head">
                    <Route size={18} aria-hidden="true" />
                    <h4>Power checks</h4>
                </div>
                <div className="data-studio-coach__check-grid">
                    {checks.map((check) => (
                        <CheckButton check={check} key={check.id} onOpenTarget={onOpenTarget} />
                    ))}
                </div>
            </div>

            <details className="data-studio-coach__details">
                <summary>Power details</summary>
                <pre>{compactJson(coach)}</pre>
            </details>
        </section>
    );
}
