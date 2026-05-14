/**
 * Annotation workspace (Story 1.2).
 *
 * Renders two views depending on the route:
 * - ``/project/:id/annotate``           → label-job list + creation form.
 * - ``/project/:id/annotate/:job_id``   → keyboard-driven labeler page.
 *
 * The labeler hand-shakes with the Story 1.1 backend:
 *   getLabelJob → fetchNextRow → submitLabel | skipRow → fetchNextRow …
 *
 * Per the Codex brief: loading + error states use the existing toast.
 */

import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';

import {
    type CreateJobBody,
    type LabelJob,
    type LabelJobDetail,
    type LabelRow,
    type LabelType,
    type SubmitLabelBody,
    createLabelJob,
    deleteLabelJob,
    fetchNextRow,
    getLabelJob,
    listLabelJobs,
    skipRow,
    submitLabel,
} from '../api/annotation';
import AnnotationProgress from '../components/annotation/AnnotationProgress';
import ClassificationLabeler from '../components/annotation/ClassificationLabeler';
import PreferencePairLabeler, {
    type PreferencePayload,
} from '../components/annotation/PreferencePairLabeler';
import SpanLabeler, {
    type SpanAnnotation,
} from '../components/annotation/SpanLabeler';
import { toast } from '../stores/toastStore';


// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

function extractError(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (err instanceof Error && err.message) return err.message;
    return fallback;
}

const TEXT_FIELD_CANDIDATES = [
    'text',
    'prompt',
    'content',
    'body',
    'question',
    'instruction',
    'input',
    'source',
];

/** Extract a human-readable text from a label_row's raw_payload. Tries
 * common keys first; falls back to JSON pretty-print so a reviewer
 * always sees something. */
function extractRowText(raw: Record<string, unknown>): string {
    for (const key of TEXT_FIELD_CANDIDATES) {
        const value = raw[key];
        if (typeof value === 'string' && value.trim()) return value;
    }
    try {
        return JSON.stringify(raw, null, 2);
    } catch {
        return String(raw);
    }
}

// ─────────────────────────────────────────────────────────────────────
// List view
// ─────────────────────────────────────────────────────────────────────

function CreateJobForm({
    projectId,
    onCreated,
}: {
    projectId: number;
    onCreated: (job: LabelJob) => void;
}) {
    const [name, setName] = useState('');
    const [labelType, setLabelType] = useState<LabelType>('classification');
    const [labelsCsv, setLabelsCsv] = useState('positive, negative, neutral');
    const [targetRows, setTargetRows] = useState<number | ''>('');
    const [busy, setBusy] = useState(false);

    const submit = async (event: React.FormEvent) => {
        event.preventDefault();
        if (busy) return;
        const cleanName = name.trim();
        if (!cleanName) {
            toast.error('Name is required.');
            return;
        }
        const items = labelsCsv
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean);

        const body: CreateJobBody = {
            name: cleanName,
            label_type: labelType,
            label_schema:
                labelType === 'classification'
                    ? { allowed_labels: items }
                    : labelType === 'span'
                      ? { span_types: items }
                      : {},
            target_rows:
                typeof targetRows === 'number' && targetRows > 0
                    ? targetRows
                    : null,
        };
        setBusy(true);
        try {
            const job = await createLabelJob(projectId, body);
            toast.success(`Created job "${job.name}".`);
            setName('');
            onCreated(job);
        } catch (err) {
            toast.error(extractError(err, 'Failed to create job.'));
        } finally {
            setBusy(false);
        }
    };

    return (
        <form
            onSubmit={submit}
            data-testid="annotate-create-form"
            style={{
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-md)',
                padding: 'var(--space-md)',
                marginBottom: 'var(--space-lg)',
                background: 'var(--bg-secondary)',
            }}
        >
            <h3 style={{ marginTop: 0 }}>New label job</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <span style={{ fontSize: '0.8rem' }}>Name</span>
                    <input
                        className="form-input"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Sentiment v1"
                        data-testid="annotate-create-name"
                    />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <span style={{ fontSize: '0.8rem' }}>Task shape</span>
                    <select
                        className="form-input"
                        value={labelType}
                        onChange={(e) =>
                            setLabelType(e.target.value as LabelType)
                        }
                        data-testid="annotate-create-type"
                    >
                        <option value="classification">classification</option>
                        <option value="span">span (NER)</option>
                        <option value="preference_pair">preference_pair</option>
                    </select>
                </label>
                {labelType !== 'preference_pair' && (
                    <label
                        style={{ display: 'flex', flexDirection: 'column', gap: 4 }}
                    >
                        <span style={{ fontSize: '0.8rem' }}>
                            {labelType === 'classification'
                                ? 'Allowed labels'
                                : 'Span types'}{' '}
                            (comma-separated)
                        </span>
                        <input
                            className="form-input"
                            value={labelsCsv}
                            onChange={(e) => setLabelsCsv(e.target.value)}
                            data-testid="annotate-create-labels"
                        />
                    </label>
                )}
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <span style={{ fontSize: '0.8rem' }}>
                        Target rows (optional)
                    </span>
                    <input
                        className="form-input"
                        type="number"
                        min={1}
                        value={targetRows}
                        onChange={(e) =>
                            setTargetRows(
                                e.target.value === ''
                                    ? ''
                                    : Math.max(0, parseInt(e.target.value, 10)),
                            )
                        }
                        data-testid="annotate-create-target"
                    />
                </label>
                <div>
                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={busy}
                        data-testid="annotate-create-submit"
                    >
                        {busy ? 'Creating…' : 'Create job'}
                    </button>
                </div>
            </div>
        </form>
    );
}

function AnnotateListView({ projectId }: { projectId: number }) {
    const [jobs, setJobs] = useState<LabelJob[] | null>(null);
    const [error, setError] = useState<string>('');

    const refresh = useCallback(async () => {
        try {
            const result = await listLabelJobs(projectId);
            setJobs(result);
            setError('');
        } catch (err) {
            setError(extractError(err, 'Failed to load label jobs.'));
        }
    }, [projectId]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const handleDelete = async (job: LabelJob) => {
        if (
            !window.confirm(
                `Delete label job "${job.name}"? This also drops all its rows.`,
            )
        ) {
            return;
        }
        try {
            await deleteLabelJob(projectId, job.id);
            toast.success(`Deleted "${job.name}".`);
            await refresh();
        } catch (err) {
            toast.error(extractError(err, 'Failed to delete job.'));
        }
    };

    return (
        <div className="page-container">
            <h2>Annotation</h2>
            <p style={{ color: 'var(--text-secondary)', marginTop: 0 }}>
                Define label jobs and label rows by hand. Each job tracks
                its own task shape, label set, and reviewer queue.
            </p>

            <CreateJobForm projectId={projectId} onCreated={() => void refresh()} />

            {error && (
                <div className="error-banner" data-testid="annotate-list-error">
                    {error}
                </div>
            )}

            {jobs === null && !error && <div>Loading…</div>}

            {jobs && jobs.length === 0 && !error && (
                <div
                    style={{
                        color: 'var(--text-secondary)',
                        fontStyle: 'italic',
                    }}
                    data-testid="annotate-list-empty"
                >
                    No label jobs yet. Create one above to start labeling.
                </div>
            )}

            {jobs && jobs.length > 0 && (
                <table className="table" data-testid="annotate-list-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Target</th>
                            <th>Updated</th>
                            <th />
                        </tr>
                    </thead>
                    <tbody>
                        {jobs.map((job) => (
                            <tr key={job.id} data-testid={`annotate-list-row-${job.id}`}>
                                <td>
                                    <Link
                                        to={`/project/${projectId}/annotate/${job.id}`}
                                    >
                                        {job.name}
                                    </Link>
                                </td>
                                <td>{job.label_type}</td>
                                <td>{job.status}</td>
                                <td>{job.target_rows ?? '—'}</td>
                                <td>
                                    {job.updated_at
                                        ? new Date(job.updated_at).toLocaleString()
                                        : '—'}
                                </td>
                                <td>
                                    <button
                                        type="button"
                                        className="btn btn-ghost btn-sm"
                                        onClick={() => void handleDelete(job)}
                                        data-testid={`annotate-list-delete-${job.id}`}
                                    >
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────
// Labeler view
// ─────────────────────────────────────────────────────────────────────

function AnnotateLabelerView({
    projectId,
    jobId,
}: {
    projectId: number;
    jobId: number;
}) {
    const [job, setJob] = useState<LabelJobDetail | null>(null);
    const [currentRow, setCurrentRow] = useState<LabelRow | null>(null);
    const [queueEmpty, setQueueEmpty] = useState(false);
    const [error, setError] = useState<string>('');
    const [busy, setBusy] = useState(false);

    const reviewerId = (() => {
        const raw = localStorage.getItem('slm_user_id');
        if (!raw) return null;
        const n = parseInt(raw, 10);
        return Number.isFinite(n) ? n : null;
    })();

    const refreshStats = useCallback(async () => {
        try {
            const detail = await getLabelJob(projectId, jobId);
            setJob(detail);
        } catch (err) {
            setError(extractError(err, 'Failed to load job.'));
        }
    }, [projectId, jobId]);

    const loadNext = useCallback(async () => {
        setBusy(true);
        try {
            const next = await fetchNextRow(projectId, jobId, reviewerId);
            setCurrentRow(next.row);
            setQueueEmpty(next.queue_empty);
        } catch (err) {
            setError(extractError(err, 'Failed to fetch next row.'));
        } finally {
            setBusy(false);
        }
    }, [projectId, jobId, reviewerId]);

    useEffect(() => {
        void refreshStats();
        void loadNext();
    }, [refreshStats, loadNext]);

    const handleSubmit = useCallback(
        async (payload: SubmitLabelBody) => {
            if (!currentRow || busy) return;
            setBusy(true);
            try {
                await submitLabel(projectId, jobId, currentRow.id, payload);
                toast.success('Label saved.');
                await refreshStats();
                await loadNext();
            } catch (err) {
                toast.error(extractError(err, 'Failed to submit label.'));
            } finally {
                setBusy(false);
            }
        },
        [projectId, jobId, currentRow, busy, refreshStats, loadNext],
    );

    const handleSkip = useCallback(async () => {
        if (!currentRow || busy) return;
        setBusy(true);
        try {
            await skipRow(projectId, jobId, currentRow.id);
            toast.info('Row skipped.');
            await refreshStats();
            await loadNext();
        } catch (err) {
            toast.error(extractError(err, 'Failed to skip row.'));
        } finally {
            setBusy(false);
        }
    }, [projectId, jobId, currentRow, busy, refreshStats, loadNext]);

    if (error) {
        return (
            <div className="page-container">
                <div className="error-banner" data-testid="annotate-error">
                    {error}
                </div>
                <Link to={`/project/${projectId}/annotate`}>← Back to jobs</Link>
            </div>
        );
    }

    if (!job) {
        return (
            <div className="page-container">
                <div className="skeleton" style={{ height: 60 }} />
            </div>
        );
    }

    const text = currentRow ? extractRowText(currentRow.raw_payload) : '';

    return (
        <div className="page-container">
            <div style={{ marginBottom: 'var(--space-md)' }}>
                <Link to={`/project/${projectId}/annotate`}>← Back to jobs</Link>
            </div>

            <AnnotationProgress jobName={job.name} stats={job.stats} />

            {job.instructions && (
                <div
                    style={{
                        marginTop: 'var(--space-md)',
                        padding: 'var(--space-md)',
                        background: 'var(--bg-secondary)',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-color)',
                    }}
                    data-testid="annotate-instructions"
                >
                    {job.instructions}
                </div>
            )}

            <div style={{ marginTop: 'var(--space-lg)' }}>
                {queueEmpty || !currentRow ? (
                    <div
                        style={{
                            padding: 'var(--space-lg)',
                            background: 'var(--bg-secondary)',
                            borderRadius: 'var(--radius-md)',
                            textAlign: 'center',
                        }}
                        data-testid="annotate-queue-empty"
                    >
                        <strong>Queue empty.</strong>
                        <div
                            style={{
                                color: 'var(--text-secondary)',
                                marginTop: 8,
                            }}
                        >
                            No unlabeled rows are waiting. Seed more rows from
                            a dataset to keep labeling.
                        </div>
                    </div>
                ) : job.label_type === 'classification' ? (
                    <ClassificationLabeler
                        text={text}
                        labels={
                            (job.label_schema.allowed_labels as string[]) || []
                        }
                        disabled={busy}
                        onSubmit={(label) =>
                            void handleSubmit({ label_payload: { label } })
                        }
                        onSkip={() => void handleSkip()}
                    />
                ) : job.label_type === 'span' ? (
                    <SpanLabeler
                        key={currentRow.id}
                        text={text}
                        spanTypes={
                            (job.label_schema.span_types as string[]) || []
                        }
                        disabled={busy}
                        onSubmit={(spans: SpanAnnotation[]) =>
                            void handleSubmit({
                                label_payload: { spans },
                            })
                        }
                        onSkip={() => void handleSkip()}
                    />
                ) : job.label_type === 'preference_pair' ? (
                    <PreferencePairLabeler
                        key={currentRow.id}
                        prompt={String(
                            currentRow.raw_payload.prompt ?? '',
                        )}
                        completionA={String(
                            currentRow.raw_payload.completion_a ?? '',
                        )}
                        completionB={String(
                            currentRow.raw_payload.completion_b ?? '',
                        )}
                        disabled={busy}
                        onSubmit={(payload: PreferencePayload) =>
                            void handleSubmit({
                                label_payload:
                                    payload as unknown as Record<
                                        string,
                                        unknown
                                    >,
                            })
                        }
                        onSkip={() => void handleSkip()}
                    />
                ) : (
                    <div data-testid="annotate-unsupported">
                        Label type <code>{job.label_type}</code> is not yet
                        supported in this UI.
                    </div>
                )}
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────

export default function ProjectAnnotatePage() {
    const { id, job_id } = useParams<{ id: string; job_id?: string }>();
    const navigate = useNavigate();

    const projectId = Number.parseInt(id ?? '', 10);
    const jobId = job_id ? Number.parseInt(job_id, 10) : null;

    useEffect(() => {
        if (!Number.isFinite(projectId)) {
            navigate('/', { replace: true });
        }
    }, [projectId, navigate]);

    if (!Number.isFinite(projectId)) {
        return null;
    }

    if (jobId !== null && Number.isFinite(jobId)) {
        return <AnnotateLabelerView projectId={projectId} jobId={jobId} />;
    }
    return <AnnotateListView projectId={projectId} />;
}
