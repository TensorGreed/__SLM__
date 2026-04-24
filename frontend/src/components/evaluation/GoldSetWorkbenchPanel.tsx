import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type KeyboardEvent as ReactKeyboardEvent,
} from 'react';

import api from '../../api/client';
import './GoldSetWorkbenchPanel.css';

interface DatasetEntry {
    id: number;
    name: string;
    dataset_type: string;
    record_count: number;
    file_path: string | null;
    is_locked?: boolean;
}

interface GoldSetRow {
    id: number;
    gold_set_id: number;
    version_id: number;
    source_row_key: string;
    source_dataset_id: number;
    input: Record<string, unknown>;
    expected: Record<string, unknown>;
    rationale: string;
    labels: Record<string, unknown>;
    reviewer_id: number | null;
    status: string;
    created_at: string | null;
    updated_at: string | null;
    reviewed_at: string | null;
}

interface SampleResponse {
    gold_set_id: number;
    version_id: number;
    version: number;
    requested: number;
    created: number;
    skipped_duplicates: number;
    strategy: string;
    stratify_by: string | null;
    rows: GoldSetRow[];
}

interface QueueRowPreview {
    status: string;
    labels: Record<string, unknown>;
    input_snippet: string;
    expected_snippet: string;
}

interface QueueEntry {
    queue_id: number;
    row_id: number;
    reviewer_id: number | null;
    priority: number;
    status: string;
    assigned_at: string | null;
    completed_at: string | null;
    row_preview: QueueRowPreview;
}

interface QueueResponse {
    gold_set_id: number;
    count: number;
    limit: number;
    offset: number;
    items: QueueEntry[];
}

interface PackGate {
    gate_id: string;
    metric_id: string;
    operator: string;
    threshold: number;
    required: boolean;
}

interface PackTaskSpec {
    task_profile?: string;
    display_name?: string;
    description?: string;
    gates?: PackGate[];
    required_metric_ids?: string[];
}

interface GeneratedPack {
    pack_id: string;
    display_name?: string;
    description?: string;
    default_task_profile?: string;
    task_specs?: PackTaskSpec[];
    provenance?: {
        blueprint_id?: number | null;
        dataset_id?: number | null;
        adapter_id?: number | null;
        generated_at?: string | null;
    };
}

interface GoldSetWorkbenchPanelProps {
    projectId: number;
}

const SAMPLING_STRATEGIES: Array<{ value: 'random' | 'stratified'; label: string }> = [
    { value: 'random', label: 'Random (seeded)' },
    { value: 'stratified', label: 'Stratified by field' },
];

const LABELLER_STATUSES: Array<{ value: string; label: string; shortcut: string }> = [
    { value: 'approved', label: 'Approve', shortcut: 'A' },
    { value: 'rejected', label: 'Reject', shortcut: 'R' },
    { value: 'changes_requested', label: 'Changes', shortcut: 'C' },
    { value: 'in_review', label: 'In Review', shortcut: 'I' },
    { value: 'pending', label: 'Pending', shortcut: 'P' },
];

const QUEUE_STATUS_FILTERS: Array<{ value: string; label: string }> = [
    { value: '', label: 'All statuses' },
    { value: 'pending', label: 'Pending' },
    { value: 'in_progress', label: 'In progress' },
    { value: 'completed', label: 'Completed' },
    { value: 'skipped', label: 'Skipped' },
];

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

function prettyJson(value: unknown): string {
    if (value === null || value === undefined) {
        return '';
    }
    if (typeof value === 'string') {
        return value;
    }
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

function parseJsonOrNull(raw: string): Record<string, unknown> | null | 'invalid' {
    const trimmed = raw.trim();
    if (!trimmed) {
        return null;
    }
    try {
        const parsed = JSON.parse(trimmed);
        if (parsed === null) {
            return null;
        }
        if (typeof parsed !== 'object' || Array.isArray(parsed)) {
            return 'invalid';
        }
        return parsed as Record<string, unknown>;
    } catch {
        return 'invalid';
    }
}

export default function GoldSetWorkbenchPanel({ projectId }: GoldSetWorkbenchPanelProps) {
    // Datasets + gold-set selection
    const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
    const [loadingDatasets, setLoadingDatasets] = useState(false);
    const [datasetsError, setDatasetsError] = useState<string | null>(null);
    const [goldSetId, setGoldSetId] = useState<number | null>(null);

    // Pack preview
    const [pack, setPack] = useState<GeneratedPack | null>(null);
    const [isGeneratingPack, setIsGeneratingPack] = useState(false);
    const [packError, setPackError] = useState<string | null>(null);

    // Builder
    const [sourceDatasetId, setSourceDatasetId] = useState<number | null>(null);
    const [targetCount, setTargetCount] = useState(20);
    const [strategy, setStrategy] = useState<'random' | 'stratified'>('random');
    const [stratifyBy, setStratifyBy] = useState('');
    const [sampleSeed, setSampleSeed] = useState<string>('42');
    const [sampleReviewerId, setSampleReviewerId] = useState<string>('');
    const [lastSample, setLastSample] = useState<SampleResponse | null>(null);
    const [isSampling, setIsSampling] = useState(false);
    const [builderError, setBuilderError] = useState<string | null>(null);

    // Labeller
    const [rows, setRows] = useState<GoldSetRow[]>([]);
    const [activeRowIndex, setActiveRowIndex] = useState(0);
    const [draftExpected, setDraftExpected] = useState('');
    const [draftRationale, setDraftRationale] = useState('');
    const [draftLabels, setDraftLabels] = useState('');
    const [draftReviewerId, setDraftReviewerId] = useState('');
    const [labellerError, setLabellerError] = useState<string | null>(null);
    const [isSavingRow, setIsSavingRow] = useState(false);
    const [lastSavedHint, setLastSavedHint] = useState<string | null>(null);
    const labellerRef = useRef<HTMLDivElement | null>(null);

    // Queue
    const [queue, setQueue] = useState<QueueEntry[]>([]);
    const [queueTotal, setQueueTotal] = useState(0);
    const [queueStatusFilter, setQueueStatusFilter] = useState('');
    const [queueReviewerFilter, setQueueReviewerFilter] = useState('');
    const [queueLimit, setQueueLimit] = useState(50);
    const [isLoadingQueue, setIsLoadingQueue] = useState(false);
    const [queueError, setQueueError] = useState<string | null>(null);

    // -- Datasets -----------------------------------------------------------

    const refreshDatasets = useCallback(async () => {
        setLoadingDatasets(true);
        setDatasetsError(null);
        try {
            const res = await api.get<{ datasets: DatasetEntry[] }>(`/projects/${projectId}/datasets`);
            const list = res.data?.datasets ?? [];
            setDatasets(list);
            const gold = list.find((d) => d.dataset_type === 'gold_dev' || d.dataset_type === 'gold_test');
            setGoldSetId((prev) => prev ?? gold?.id ?? null);
            const firstNonGold = list.find(
                (d) => d.dataset_type !== 'gold_dev' && d.dataset_type !== 'gold_test' && d.record_count > 0,
            );
            setSourceDatasetId((prev) => prev ?? firstNonGold?.id ?? null);
        } catch (err) {
            setDatasetsError(errorDetail(err, 'Failed to load datasets.'));
        } finally {
            setLoadingDatasets(false);
        }
    }, [projectId]);

    useEffect(() => {
        void refreshDatasets();
    }, [refreshDatasets]);

    const goldSets = useMemo(
        () => datasets.filter((d) => d.dataset_type === 'gold_dev' || d.dataset_type === 'gold_test'),
        [datasets],
    );

    const candidateSources = useMemo(
        () => datasets.filter((d) => d.dataset_type !== 'gold_dev' && d.dataset_type !== 'gold_test'),
        [datasets],
    );

    // -- Pack preview -------------------------------------------------------

    const generatePack = useCallback(async () => {
        setIsGeneratingPack(true);
        setPackError(null);
        try {
            const res = await api.post<GeneratedPack>(
                `/projects/${projectId}/evaluation/packs/generate`,
                { include_judge_rubric: true },
            );
            setPack(res.data);
        } catch (err) {
            setPackError(errorDetail(err, 'Could not generate starter pack.'));
        } finally {
            setIsGeneratingPack(false);
        }
    }, [projectId]);

    // -- Builder ------------------------------------------------------------

    const runSample = useCallback(async () => {
        if (goldSetId === null) {
            setBuilderError('Pick a gold set first.');
            return;
        }
        if (sourceDatasetId === null) {
            setBuilderError('Pick a source dataset.');
            return;
        }
        if (strategy === 'stratified' && !stratifyBy.trim()) {
            setBuilderError('Stratified sampling needs a stratify-by field.');
            return;
        }
        setIsSampling(true);
        setBuilderError(null);
        try {
            const payload: Record<string, unknown> = {
                source_dataset_id: sourceDatasetId,
                target_count: Math.max(1, targetCount),
                strategy,
            };
            if (strategy === 'stratified') {
                payload.stratify_by = stratifyBy.trim();
            }
            const seedNum = sampleSeed.trim() === '' ? null : Number(sampleSeed);
            if (seedNum !== null && Number.isFinite(seedNum)) {
                payload.seed = seedNum;
            }
            const reviewerNum = sampleReviewerId.trim() === '' ? null : Number(sampleReviewerId);
            if (reviewerNum !== null && Number.isFinite(reviewerNum) && reviewerNum > 0) {
                payload.reviewer_id = reviewerNum;
            }
            const res = await api.post<SampleResponse>(
                `/api/gold-sets/${goldSetId}/rows/sample`,
                payload,
            );
            setLastSample(res.data);
            setRows(res.data.rows);
            setActiveRowIndex(0);
        } catch (err) {
            setBuilderError(errorDetail(err, 'Sampling failed.'));
        } finally {
            setIsSampling(false);
        }
    }, [goldSetId, sourceDatasetId, strategy, stratifyBy, targetCount, sampleSeed, sampleReviewerId]);

    // -- Labeller -----------------------------------------------------------

    const activeRow = rows[activeRowIndex] ?? null;

    useEffect(() => {
        if (!activeRow) {
            setDraftExpected('');
            setDraftRationale('');
            setDraftLabels('');
            setDraftReviewerId('');
            setLabellerError(null);
            return;
        }
        setDraftExpected(prettyJson(activeRow.expected));
        setDraftRationale(activeRow.rationale || '');
        setDraftLabels(prettyJson(activeRow.labels));
        setDraftReviewerId(activeRow.reviewer_id ? String(activeRow.reviewer_id) : '');
        setLabellerError(null);
    }, [activeRow]);

    const commitRowPatch = useCallback(
        async (patch: Record<string, unknown>) => {
            if (!activeRow || goldSetId === null) {
                return null;
            }
            setIsSavingRow(true);
            setLabellerError(null);
            try {
                const res = await api.patch<GoldSetRow>(
                    `/api/gold-sets/${goldSetId}/rows/${activeRow.id}`,
                    patch,
                );
                const updated = res.data;
                setRows((prev) => prev.map((row, idx) => (idx === activeRowIndex ? updated : row)));
                setLastSavedHint(`Saved #${updated.id} → ${updated.status}`);
                return updated;
            } catch (err) {
                setLabellerError(errorDetail(err, 'Save failed.'));
                return null;
            } finally {
                setIsSavingRow(false);
            }
        },
        [activeRow, activeRowIndex, goldSetId],
    );

    const buildDraftPatch = useCallback((): Record<string, unknown> | 'invalid' => {
        const expectedParsed = parseJsonOrNull(draftExpected);
        if (expectedParsed === 'invalid') {
            setLabellerError('`expected` must be a JSON object (or empty).');
            return 'invalid';
        }
        const labelsParsed = parseJsonOrNull(draftLabels);
        if (labelsParsed === 'invalid') {
            setLabellerError('`labels` must be a JSON object (or empty).');
            return 'invalid';
        }
        const reviewerId = draftReviewerId.trim() === '' ? null : Number(draftReviewerId);
        if (reviewerId !== null && (!Number.isFinite(reviewerId) || reviewerId <= 0)) {
            setLabellerError('Reviewer id must be a positive integer (or blank).');
            return 'invalid';
        }
        return {
            expected: expectedParsed ?? {},
            rationale: draftRationale,
            labels: labelsParsed ?? {},
            reviewer_id: reviewerId,
        };
    }, [draftExpected, draftLabels, draftRationale, draftReviewerId]);

    const saveDraft = useCallback(async () => {
        const patch = buildDraftPatch();
        if (patch === 'invalid') {
            return;
        }
        await commitRowPatch(patch);
    }, [buildDraftPatch, commitRowPatch]);

    const setStatus = useCallback(
        async (status: string) => {
            const basePatch = buildDraftPatch();
            if (basePatch === 'invalid') {
                return;
            }
            await commitRowPatch({ ...basePatch, status });
        },
        [buildDraftPatch, commitRowPatch],
    );

    const goNextRow = useCallback(() => {
        setActiveRowIndex((idx) => Math.min(rows.length - 1, idx + 1));
    }, [rows.length]);

    const goPrevRow = useCallback(() => {
        setActiveRowIndex((idx) => Math.max(0, idx - 1));
    }, []);

    const handleLabellerKeyDown = useCallback(
        (event: ReactKeyboardEvent<HTMLDivElement>) => {
            const target = event.target as HTMLElement | null;
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) {
                return;
            }
            if (event.metaKey || event.ctrlKey || event.altKey) {
                return;
            }
            const key = event.key.toLowerCase();
            if (key === 'a') {
                event.preventDefault();
                void setStatus('approved');
            } else if (key === 'r') {
                event.preventDefault();
                void setStatus('rejected');
            } else if (key === 'c') {
                event.preventDefault();
                void setStatus('changes_requested');
            } else if (key === 'i') {
                event.preventDefault();
                void setStatus('in_review');
            } else if (key === 'p') {
                event.preventDefault();
                void setStatus('pending');
            } else if (key === 'arrowright' || key === 'n') {
                event.preventDefault();
                goNextRow();
            } else if (key === 'arrowleft' || key === 'b') {
                event.preventDefault();
                goPrevRow();
            } else if (key === 's') {
                event.preventDefault();
                void saveDraft();
            }
        },
        [setStatus, goNextRow, goPrevRow, saveDraft],
    );

    // -- Queue --------------------------------------------------------------

    const refreshQueue = useCallback(async () => {
        if (goldSetId === null) {
            setQueue([]);
            setQueueTotal(0);
            return;
        }
        setIsLoadingQueue(true);
        setQueueError(null);
        try {
            const params: Record<string, string | number> = { limit: queueLimit, offset: 0 };
            if (queueStatusFilter) {
                params.status = queueStatusFilter;
            }
            const reviewerNum = queueReviewerFilter.trim() === '' ? null : Number(queueReviewerFilter);
            if (reviewerNum !== null && Number.isFinite(reviewerNum) && reviewerNum > 0) {
                params.reviewer_id = reviewerNum;
            }
            const res = await api.get<QueueResponse>(`/api/gold-sets/${goldSetId}/queue`, { params });
            setQueue(res.data.items ?? []);
            setQueueTotal(res.data.count ?? 0);
        } catch (err) {
            setQueueError(errorDetail(err, 'Failed to load reviewer queue.'));
            setQueue([]);
            setQueueTotal(0);
        } finally {
            setIsLoadingQueue(false);
        }
    }, [goldSetId, queueLimit, queueReviewerFilter, queueStatusFilter]);

    useEffect(() => {
        void refreshQueue();
    }, [refreshQueue]);

    // -- Render -------------------------------------------------------------

    const selectedGoldSet = goldSets.find((d) => d.id === goldSetId) ?? null;

    return (
        <div className="gold-workbench">
            <section className="card gold-workbench-section">
                <div className="gold-workbench-section-head">
                    <div>
                        <h3>Generated pack preview</h3>
                        <p className="gold-workbench-subtitle">
                            Auto-generate a starter evaluation pack from the project blueprint + active dataset.
                        </p>
                    </div>
                    <button
                        className="btn btn-primary"
                        onClick={() => void generatePack()}
                        disabled={isGeneratingPack}
                    >
                        {isGeneratingPack ? 'Generating…' : pack ? 'Regenerate' : 'Generate pack'}
                    </button>
                </div>
                {packError && <div className="gold-workbench-error">{packError}</div>}
                {pack ? (
                    <div className="gold-workbench-pack-preview">
                        <div className="gold-workbench-pack-meta">
                            <span><strong>{pack.display_name || pack.pack_id}</strong></span>
                            <span className="gold-workbench-pack-id">{pack.pack_id}</span>
                            {pack.default_task_profile && (
                                <span className="badge badge-neutral">{pack.default_task_profile}</span>
                            )}
                        </div>
                        {pack.description && <p className="gold-workbench-pack-desc">{pack.description}</p>}
                        {(pack.task_specs ?? []).map((task) => (
                            <div key={task.task_profile ?? 'default'} className="gold-workbench-task-spec">
                                <div className="gold-workbench-task-head">
                                    <strong>{task.display_name || task.task_profile}</strong>
                                    {task.required_metric_ids && task.required_metric_ids.length > 0 && (
                                        <span className="gold-workbench-task-metrics">
                                            metrics: {task.required_metric_ids.join(', ')}
                                        </span>
                                    )}
                                </div>
                                {task.gates && task.gates.length > 0 ? (
                                    <table className="gold-workbench-gate-table">
                                        <thead>
                                            <tr>
                                                <th>Gate</th>
                                                <th>Metric</th>
                                                <th>Target</th>
                                                <th>Required</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {task.gates.map((gate) => (
                                                <tr key={gate.gate_id}>
                                                    <td>{gate.gate_id}</td>
                                                    <td>{gate.metric_id}</td>
                                                    <td>{gate.operator} {gate.threshold}</td>
                                                    <td>{gate.required ? 'yes' : 'no'}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                ) : (
                                    <p className="gold-workbench-pack-note">No gates defined.</p>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="gold-workbench-pack-note">
                        No preview yet. Click <em>Generate pack</em> to derive starter gates from your project blueprint.
                    </p>
                )}
            </section>

            <section className="card gold-workbench-section">
                <div className="gold-workbench-section-head">
                    <div>
                        <h3>Gold-set builder</h3>
                        <p className="gold-workbench-subtitle">
                            Sample rows from a source dataset into a draft version of your gold set. Dedup is keyed on row content.
                        </p>
                    </div>
                    <button
                        className="btn btn-ghost"
                        onClick={() => void refreshDatasets()}
                        disabled={loadingDatasets}
                    >
                        {loadingDatasets ? 'Refreshing…' : 'Refresh datasets'}
                    </button>
                </div>
                {datasetsError && <div className="gold-workbench-error">{datasetsError}</div>}
                <div className="gold-workbench-grid">
                    <div className="form-group">
                        <label>Gold set</label>
                        <select
                            className="input"
                            value={goldSetId ?? ''}
                            onChange={(e) => setGoldSetId(e.target.value === '' ? null : Number(e.target.value))}
                        >
                            <option value="">— pick a gold set —</option>
                            {goldSets.map((ds) => (
                                <option key={ds.id} value={ds.id}>
                                    #{ds.id} · {ds.name} ({ds.dataset_type}, {ds.record_count} rows)
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Source dataset</label>
                        <select
                            className="input"
                            value={sourceDatasetId ?? ''}
                            onChange={(e) => setSourceDatasetId(e.target.value === '' ? null : Number(e.target.value))}
                        >
                            <option value="">— pick a source —</option>
                            {candidateSources.map((ds) => (
                                <option key={ds.id} value={ds.id}>
                                    #{ds.id} · {ds.name} ({ds.dataset_type}, {ds.record_count} rows)
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Target count</label>
                        <input
                            className="input"
                            type="number"
                            min={1}
                            value={targetCount}
                            onChange={(e) => setTargetCount(Number(e.target.value) || 1)}
                        />
                    </div>
                    <div className="form-group">
                        <label>Strategy</label>
                        <select
                            className="input"
                            value={strategy}
                            onChange={(e) => setStrategy(e.target.value as 'random' | 'stratified')}
                        >
                            {SAMPLING_STRATEGIES.map((s) => (
                                <option key={s.value} value={s.value}>{s.label}</option>
                            ))}
                        </select>
                    </div>
                    {strategy === 'stratified' && (
                        <div className="form-group">
                            <label>Stratify by (JSON field name)</label>
                            <input
                                className="input"
                                value={stratifyBy}
                                placeholder="e.g. difficulty"
                                onChange={(e) => setStratifyBy(e.target.value)}
                            />
                        </div>
                    )}
                    <div className="form-group">
                        <label>Seed (optional)</label>
                        <input
                            className="input"
                            value={sampleSeed}
                            onChange={(e) => setSampleSeed(e.target.value)}
                            placeholder="e.g. 42"
                        />
                    </div>
                    <div className="form-group">
                        <label>Auto-assign reviewer ID (optional)</label>
                        <input
                            className="input"
                            value={sampleReviewerId}
                            onChange={(e) => setSampleReviewerId(e.target.value)}
                            placeholder="blank = no auto-assignment"
                        />
                    </div>
                </div>
                {builderError && <div className="gold-workbench-error">{builderError}</div>}
                <div className="gold-workbench-actions">
                    <button
                        className="btn btn-primary"
                        onClick={() => void runSample()}
                        disabled={isSampling || goldSetId === null || sourceDatasetId === null}
                    >
                        {isSampling ? 'Sampling…' : 'Sample rows into gold set'}
                    </button>
                    {lastSample && (
                        <span className="gold-workbench-actions-note">
                            v{lastSample.version}: requested {lastSample.requested}, created {lastSample.created},
                            skipped {lastSample.skipped_duplicates} duplicate(s).
                        </span>
                    )}
                </div>
                {selectedGoldSet && (
                    <p className="gold-workbench-hint">
                        Working on <strong>#{selectedGoldSet.id} · {selectedGoldSet.name}</strong>
                        {' — '}
                        <span>a draft version is created on first sample and new rows append to the current draft.</span>
                    </p>
                )}
            </section>

            <section
                className="card gold-workbench-section"
                tabIndex={0}
                onKeyDown={handleLabellerKeyDown}
                ref={labellerRef}
                aria-label="Single-row labeller with keyboard shortcuts"
            >
                <div className="gold-workbench-section-head">
                    <div>
                        <h3>Single-row labeller</h3>
                        <p className="gold-workbench-subtitle">
                            Shortcuts: <kbd>A</kbd> approve, <kbd>R</kbd> reject, <kbd>C</kbd> changes,
                            {' '}<kbd>I</kbd> in-review, <kbd>P</kbd> pending,
                            {' '}<kbd>S</kbd> save, <kbd>←</kbd>/<kbd>B</kbd> prev, <kbd>→</kbd>/<kbd>N</kbd> next.
                        </p>
                    </div>
                    <div className="gold-workbench-row-nav">
                        <button
                            className="btn btn-ghost"
                            onClick={goPrevRow}
                            disabled={activeRowIndex <= 0 || rows.length === 0}
                        >
                            ← Prev
                        </button>
                        <span className="gold-workbench-row-pos">
                            {rows.length === 0 ? '0 / 0' : `${activeRowIndex + 1} / ${rows.length}`}
                        </span>
                        <button
                            className="btn btn-ghost"
                            onClick={goNextRow}
                            disabled={activeRowIndex >= rows.length - 1 || rows.length === 0}
                        >
                            Next →
                        </button>
                    </div>
                </div>
                {activeRow ? (
                    <div className="gold-workbench-labeller">
                        <div className="gold-workbench-labeller-meta">
                            <span className={`badge gold-workbench-status-${activeRow.status}`}>{activeRow.status}</span>
                            <span>row #{activeRow.id}</span>
                            <span>version #{activeRow.version_id}</span>
                            <span>source #{activeRow.source_dataset_id}</span>
                            {activeRow.reviewer_id !== null && <span>reviewer #{activeRow.reviewer_id}</span>}
                        </div>
                        <div className="gold-workbench-labeller-body">
                            <div className="form-group">
                                <label>Input (read-only)</label>
                                <textarea
                                    className="input"
                                    readOnly
                                    rows={8}
                                    value={prettyJson(activeRow.input)}
                                />
                            </div>
                            <div className="form-group">
                                <label>Expected (JSON object)</label>
                                <textarea
                                    className="input"
                                    rows={8}
                                    value={draftExpected}
                                    onChange={(e) => setDraftExpected(e.target.value)}
                                    placeholder={'{\n  "answer": "…"\n}'}
                                />
                            </div>
                            <div className="form-group">
                                <label>Rationale</label>
                                <textarea
                                    className="input"
                                    rows={4}
                                    value={draftRationale}
                                    onChange={(e) => setDraftRationale(e.target.value)}
                                    placeholder="Why this answer? (free text)"
                                />
                            </div>
                            <div className="form-group">
                                <label>Labels (JSON object)</label>
                                <textarea
                                    className="input"
                                    rows={4}
                                    value={draftLabels}
                                    onChange={(e) => setDraftLabels(e.target.value)}
                                    placeholder={'{\n  "difficulty": "medium"\n}'}
                                />
                            </div>
                            <div className="form-group">
                                <label>Reviewer ID</label>
                                <input
                                    className="input"
                                    value={draftReviewerId}
                                    onChange={(e) => setDraftReviewerId(e.target.value)}
                                    placeholder="blank = unassigned"
                                />
                            </div>
                        </div>
                        {labellerError && <div className="gold-workbench-error">{labellerError}</div>}
                        <div className="gold-workbench-actions">
                            <button
                                className="btn btn-secondary"
                                onClick={() => void saveDraft()}
                                disabled={isSavingRow}
                            >
                                {isSavingRow ? 'Saving…' : 'Save (S)'}
                            </button>
                            {LABELLER_STATUSES.map((s) => (
                                <button
                                    key={s.value}
                                    className="btn btn-ghost"
                                    onClick={() => void setStatus(s.value)}
                                    disabled={isSavingRow}
                                    title={`Shortcut: ${s.shortcut}`}
                                >
                                    {s.label} ({s.shortcut})
                                </button>
                            ))}
                            {lastSavedHint && !labellerError && (
                                <span className="gold-workbench-actions-note">{lastSavedHint}</span>
                            )}
                        </div>
                    </div>
                ) : (
                    <p className="gold-workbench-pack-note">
                        Sample some rows above, then use this panel to label them.
                    </p>
                )}
            </section>

            <section className="card gold-workbench-section">
                <div className="gold-workbench-section-head">
                    <div>
                        <h3>Reviewer queue</h3>
                        <p className="gold-workbench-subtitle">
                            Queue entries stay in lockstep with row-level assignment and review status.
                        </p>
                    </div>
                    <button
                        className="btn btn-ghost"
                        onClick={() => void refreshQueue()}
                        disabled={goldSetId === null || isLoadingQueue}
                    >
                        {isLoadingQueue ? 'Loading…' : 'Refresh'}
                    </button>
                </div>
                <div className="gold-workbench-grid">
                    <div className="form-group">
                        <label>Status</label>
                        <select
                            className="input"
                            value={queueStatusFilter}
                            onChange={(e) => setQueueStatusFilter(e.target.value)}
                        >
                            {QUEUE_STATUS_FILTERS.map((opt) => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Reviewer ID</label>
                        <input
                            className="input"
                            value={queueReviewerFilter}
                            onChange={(e) => setQueueReviewerFilter(e.target.value)}
                            placeholder="blank = all reviewers"
                        />
                    </div>
                    <div className="form-group">
                        <label>Limit</label>
                        <input
                            className="input"
                            type="number"
                            min={1}
                            max={500}
                            value={queueLimit}
                            onChange={(e) => setQueueLimit(Number(e.target.value) || 50)}
                        />
                    </div>
                </div>
                {queueError && <div className="gold-workbench-error">{queueError}</div>}
                {queue.length === 0 ? (
                    <p className="gold-workbench-pack-note">
                        {goldSetId === null
                            ? 'Pick a gold set to see its reviewer queue.'
                            : 'No queue entries. Auto-assign a reviewer when sampling, or set a reviewer on individual rows.'}
                    </p>
                ) : (
                    <div className="table-container">
                        <table className="gold-workbench-queue-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Row</th>
                                    <th>Reviewer</th>
                                    <th>Status</th>
                                    <th>Priority</th>
                                    <th>Input snippet</th>
                                </tr>
                            </thead>
                            <tbody>
                                {queue.map((entry) => (
                                    <tr key={entry.queue_id}>
                                        <td>{entry.queue_id}</td>
                                        <td>#{entry.row_id}</td>
                                        <td>{entry.reviewer_id ?? '—'}</td>
                                        <td>
                                            <span className={`badge gold-workbench-status-${entry.status}`}>
                                                {entry.status}
                                            </span>
                                        </td>
                                        <td>{entry.priority}</td>
                                        <td className="gold-workbench-snippet">{entry.row_preview.input_snippet}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
                <p className="gold-workbench-hint">
                    Showing {queue.length} of {queueTotal} total queue entries.
                </p>
            </section>
        </div>
    );
}
