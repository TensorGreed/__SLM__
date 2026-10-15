/**
 * External benchmarks (Epic G-B1/B2) — score the selected run on labelled
 * sets someone else made, next to the project's own answer key. Import: a
 * source (Hugging Face dataset or an uploaded file) → preview (detected
 * shape, its labels, the project's) → map every source label onto a project
 * label, a GROUP of project labels (several picked: a prediction of any
 * member counts), or nothing (= ignore) → a locked, eval-only benchmark with
 * leakage-checked rows. Q&A and summarization benchmarks have no labels to
 * map. Scores run as `benchmark_eval` Jobs; each benchmark shows its score
 * history across runs as a small trend.
 */
import { useCallback, useEffect, useState } from 'react';

import {
    benchmarkHeadlineLabel,
    benchmarkLabelText,
    benchmarkScoreText,
    benchmarkSourceText,
    createBenchmark,
    deleteBenchmark,
    evaluateBenchmark,
    getBenchmarkHistory,
    listBenchmarks,
    mappingGroupConflict,
    mappingMembers,
    previewBenchmark,
    type BenchmarkHistoryPoint,
    type BenchmarkInfo,
    type BenchmarkMappingValue,
    type BenchmarkPreview,
} from '../../api/benchmarks';
import { uploadImportFile } from '../../api/datasetImport';
import { toast } from '../../stores/toastStore';
import './ExternalBenchmarksPanel.css';

interface Props {
    projectId: number;
    experimentId: number | null;
}

const IGNORE = 'ignore';

function errorText(err: unknown, fallback: string): string {
    const anyErr = err as { response?: { data?: { message?: string; detail?: string } }; message?: string };
    return anyErr?.response?.data?.message || anyErr?.response?.data?.detail || anyErr?.message || fallback;
}

const SHAPE_TEXT: Record<string, string> = {
    classification: 'classification',
    structured_extraction: 'spans (entity tagging)',
    qa: 'Q&A',
    summarization: 'summarization',
};

const TREND_W = 96;
const TREND_H = 22;

/** Score history of one benchmark: an inline sparkline (each point = one
 *  scored run, hover for run / date / score) + "first → last". Scores of
 *  different headline metrics (e.g. F1 before a judge was available, judged
 *  later) are not one series — the line only joins points of the latest
 *  result's metric. */
export function BenchmarkTrend({ points, slug }: { points: BenchmarkHistoryPoint[]; slug: string }) {
    const scored = points.filter((p) => typeof p.score === 'number');
    if (scored.length === 0) return null;
    const metric = scored[scored.length - 1].headline;
    const series = scored.filter((p) => p.headline === metric);
    const n = series.length;
    const x = (i: number) => (n === 1 ? TREND_W / 2 : 3 + (i * (TREND_W - 6)) / (n - 1));
    const y = (s: number) => 2 + (1 - Math.max(0, Math.min(1, s))) * (TREND_H - 4);
    const first = series[0].score as number;
    const last = series[n - 1].score as number;
    const mixed = series.length < scored.length;
    return (
        <div className="ext-bench__trend" data-testid={`ext-bench-trend-${slug}`}>
            <svg className="ext-bench__spark" width={TREND_W} height={TREND_H} viewBox={`0 0 ${TREND_W} ${TREND_H}`}
                role="img" aria-label={`${benchmarkHeadlineLabel(metric)} over ${n} scored run${n === 1 ? '' : 's'}`}>
                {n > 1 && (
                    <polyline className="ext-bench__spark-line" fill="none"
                        points={series.map((p, i) => `${x(i).toFixed(1)},${y(p.score as number).toFixed(1)}`).join(' ')} />
                )}
                {series.map((p, i) => (
                    <circle key={`${p.experiment_id}-${p.evaluated_at}`} className="ext-bench__spark-dot" cx={x(i)} cy={y(p.score as number)} r={2}>
                        <title>{`Run #${p.experiment_id}${p.experiment_name ? ` (${p.experiment_name})` : ''} · ${String(p.evaluated_at).slice(0, 10)} · ${benchmarkHeadlineLabel(p.headline)} ${(p.score as number).toFixed(2)}`}</title>
                    </circle>
                ))}
            </svg>
            <span className="ext-bench__meta">
                {n === 1
                    ? `${benchmarkHeadlineLabel(metric)} ${last.toFixed(2)} · run #${series[0].experiment_id}`
                    : `${benchmarkHeadlineLabel(metric)} ${first.toFixed(2)} (run #${series[0].experiment_id}) → ${last.toFixed(2)} (run #${series[n - 1].experiment_id}) · ${n} runs`}
                {mixed ? ` · ${scored.length - n} earlier score${scored.length - n === 1 ? '' : 's'} on another metric not drawn` : ''}
            </span>
        </div>
    );
}

export default function ExternalBenchmarksPanel({ projectId, experimentId }: Props) {
    const [benchmarks, setBenchmarks] = useState<BenchmarkInfo[]>([]);
    const [history, setHistory] = useState<Record<number, BenchmarkHistoryPoint[]>>({});
    const [scoring, setScoring] = useState<Record<number, number>>({});
    const [adding, setAdding] = useState(false);

    const load = useCallback(async () => {
        let list: BenchmarkInfo[] = [];
        try {
            list = await listBenchmarks(projectId, experimentId);
        } catch {
            list = [];
        }
        setBenchmarks(list);
        const entries = await Promise.all(list.map(async (b) => {
            try {
                return [b.id, await getBenchmarkHistory(projectId, b.id)] as const;
            } catch {
                return [b.id, []] as const;
            }
        }));
        setHistory(Object.fromEntries(entries));
    }, [projectId, experimentId]);

    useEffect(() => {
        void load();
    }, [load]);

    // While a score is running, re-read until this run's result lands.
    useEffect(() => {
        if (Object.keys(scoring).length === 0) return undefined;
        const timer = window.setInterval(() => void load(), 4000);
        return () => window.clearInterval(timer);
    }, [scoring, load]);
    useEffect(() => {
        setScoring((prev) => {
            const next = { ...prev };
            for (const b of benchmarks) {
                if (next[b.id] != null && b.result) delete next[b.id];
            }
            return Object.keys(next).length === Object.keys(prev).length ? prev : next;
        });
    }, [benchmarks]);

    const score = async (b: BenchmarkInfo) => {
        try {
            const job = await evaluateBenchmark(projectId, b.id, experimentId);
            setScoring((prev) => ({ ...prev, [b.id]: job.id }));
            toast.info(`Scoring ${experimentId != null ? `run #${experimentId}` : 'the latest run'} on ${b.name} (job #${job.id}).`, 4000);
        } catch (err) {
            toast.error(errorText(err, 'Could not start scoring.'), 5000);
        }
    };

    const remove = async (b: BenchmarkInfo) => {
        if (!window.confirm(`Delete the benchmark "${b.name}" and its scores?`)) return;
        try {
            await deleteBenchmark(projectId, b.id);
            void load();
        } catch (err) {
            toast.error(errorText(err, 'Could not delete the benchmark.'), 5000);
        }
    };

    return (
        <section className="card ext-bench" id="external-benchmarks" data-testid="external-benchmarks">
            <div className="ext-bench__head">
                <h3 className="ext-bench__title">External benchmarks</h3>
                <button type="button" className="btn btn-secondary btn-sm" onClick={() => setAdding((v) => !v)} data-testid="ext-bench-add-toggle">
                    {adding ? 'Close' : '+ Add a benchmark'}
                </button>
            </div>
            <p className="ext-bench__hint">
                A labelled set someone else made — scored next to your own answer key. Eval only: it never trains, and rows that
                match your training data are dropped.
            </p>
            {benchmarks.length === 0 && !adding && (
                <p className="ext-bench__empty" data-testid="ext-bench-empty">No benchmarks yet.</p>
            )}
            {benchmarks.length > 0 && (
                <ul className="ext-bench__list">
                    {benchmarks.map((b) => (
                        <li key={b.id} className="ext-bench__item" data-testid={`ext-bench-${b.slug}`}>
                            <div>
                                <strong>{b.name}</strong>{' '}
                                <span className="ext-bench__meta">
                                    {benchmarkSourceText(b)} · {b.rows} rows ·{' '}
                                    {b.scored_labels.length > 0
                                        ? `scores ${b.scored_labels.map(benchmarkLabelText).join(', ')}`
                                        : `${SHAPE_TEXT[b.task_profile] ?? b.task_profile} — scores the answers`}
                                    {b.context_folded ? ' · context folded into the question' : ''}
                                    {b.leakage && (b.leakage.exact + b.leakage.near) > 0
                                        ? ` · ${b.leakage.exact + b.leakage.near} rows dropped (in your training data)`
                                        : ''}
                                </span>
                            </div>
                            <div className="ext-bench__score" data-testid={`ext-bench-score-${b.slug}`}>
                                {experimentId != null ? `Run #${experimentId}: ` : 'Latest run: '}
                                {scoring[b.id] != null ? `scoring (job #${scoring[b.id]})…` : benchmarkScoreText(b.result)}
                            </div>
                            <BenchmarkTrend points={history[b.id] ?? []} slug={b.slug} />
                            <div className="ext-bench__actions">
                                <button type="button" className="btn btn-secondary btn-sm" disabled={scoring[b.id] != null}
                                    onClick={() => void score(b)} data-testid={`ext-bench-evaluate-${b.slug}`}>
                                    {b.result ? 'Re-score' : 'Score this run'}
                                </button>
                                <button type="button" className="btn btn-ghost btn-sm" onClick={() => void remove(b)}>Delete</button>
                            </div>
                        </li>
                    ))}
                </ul>
            )}
            {adding && <AddBenchmarkForm projectId={projectId} onCreated={() => { setAdding(false); void load(); }} />}
        </section>
    );
}

interface AddProps {
    projectId: number;
    onCreated: () => void;
}

function AddBenchmarkForm({ projectId, onCreated }: AddProps) {
    const [locator, setLocator] = useState('');
    const [limit, setLimit] = useState(300);
    const [busy, setBusy] = useState(false);
    const [preview, setPreview] = useState<BenchmarkPreview | null>(null);
    const [mapping, setMapping] = useState<Record<string, BenchmarkMappingValue>>({});
    const [name, setName] = useState('');
    const [licence, setLicence] = useState('');
    const [localOnly, setLocalOnly] = useState(false);
    const [joinAdjacent, setJoinAdjacent] = useState(false);

    const upload = async (file: File | undefined) => {
        if (!file) return;
        setBusy(true);
        try {
            const res = await uploadImportFile(projectId, file);
            setLocator(res.locator);
            if (!name) setName(file.name.replace(/\.[^.]+$/, ''));
        } catch (err) {
            toast.error(errorText(err, 'Upload failed.'), 5000);
        } finally {
            setBusy(false);
        }
    };

    const runPreview = async () => {
        setBusy(true);
        try {
            const p = await previewBenchmark(projectId, locator.trim(), limit);
            setPreview(p);
            setMapping(p.suggested_mapping);
            setLicence(p.licence || '');
            if (!name) setName(p.source.dataset_id.split('/').pop() || 'benchmark');
        } catch (err) {
            setPreview(null);
            toast.error(errorText(err, 'Could not read that source.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const create = async () => {
        setBusy(true);
        try {
            await createBenchmark(projectId, {
                locator: locator.trim(), name: name.trim(), limit, mapping, local_only: localOnly,
                licence: licence.trim() || null, join_adjacent: joinAdjacent,
            });
            toast.success(`Benchmark "${name}" added.`, 4000);
            onCreated();
        } catch (err) {
            toast.error(errorText(err, 'Could not add the benchmark.'), 6000);
        } finally {
            setBusy(false);
        }
    };

    const labelsToMap = preview?.labels_to_map !== false;
    const mapped = Object.values(mapping).filter((v) => mappingMembers(v).length > 0).length;
    const conflict = labelsToMap ? mappingGroupConflict(mapping) : null;
    const canCreate = !busy && !!name.trim() && (!labelsToMap || (mapped > 0 && !conflict));
    const pick = (src: string, select: HTMLSelectElement) => {
        const chosen = Array.from(select.selectedOptions).map((o) => o.value).filter((v) => v !== IGNORE);
        setMapping((m) => ({ ...m, [src]: chosen.length === 0 ? IGNORE : chosen.length === 1 ? chosen[0] : chosen }));
    };
    let createText = 'Add benchmark';
    if (labelsToMap) {
        if (conflict) createText = 'Fix the overlapping groups';
        else if (mapped === 0) createText = 'Map at least one label';
        else createText = `Add benchmark (${mapped} label${mapped === 1 ? '' : 's'} mapped)`;
    }
    return (
        <div className="ext-bench__form" data-testid="ext-bench-form">
            <label className="ext-bench__field">
                Source
                <input className="input" value={locator} onChange={(e) => setLocator(e.target.value)}
                    placeholder="hf:Yelp/yelp_review_full:test  (hf:<dataset>[:<split>[:<revision>]])" data-testid="ext-bench-locator" />
            </label>
            <label className="ext-bench__field">
                …or upload a file (CSV / JSONL / Parquet…)
                <input type="file" onChange={(e) => void upload(e.target.files?.[0])} data-testid="ext-bench-file" />
            </label>
            <label className="ext-bench__field ext-bench__field--narrow">
                Rows (sampled across the source)
                <input className="input" type="number" min={10} max={2000} value={limit}
                    onChange={(e) => setLimit(Math.max(10, Math.min(2000, Number(e.target.value) || 300)))} />
            </label>
            <button type="button" className="btn btn-secondary btn-sm" disabled={busy || !locator.trim()} onClick={() => void runPreview()} data-testid="ext-bench-preview">
                {busy && !preview ? 'Reading…' : 'Preview'}
            </button>

            {preview && (
                <div className="ext-bench__preview" data-testid="ext-bench-preview-result">
                    <p>
                        Detected <strong>{SHAPE_TEXT[preview.task_profile] ?? preview.task_profile}</strong>
                        {' '}({preview.format}) · {preview.rows_scanned} rows read · licence {preview.licence || 'unknown — fill it in below'}
                    </p>
                    {!preview.shape_matches_project && (
                        <p className="ext-bench__warn">This project&apos;s data is {preview.project.task_profile} — the benchmark won&apos;t fit.</p>
                    )}
                    {!labelsToMap && (
                        <p className="ext-bench__hint" data-testid="ext-bench-no-labels">
                            No labels to map — each row is a {preview.task_profile === 'summarization' ? 'document and its summary' : 'question and its answer'}.
                            Scored like your own test examples: exact match and F1, plus the answer judge on long answers when one is available
                            (never a cloud judge for a local-only benchmark).
                            {preview.fields?.context ? ' A context column was found: kept for a project that answers from a context, otherwise put in front of the question.' : ''}
                        </p>
                    )}
                    {labelsToMap && (
                        <>
                            <table className="ext-bench__map">
                                <thead><tr><th>Benchmark label</th><th>Rows / spans</th><th>Your label(s)</th></tr></thead>
                                <tbody>
                                    {Object.entries(preview.source_labels).map(([src, count]) => {
                                        const members = mappingMembers(mapping[src]);
                                        return (
                                            <tr key={src}>
                                                <td><code>{src}</code></td>
                                                <td>{count}</td>
                                                <td>
                                                    <select multiple size={Math.min(4, Math.max(2, preview.project.labels.length))}
                                                        value={members} aria-label={`Map ${src}`}
                                                        onChange={(e) => pick(src, e.target)}
                                                        data-testid={`ext-bench-map-${src}`}>
                                                        {preview.project.labels.map((l) => <option key={l} value={l}>{l}</option>)}
                                                    </select>
                                                    <div className="ext-bench__meta" data-testid={`ext-bench-map-result-${src}`}>
                                                        {members.length === 0 ? 'ignored' : members.length === 1 ? `→ ${members[0]}` : `→ group: ${members.join(' + ')}`}
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                            <p className="ext-bench__hint">
                                Pick one of your labels, several (Ctrl/Cmd-click) to map onto a group — a prediction of any label in the group
                                counts — or none to ignore it. Several benchmark labels can map to one of yours; a label of yours can be in one
                                group only. Ignored labels: classification rows are dropped; spans are removed and those types are not scored.
                            </p>
                            {conflict && <p className="ext-bench__warn" data-testid="ext-bench-map-conflict">{conflict}</p>}
                        </>
                    )}
                    <label className="ext-bench__field">
                        Name
                        <input className="input" value={name} onChange={(e) => setName(e.target.value)} data-testid="ext-bench-name" />
                    </label>
                    <label className="ext-bench__field">
                        Licence
                        <input className="input" value={licence} onChange={(e) => setLicence(e.target.value)} placeholder="e.g. cc-by-4.0, research only" />
                    </label>
                    <label className="ext-bench__check">
                        <input type="checkbox" checked={localOnly} onChange={(e) => setLocalOnly(e.target.checked)} data-testid="ext-bench-local-only" />
                        {' '}Local only — its rows may only be read by models on this machine (e.g. data under a use agreement)
                    </label>
                    {preview.task_profile === 'structured_extraction' && (
                        <label className="ext-bench__check">
                            <input type="checkbox" checked={joinAdjacent} onChange={(e) => setJoinAdjacent(e.target.checked)} />
                            {' '}Join neighbouring spans that map to the same label (e.g. first name + last name → one name)
                        </label>
                    )}
                    <button type="button" className="btn btn-primary btn-sm" disabled={!canCreate}
                        onClick={() => void create()} data-testid="ext-bench-create">
                        {createText}
                    </button>
                </div>
            )}
        </div>
    );
}
