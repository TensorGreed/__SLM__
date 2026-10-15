/**
 * External benchmarks (Epic G-B1) — score the selected run on labelled sets
 * someone else made, next to the project's own answer key. Import: a source
 * (Hugging Face dataset or an uploaded file) → preview (detected shape,
 * its labels, the project's) → map every source label onto a project label
 * or "ignore" (many-to-one allowed) → a locked, eval-only benchmark with
 * leakage-checked rows. Scores run as `benchmark_eval` Jobs.
 */
import { useCallback, useEffect, useState } from 'react';

import {
    benchmarkScoreText,
    benchmarkSourceText,
    createBenchmark,
    deleteBenchmark,
    evaluateBenchmark,
    listBenchmarks,
    previewBenchmark,
    type BenchmarkInfo,
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

export default function ExternalBenchmarksPanel({ projectId, experimentId }: Props) {
    const [benchmarks, setBenchmarks] = useState<BenchmarkInfo[]>([]);
    const [scoring, setScoring] = useState<Record<number, number>>({});
    const [adding, setAdding] = useState(false);

    const load = useCallback(async () => {
        try {
            setBenchmarks(await listBenchmarks(projectId, experimentId));
        } catch {
            setBenchmarks([]);
        }
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
                                    {benchmarkSourceText(b)} · {b.rows} rows · scores {b.scored_labels.join(', ')}
                                    {b.leakage && (b.leakage.exact + b.leakage.near) > 0
                                        ? ` · ${b.leakage.exact + b.leakage.near} rows dropped (in your training data)`
                                        : ''}
                                </span>
                            </div>
                            <div className="ext-bench__score" data-testid={`ext-bench-score-${b.slug}`}>
                                {experimentId != null ? `Run #${experimentId}: ` : 'Latest run: '}
                                {scoring[b.id] != null ? `scoring (job #${scoring[b.id]})…` : benchmarkScoreText(b.result)}
                            </div>
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
    const [mapping, setMapping] = useState<Record<string, string>>({});
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

    const mapped = Object.values(mapping).filter((v) => v !== IGNORE).length;
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
                        Detected <strong>{preview.task_profile === 'classification' ? 'classification' : 'spans (entity tagging)'}</strong>
                        {' '}({preview.format}) · {preview.rows_scanned} rows read · licence {preview.licence || 'unknown — fill it in below'}
                    </p>
                    {!preview.shape_matches_project && (
                        <p className="ext-bench__warn">This project&apos;s data is {preview.project.task_profile} — the benchmark won&apos;t fit.</p>
                    )}
                    <table className="ext-bench__map">
                        <thead><tr><th>Benchmark label</th><th>Rows / spans</th><th>Your label</th></tr></thead>
                        <tbody>
                            {Object.entries(preview.source_labels).map(([src, count]) => (
                                <tr key={src}>
                                    <td><code>{src}</code></td>
                                    <td>{count}</td>
                                    <td>
                                        <select value={mapping[src] ?? IGNORE} aria-label={`Map ${src}`}
                                            onChange={(e) => setMapping((m) => ({ ...m, [src]: e.target.value }))}
                                            data-testid={`ext-bench-map-${src}`}>
                                            <option value={IGNORE}>ignore</option>
                                            {preview.project.labels.map((l) => <option key={l} value={l}>{l}</option>)}
                                        </select>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p className="ext-bench__hint">
                        Several benchmark labels can map to one of yours. Ignored labels: classification rows are dropped; spans are removed
                        and those types are not scored.
                    </p>
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
                    <button type="button" className="btn btn-primary btn-sm" disabled={busy || mapped === 0 || !name.trim()}
                        onClick={() => void create()} data-testid="ext-bench-create">
                        {mapped === 0 ? 'Map at least one label' : `Add benchmark (${mapped} label${mapped === 1 ? '' : 's'} mapped)`}
                    </button>
                </div>
            )}
        </div>
    );
}
