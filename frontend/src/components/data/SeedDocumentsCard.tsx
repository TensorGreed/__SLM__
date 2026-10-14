import { useEffect, useState } from 'react';

import { fetchSeedDocumentsDefaults, startSeedDocuments } from '../../api/flows';
import { useJobsStore } from '../../stores/jobsStore';
import { toast } from '../../stores/toastStore';

interface SeedDocumentsCardProps {
    projectId: number;
    /** Called when the Job succeeds, so the documents list can refresh. */
    onDone?: () => void;
}

/** "Write sample documents": when there is no corpus yet, a generation
 *  model writes realistic (invented) documents in the genres the user
 *  names, containing what the user says they must contain; they are
 *  ingested as one text document and cleaned into passages, so every
 *  later step (span synthesis, split, training) is the same as for
 *  uploaded documents. Runs as a Job the bell tracks. */
export default function SeedDocumentsCard({ projectId, onDone }: SeedDocumentsCardProps) {
    const jobs = useJobsStore((s) => s.jobs);
    const [open, setOpen] = useState(false);
    const [kinds, setKinds] = useState('');
    const [genres, setGenres] = useState('');
    const [count, setCount] = useState(60);
    const [words, setWords] = useState(160);
    const [paged, setPaged] = useState(false);
    const [starting, setStarting] = useState(false);
    const [startedJobId, setStartedJobId] = useState<number | null>(null);

    useEffect(() => {
        let cancelled = false;
        fetchSeedDocumentsDefaults(projectId)
            .then((d) => {
                if (cancelled) return;
                setGenres((prev) => prev || d.genres.join('\n'));
                setCount((prev) => prev || d.count);
                setWords((prev) => prev || d.words);
            })
            .catch(() => undefined);
        return () => {
            cancelled = true;
        };
    }, [projectId]);

    const job = startedJobId != null ? jobs.find((j) => j.id === startedJobId) ?? null : null;
    const running = job != null && (job.status === 'queued' || job.status === 'running');
    const result = job?.status === 'succeeded' ? (job.result as Record<string, unknown> | null) : null;

    useEffect(() => {
        if (job?.status === 'succeeded') onDone?.();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [job?.status]);

    const start = async () => {
        setStarting(true);
        try {
            const started = await startSeedDocuments(projectId, {
                kinds,
                genres: genres.split('\n').map((g) => g.trim()).filter(Boolean),
                count: Math.max(1, count),
                words: Math.max(60, words),
                layout: paged ? 'paged' : 'plain',
            });
            setStartedJobId(started.id);
            toast.info(`Writing ${count} sample documents — the bell tracks it.`, 4000);
        } catch (err) {
            toast.error(err instanceof Error ? err.message : 'Could not start writing documents.', 5000);
        } finally {
            setStarting(false);
        }
    };

    return (
        <div className="card seed-documents-card" data-testid="seed-documents-card" id="seed-documents">
            <div className="card-header">
                <h3>No documents yet? Write sample documents</h3>
                <button type="button" className="btn btn-secondary" onClick={() => setOpen((v) => !v)} data-testid="seed-documents-toggle">
                    {open ? 'Hide' : 'Write sample documents'}
                </button>
            </div>
            <p className="text-muted">
                A generation model (your local model first) writes realistic, invented documents in the genres you name,
                containing what you say they must contain. They are ingested like an upload and cleaned into passages,
                so everything after this step — labelling, splitting, training — is unchanged.
            </p>
            {open && !running && !result && (
                <div className="seed-documents-card__form">
                    <label>
                        What the documents must contain
                        <textarea
                            value={kinds}
                            onChange={(e) => setKinds(e.target.value)}
                            rows={3}
                            placeholder="e.g. card numbers, expiry dates, IBANs, names, emails, phone numbers, dates of birth, national ids, medical record numbers, diagnoses…"
                            data-testid="seed-documents-kinds"
                        />
                    </label>
                    <label>
                        Genres (one per line)
                        <textarea value={genres} onChange={(e) => setGenres(e.target.value)} rows={6} data-testid="seed-documents-genres" />
                    </label>
                    <div className="seed-documents-card__row">
                        <label>
                            Documents
                            <input type="number" min={1} max={400} value={count} onChange={(e) => setCount(e.target.value === '' ? 0 : Number(e.target.value))} data-testid="seed-documents-count" />
                        </label>
                        <label>
                            Words each
                            <input type="number" min={60} max={400} value={words} onChange={(e) => setWords(e.target.value === '' ? 0 : Number(e.target.value))} data-testid="seed-documents-words" />
                        </label>
                        <label title="Splits each document into pages and adds header and footer lines (a letterhead, 'Page 2 of 5', a confidentiality line, a print date) — so a model trained on them learns that page furniture is not data.">
                            <input type="checkbox" checked={paged} onChange={(e) => setPaged(e.target.checked)} data-testid="seed-documents-paged" />
                            {' '}Multi-page, with headers and footers
                        </label>
                        <button type="button" className="btn btn-primary" onClick={() => void start()} disabled={starting || count < 1} data-testid="seed-documents-start">
                            {starting ? 'Starting…' : `Write ${count} documents`}
                        </button>
                    </div>
                </div>
            )}
            {running && (
                <p data-testid="seed-documents-progress">
                    {job?.progress_message || 'Writing documents…'}
                </p>
            )}
            {result && (
                <p data-testid="seed-documents-result">
                    Done. {String(result.written)} documents written ({String(result.backend)}) → {String(result.filename)},{' '}
                    {typeof result.passages === 'number' ? `${result.passages} passages` : 'cleaned into passages'}.
                    {Array.isArray(result.warnings) && result.warnings.length > 0 ? ` ${(result.warnings as string[]).join(' ')}` : ''}
                </p>
            )}
            {job?.status === 'failed' && (
                <p className="text-danger" data-testid="seed-documents-error">{job.error || 'Writing documents failed.'}</p>
            )}
        </div>
    );
}
