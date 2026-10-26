import { useEffect, useState } from 'react';

import {
    startGitRepoImport,
    startWebCrawl,
    type DocumentSourceResult,
    type ImportedDocument,
    type RejectedSource,
} from '../../api/documentSources';
import api from '../../api/client';
import { getJob, type Job } from '../../api/jobs';
import { toast } from '../../stores/toastStore';
import { extractApiErrorMessage } from '../../utils/apiError';
import './DocumentSourceImportCard.css';

interface DocumentSourceImportCardProps {
    projectId: number;
    /** Called when an import finishes or documents are dropped, so the documents list refreshes. */
    onDone?: () => void;
    /** Job poll interval in ms (tests shorten it). */
    pollMs?: number;
}

type Tab = 'web' | 'git';

const lines = (text: string): string[] => text.split('\n').map((l) => l.trim()).filter(Boolean);
const rejectedKey = (r: RejectedSource): string => `${r.reason}|${r.url ?? r.path ?? ''}`;
const isTerminal = (j: Job | null): boolean =>
    j != null && (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled');

/** "Documents from website / repo": crawl a documentation site or import a
 *  git repository as project documents. Runs as a Job (polled here and
 *  tracked by the bell); the result lists what was imported with its
 *  provenance and what was rejected, grouped by reason, so the user can retry
 *  or drop in bulk. */
export default function DocumentSourceImportCard({ projectId, onDone, pollMs = 2000 }: DocumentSourceImportCardProps) {
    const [tab, setTab] = useState<Tab>('web');
    // Website form.
    const [seeds, setSeeds] = useState('');
    const [prefixes, setPrefixes] = useState('');
    const [sitemap, setSitemap] = useState('');
    const [maxPages, setMaxPages] = useState(200);
    const [maxDepth, setMaxDepth] = useState(3);
    const [delay, setDelay] = useState(1);
    const [webInclude, setWebInclude] = useState('');
    const [webExclude, setWebExclude] = useState('');
    const [webLicence, setWebLicence] = useState('');
    // Git form.
    const [repoUrl, setRepoUrl] = useState('');
    const [ref, setRef] = useState('HEAD');
    const [gitInclude, setGitInclude] = useState('');
    const [gitExclude, setGitExclude] = useState('');
    const [defaultExcludes, setDefaultExcludes] = useState(true);
    const [maxFileKb, setMaxFileKb] = useState(200);

    const [starting, setStarting] = useState(false);
    const [jobId, setJobId] = useState<number | null>(null);
    const [job, setJob] = useState<Job | null>(null);
    const [pickedRejected, setPickedRejected] = useState<Set<string>>(new Set());
    const [pickedDocs, setPickedDocs] = useState<Set<number>>(new Set());
    const [deleting, setDeleting] = useState(false);
    const [droppedIds, setDroppedIds] = useState<Set<number>>(new Set());

    useEffect(() => {
        if (jobId == null) return undefined;
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const tick = async () => {
            try {
                const j = await getJob(jobId);
                if (cancelled) return;
                setJob(j);
                if (isTerminal(j)) {
                    if (j.status === 'succeeded') onDone?.();
                    return;
                }
            } catch {
                if (cancelled) return;
            }
            timer = setTimeout(() => void tick(), pollMs);
        };
        void tick();
        return () => {
            cancelled = true;
            if (timer) clearTimeout(timer);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [jobId, pollMs]);

    const running = jobId != null && !isTerminal(job);
    const result = job?.status === 'succeeded' ? (job.result as DocumentSourceResult | null) : null;

    const begin = async (starter: () => Promise<Job>, what: string) => {
        setStarting(true);
        try {
            const started = await starter();
            setJob(started);
            setJobId(started.id);
            setPickedRejected(new Set());
            setPickedDocs(new Set());
            setDroppedIds(new Set());
            toast.info(`${what} started. The bell tracks it.`, 4000);
        } catch (err) {
            toast.error(extractApiErrorMessage(err, `Could not start: ${what}.`).message, 5000);
        } finally {
            setStarting(false);
        }
    };

    // A retry re-fetches exactly the picked URLs: the include / exclude patterns and the
    // sitemap that rejected or listed them are not sent again.
    const startWeb = (seedUrls: string[], depth: number, retry = false) => begin(() => startWebCrawl(projectId, {
        seed_urls: seedUrls,
        ...(lines(prefixes).length ? { allowed_prefixes: lines(prefixes) } : {}),
        ...(!retry && sitemap.trim() ? { sitemap_url: sitemap.trim() } : {}),
        max_pages: maxPages,
        max_depth: depth,
        delay_seconds: delay,
        include_patterns: retry ? [] : lines(webInclude),
        exclude_patterns: retry ? [] : lines(webExclude),
        ...(webLicence.trim() ? { licence: webLicence.trim() } : {}),
    }), retry ? 'Retry of rejected pages' : 'Website crawl');

    const startGit = () => begin(() => startGitRepoImport(projectId, {
        url: repoUrl.trim(),
        ref: ref.trim() || 'HEAD',
        include_globs: lines(gitInclude),
        exclude_globs: lines(gitExclude),
        default_excludes: defaultExcludes,
        max_file_bytes: Math.max(1, maxFileKb) * 1000,
    }), 'Repository import');

    const retrySelected = () => {
        const urls = (result?.rejected ?? [])
            .filter((r) => r.url && pickedRejected.has(rejectedKey(r)))
            .map((r) => r.url as string);
        if (urls.length) void startWeb(Array.from(new Set(urls)), 0, true);
    };

    const toggle = <T,>(set: Set<T>, value: T, put: (s: Set<T>) => void) => {
        const next = new Set(set);
        if (next.has(value)) next.delete(value); else next.add(value);
        put(next);
    };

    const groups = new Map<string, RejectedSource[]>();
    for (const r of result?.rejected ?? []) {
        groups.set(r.reason, [...(groups.get(r.reason) ?? []), r]);
    }
    const toggleGroup = (items: RejectedSource[]) => {
        const keys = items.map(rejectedKey);
        const all = keys.every((k) => pickedRejected.has(k));
        const next = new Set(pickedRejected);
        for (const k of keys) { if (all) next.delete(k); else next.add(k); }
        setPickedRejected(next);
    };

    const docs: ImportedDocument[] = (result?.documents ?? []).filter((d) => !droppedIds.has(d.id));
    const provenance = (d: ImportedDocument): string => {
        if (result?.source === 'git_repo') {
            const at = result.commit ? ` @ ${result.commit.slice(0, 7)}` : '';
            const lic = result.licence ? ` · ${result.licence}` : '';
            return `${d.path ?? d.filename}${at}${lic}`;
        }
        return d.source_url ?? d.filename;
    };

    const deleteSelected = async () => {
        const ids = Array.from(pickedDocs);
        if (!ids.length || !window.confirm(`Delete ${ids.length} imported document(s)?`)) return;
        setDeleting(true);
        const gone = new Set(droppedIds);
        let failed = 0;
        for (const id of ids) {
            try {
                await api.delete(`/projects/${projectId}/ingestion/documents/${id}`);
                gone.add(id);
            } catch {
                failed += 1;
            }
        }
        setDroppedIds(gone);
        setPickedDocs(new Set());
        setDeleting(false);
        if (failed) toast.error(`${failed} document(s) could not be deleted.`, 5000);
        onDone?.();
    };

    const canStart = tab === 'web' ? lines(seeds).length > 0 : repoUrl.trim().length > 0;

    return (
        <div className="card document-source-card" data-testid="document-source-card" id="document-sources">
            <div className="card-header">
                <h3>Documents from website / repo</h3>
            </div>
            <p className="text-muted">
                Crawl a documentation site (robots.txt respected) or import a git repository. Pages and files become
                project documents with their source recorded, then are cleaned into passages like an upload.
            </p>
            <div className="dsi__tabs" role="tablist" aria-label="Document source">
                <button type="button" role="tab" className={`btn btn-sm ${tab === 'web' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setTab('web')} data-testid="dsi-tab-web">Website</button>
                <button type="button" role="tab" className={`btn btn-sm ${tab === 'git' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setTab('git')} data-testid="dsi-tab-git">Git repository</button>
            </div>

            {!running && tab === 'web' && (
                <div className="dsi__form">
                    <label>
                        Seed URLs (one per line)
                        <textarea rows={3} value={seeds} onChange={(e) => setSeeds(e.target.value)}
                            placeholder="https://docs.example.com/guide/" data-testid="dsi-seeds" />
                    </label>
                    <label>
                        Licence (optional)
                        <input type="text" value={webLicence} onChange={(e) => setWebLicence(e.target.value)}
                            placeholder="e.g. CC-BY-4.0" data-testid="dsi-licence" />
                    </label>
                    <details className="dsi__adv">
                        <summary>Advanced</summary>
                        <div className="dsi__form">
                            <label>
                                Allowed prefixes (one per line; default: each seed&apos;s folder)
                                <textarea rows={2} value={prefixes} onChange={(e) => setPrefixes(e.target.value)} data-testid="dsi-prefixes" />
                            </label>
                            <label>
                                Sitemap URL
                                <input type="text" value={sitemap} onChange={(e) => setSitemap(e.target.value)}
                                    placeholder="https://docs.example.com/sitemap.xml" data-testid="dsi-sitemap" />
                            </label>
                            <div className="dsi__row">
                                <label>Max pages
                                    <input type="number" min={1} max={5000} value={maxPages}
                                        onChange={(e) => setMaxPages(Number(e.target.value))} data-testid="dsi-max-pages" />
                                </label>
                                <label>Max depth
                                    <input type="number" min={0} max={10} value={maxDepth}
                                        onChange={(e) => setMaxDepth(Number(e.target.value))} data-testid="dsi-max-depth" />
                                </label>
                                <label>Delay between requests (s)
                                    <input type="number" min={0} max={30} step={0.5} value={delay}
                                        onChange={(e) => setDelay(Number(e.target.value))} data-testid="dsi-delay" />
                                </label>
                            </div>
                            <label>
                                Include URLs matching (regex, one per line)
                                <textarea rows={2} value={webInclude} onChange={(e) => setWebInclude(e.target.value)} data-testid="dsi-web-include" />
                            </label>
                            <label>
                                Exclude URLs matching (regex, one per line)
                                <textarea rows={2} value={webExclude} onChange={(e) => setWebExclude(e.target.value)} data-testid="dsi-web-exclude" />
                            </label>
                        </div>
                    </details>
                    <div className="dsi__actions">
                        <button type="button" className="btn btn-primary" disabled={starting || !canStart}
                            onClick={() => void startWeb(lines(seeds), maxDepth)} data-testid="dsi-start-web">
                            {starting ? 'Starting…' : 'Crawl website'}
                        </button>
                    </div>
                </div>
            )}

            {!running && tab === 'git' && (
                <div className="dsi__form">
                    <div className="dsi__row">
                        <label>
                            Repository URL
                            <input type="text" value={repoUrl} onChange={(e) => setRepoUrl(e.target.value)}
                                placeholder="https://github.com/org/repo" data-testid="dsi-repo-url" />
                        </label>
                        <label>
                            Branch, tag or commit
                            <input type="text" value={ref} onChange={(e) => setRef(e.target.value)} data-testid="dsi-ref" />
                        </label>
                    </div>
                    <details className="dsi__adv">
                        <summary>Advanced</summary>
                        <div className="dsi__form">
                            <label>
                                Include globs (one per line; empty = everything supported)
                                <textarea rows={2} value={gitInclude} onChange={(e) => setGitInclude(e.target.value)}
                                    placeholder="src/**" data-testid="dsi-git-include" />
                            </label>
                            <label>
                                Exclude globs (one per line; added to the built-in defaults)
                                <textarea rows={2} value={gitExclude} onChange={(e) => setGitExclude(e.target.value)}
                                    placeholder="tests/**" data-testid="dsi-git-exclude" />
                            </label>
                            <label className="dsi__check">
                                <input type="checkbox" checked={defaultExcludes}
                                    onChange={(e) => setDefaultExcludes(e.target.checked)} data-testid="dsi-default-excludes" />
                                {' '}Skip vendored / build folders (default)
                            </label>
                            <label>
                                Max file size (KB)
                                <input type="number" min={1} value={maxFileKb}
                                    onChange={(e) => setMaxFileKb(Number(e.target.value))} data-testid="dsi-max-file-kb" />
                            </label>
                        </div>
                    </details>
                    <div className="dsi__actions">
                        <button type="button" className="btn btn-primary" disabled={starting || !canStart}
                            onClick={() => void startGit()} data-testid="dsi-start-git">
                            {starting ? 'Starting…' : 'Import repository'}
                        </button>
                    </div>
                </div>
            )}

            {running && (
                <p data-testid="dsi-progress">{job?.progress_message || 'Working…'}</p>
            )}
            {job?.status === 'failed' && (
                <p className="text-danger" role="alert" data-testid="dsi-error">{job.error || 'The import failed.'}</p>
            )}

            {result && (
                <div className="dsi__result" data-testid="dsi-result">
                    <p className="u-m0" data-testid="dsi-summary">
                        {result.documents_created} documents imported
                        {result.source === 'git_repo' && result.commit ? ` from ${result.ref ?? 'HEAD'} @ ${result.commit.slice(0, 7)}` : ''}
                        {result.source === 'git_repo' && result.licence ? ` · licence ${result.licence}` : ''}
                        {result.limit_reached > 0 ? ` · ${result.limit_reached} more skipped at the limit` : ''}.
                    </p>

                    {docs.length > 0 && (
                        <details className="dsi__group" open data-testid="dsi-imported">
                            <summary>Imported documents ({docs.length})</summary>
                            <div className="dsi__actions">
                                <button type="button" className="btn btn-secondary btn-sm"
                                    onClick={() => setPickedDocs(pickedDocs.size === docs.length ? new Set() : new Set(docs.map((d) => d.id)))}
                                    data-testid="dsi-docs-select-all">
                                    {pickedDocs.size === docs.length ? 'Clear selection' : 'Select all'}
                                </button>
                                <button type="button" className="btn btn-secondary btn-sm" disabled={!pickedDocs.size || deleting}
                                    onClick={() => void deleteSelected()} data-testid="dsi-delete-selected">
                                    Delete selected ({pickedDocs.size})
                                </button>
                            </div>
                            <ul className="dsi__list">
                                {docs.map((d) => (
                                    <li key={d.id} className="dsi__item">
                                        <input type="checkbox" checked={pickedDocs.has(d.id)}
                                            onChange={() => toggle(pickedDocs, d.id, setPickedDocs)}
                                            aria-label={`Select ${d.filename}`} data-testid={`dsi-doc-${d.id}`} />
                                        <span data-testid={`dsi-doc-label-${d.id}`}>{provenance(d)}</span>
                                    </li>
                                ))}
                            </ul>
                        </details>
                    )}

                    {Array.from(groups.entries()).map(([reason, items]) => (
                        <details key={reason} className="dsi__group" data-testid={`dsi-group-${reason}`}>
                            <summary>
                                {reason.replace(/_/g, ' ')} ({result.rejection_counts?.[reason] ?? items.length})
                            </summary>
                            <div className="dsi__actions">
                                <button type="button" className="btn btn-secondary btn-sm"
                                    onClick={() => toggleGroup(items)} data-testid={`dsi-group-all-${reason}`}>
                                    Select all in group
                                </button>
                            </div>
                            <ul className="dsi__list">
                                {items.map((r, index) => (
                                    <li key={`${rejectedKey(r)}#${index}`} className="dsi__item">
                                        <input type="checkbox" checked={pickedRejected.has(rejectedKey(r))}
                                            onChange={() => toggle(pickedRejected, rejectedKey(r), setPickedRejected)}
                                            aria-label={`Select ${r.url ?? r.path ?? reason}`} />
                                        <span>{r.url ?? r.path}{r.detail != null && r.detail !== '' ? ` (${r.detail})` : ''}</span>
                                    </li>
                                ))}
                            </ul>
                        </details>
                    ))}

                    {result.source === 'web_crawl' && groups.size > 0 && (
                        <div className="dsi__actions">
                            <button type="button" className="btn btn-secondary btn-sm"
                                disabled={starting || !(result.rejected ?? []).some((r) => r.url && pickedRejected.has(rejectedKey(r)))}
                                onClick={retrySelected} data-testid="dsi-retry-selected">
                                Retry selected
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
