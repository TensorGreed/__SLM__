/**
 * Data page — the project's label taxonomy: a hierarchy of labels with
 * aliases, attributes and per-namespace crosswalk references. Findings
 * importers resolve raw labels against it, so the version + entry count are
 * shown wherever it is used. Collapsed by default; visible for every project.
 */

import { ChevronDown, ChevronRight } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type ReactElement } from 'react';

import {
    deleteTaxonomy,
    getTaxonomy,
    resolveLabel,
    restoreTaxonomy,
    saveTaxonomy,
    taxonomyErrorIssues,
    taxonomyErrorMessage,
    validateTaxonomy,
    type TaxonomyBody,
    type TaxonomyEntry,
    type TaxonomyIssue,
    type TaxonomyResolution,
    type TaxonomyState,
} from '../../api/labelTaxonomy';
import { toast } from '../../stores/toastStore';
import './LabelTaxonomyCard.css';

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');

function groupByCode(issues: TaxonomyIssue[]): Array<[string, TaxonomyIssue[]]> {
    const map = new Map<string, TaxonomyIssue[]>();
    for (const i of issues) map.set(i.code, [...(map.get(i.code) ?? []), i]);
    return Array.from(map.entries());
}

function IssueList({ issues, tone, testId }: { issues: TaxonomyIssue[]; tone: 'error' | 'warning'; testId: string }) {
    if (issues.length === 0) return null;
    return (
        <div data-testid={testId} className={tone === 'error' ? 'lt-issues-error' : 'lt-meta'}>
            {groupByCode(issues).map(([code, items]) => (
                <div key={code}>
                    <strong><code>{code}</code></strong> ({items.length})
                    <ul className="lt-issues">
                        {items.map((i, n) => (
                            <li key={`${code}-${i.id ?? ''}-${n}`}>
                                {i.id ? <code>{i.id}</code> : null}{i.id ? ': ' : ''}{i.message}
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}

function formatAttrs(attrs: Record<string, unknown>): string {
    return Object.entries(attrs)
        .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
        .join(', ');
}

interface TreeProps {
    entries: TaxonomyEntry[];
    query: string;
}

function TaxonomyTree({ entries, query }: TreeProps) {
    const [expanded, setExpanded] = useState<Set<string>>(new Set());
    const byParent = useMemo(() => {
        const m = new Map<string | null, TaxonomyEntry[]>();
        const ids = new Set(entries.map((e) => e.id));
        for (const e of entries) {
            const p = e.parent && ids.has(e.parent) ? e.parent : null;
            m.set(p, [...(m.get(p) ?? []), e]);
        }
        return m;
    }, [entries]);

    const q = norm(query);
    const matches = useCallback(
        (e: TaxonomyEntry) => norm(e.id).includes(q) || norm(e.title).includes(q) || e.aliases.some((a) => norm(a).includes(q)),
        [q],
    );
    // With a query, show matches and their ancestors, all expanded.
    const visible = useMemo(() => {
        if (!q) return null;
        const byId = new Map(entries.map((e) => [e.id, e]));
        const keep = new Set<string>();
        for (const e of entries) {
            if (!matches(e)) continue;
            let cur: TaxonomyEntry | undefined = e;
            const guard = new Set<string>();
            while (cur && !guard.has(cur.id)) {
                guard.add(cur.id);
                keep.add(cur.id);
                cur = cur.parent ? byId.get(cur.parent) : undefined;
            }
        }
        return keep;
    }, [entries, q, matches]);

    const toggle = (id: string) =>
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });

    const render = (parent: string | null, depth: number): ReactElement[] =>
        (byParent.get(parent) ?? [])
            .filter((e) => !visible || visible.has(e.id))
            .map((e) => {
                const kids = (byParent.get(e.id) ?? []).filter((k) => !visible || visible.has(k.id));
                const open = Boolean(visible) || expanded.has(e.id);
                const xw = Object.entries(e.crosswalk ?? {});
                const attrs = formatAttrs(e.attrs ?? {});
                return (
                    <li key={e.id} className="lt-node" data-testid={`taxonomy-node-${e.id}`}>
                        <div className={`lt-node-row lt-depth-${Math.min(depth, 5)}`}>
                            {kids.length > 0 ? (
                                <button type="button" className="btn btn-secondary" onClick={() => toggle(e.id)} data-testid={`taxonomy-toggle-${e.id}`}>
                                    {open ? <ChevronDown className="u-icon" aria-hidden="true" /> : <ChevronRight className="u-icon" aria-hidden="true" />}
                                    <span className="sr-only">{open ? 'Collapse' : 'Expand'} {e.id}</span>
                                </button>
                            ) : null}
                            <code>{e.id}</code>
                            <span>{e.title}</span>
                            {e.aliases.length > 0 && <span className="lt-meta">aliases: {e.aliases.join(', ')}</span>}
                        </div>
                        {(xw.length > 0 || attrs) && (
                            <div className={`lt-meta lt-depth-${Math.min(depth, 5)}`}>
                                {xw.map(([ns, refs]) => (
                                    <span key={ns} data-testid={`taxonomy-xw-${e.id}-${ns}`}>{ns}: {refs.join(', ')}{' '}</span>
                                ))}
                                {attrs && <span data-testid={`taxonomy-attrs-${e.id}`}>{attrs}</span>}
                            </div>
                        )}
                        {open && kids.length > 0 && <ul className="lt-tree">{render(e.id, depth + 1)}</ul>}
                    </li>
                );
            });

    const rows = render(null, 0);
    if (rows.length === 0) return <div className="form-hint" data-testid="taxonomy-no-match">No label matches.</div>;
    return <ul className="lt-tree" data-testid="taxonomy-tree">{rows}</ul>;
}

export default function LabelTaxonomyCard({ projectId }: { projectId: number }) {
    const [open, setOpen] = useState(false);
    const [state, setState] = useState<TaxonomyState | null>(null);
    const [loadError, setLoadError] = useState('');
    const [query, setQuery] = useState('');
    const [text, setText] = useState('');
    const [fmt, setFmt] = useState<'auto' | 'yaml' | 'json'>('auto');
    const [crosswalkFrom, setCrosswalkFrom] = useState('');
    const [busy, setBusy] = useState(false);
    const [errors, setErrors] = useState<TaxonomyIssue[]>([]);
    const [warnings, setWarnings] = useState<TaxonomyIssue[]>([]);
    const [validNote, setValidNote] = useState('');
    const [tryLabel, setTryLabel] = useState('');
    const [resolution, setResolution] = useState<TaxonomyResolution | null>(null);

    // Loaded on mount so the header shows the version + count while collapsed.
    useEffect(() => {
        let cancelled = false;
        getTaxonomy(projectId)
            .then((s) => { if (!cancelled) { setState(s); setLoadError(''); } })
            .catch((err) => { if (!cancelled) setLoadError(taxonomyErrorMessage(err, 'Could not load the taxonomy.')); });
        return () => { cancelled = true; };
    }, [projectId]);

    const body = (): TaxonomyBody => {
        const out: TaxonomyBody = { content: text, format: fmt };
        const from = crosswalkFrom.split(',').map((s) => s.trim()).filter(Boolean);
        if (from.length > 0) out.crosswalk_from = from;
        return out;
    };

    const onFile = async (file: File | undefined) => {
        if (!file) return;
        setText(await file.text());
        const lower = file.name.toLowerCase();
        setFmt(lower.endsWith('.json') ? 'json' : lower.endsWith('.yaml') || lower.endsWith('.yml') ? 'yaml' : 'auto');
        setErrors([]); setWarnings([]); setValidNote('');
    };

    const validate = async () => {
        setBusy(true); setValidNote('');
        try {
            const res = await validateTaxonomy(projectId, body());
            setErrors(res.errors); setWarnings(res.warnings);
            if (res.valid) setValidNote(`Valid: ${res.entry_count} entries${res.version ? `, version ${res.version}` : ''}.`);
        } catch (err) {
            const issues = taxonomyErrorIssues(err);
            if (issues) setErrors(issues);
            else toast.error(taxonomyErrorMessage(err, 'Could not validate.'), 5000);
        } finally {
            setBusy(false);
        }
    };

    const save = async () => {
        setBusy(true); setValidNote('');
        try {
            const next = await saveTaxonomy(projectId, body());
            setState(next); setErrors([]); setWarnings([]);
            toast.success(`Saved taxonomy ${next.version ?? ''} (${next.entry_count} entries).`, 4000);
        } catch (err) {
            const issues = taxonomyErrorIssues(err);
            if (issues) setErrors(issues);
            else toast.error(taxonomyErrorMessage(err, 'Could not save the taxonomy.'), 5000);
        } finally {
            setBusy(false);
        }
    };

    const restore = async (version: string) => {
        setBusy(true);
        try {
            setState(await restoreTaxonomy(projectId, version));
            toast.success(`Restored taxonomy ${version}.`, 4000);
        } catch (err) {
            toast.error(taxonomyErrorMessage(err, 'Could not restore.'), 5000);
        } finally {
            setBusy(false);
        }
    };

    const remove = async () => {
        if (!window.confirm('Delete the label taxonomy? Earlier versions stay in the history.')) return;
        setBusy(true);
        try {
            setState(await deleteTaxonomy(projectId));
            toast.success('Taxonomy deleted; earlier versions are still in the history.', 4000);
        } catch (err) {
            toast.error(taxonomyErrorMessage(err, 'Could not delete.'), 5000);
        } finally {
            setBusy(false);
        }
    };

    const tryIt = async () => {
        if (!tryLabel.trim()) return;
        try {
            setResolution(await resolveLabel(projectId, tryLabel));
        } catch (err) {
            toast.error(taxonomyErrorMessage(err, 'Could not resolve.'), 5000);
        }
    };

    const entries = state?.taxonomy?.entries ?? [];

    return (
        <section className="card" data-testid="label-taxonomy-card">
            <div className="lt-header">
                <h3 className="u-card-title u-mb0">Label taxonomy</h3>
                {state?.version && <span className="lt-chip" data-testid="taxonomy-version">version {state.version}</span>}
                {state && <span className="lt-chip" data-testid="taxonomy-count">{state.entry_count} entries</span>}
                <button type="button" className="btn btn-secondary" onClick={() => setOpen((v) => !v)} data-testid="taxonomy-toggle">
                    {open ? 'Hide' : 'Show'}
                </button>
            </div>
            {open && (
                <div>
                    <p className="form-hint u-mt-sm">
                        The label set findings importers resolve raw labels against: ids, aliases and crosswalk references per namespace.
                    </p>
                    {loadError && <div className="error-banner" data-testid="taxonomy-load-error">{loadError}</div>}
                    {state && !state.taxonomy && (
                        <div className="form-hint" data-testid="taxonomy-empty">No taxonomy yet. Imported labels are kept as-is until you add one.</div>
                    )}
                    {state?.taxonomy && (
                        <div className="lt-section">
                            <div className="form-hint">
                                {state.taxonomy.name ? <>{state.taxonomy.name}{' '}</> : null}
                                {state.taxonomy.source_version ? <>(source version {state.taxonomy.source_version}) </> : null}
                                {state.namespaces.length > 0 && <>Namespaces: {state.namespaces.join(', ')}</>}
                            </div>
                            <input
                                className="form-input u-mb-sm"
                                placeholder="Search id, title or alias"
                                aria-label="Search labels"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                data-testid="taxonomy-search"
                            />
                            <TaxonomyTree entries={entries} query={query} />
                        </div>
                    )}

                    <div className="lt-section">
                        <h4 className="u-card-title">Try a label</h4>
                        <div className="lt-actions">
                            <input
                                className="form-input"
                                placeholder="A raw label, e.g. leak 1"
                                aria-label="Raw label to resolve"
                                value={tryLabel}
                                onChange={(e) => setTryLabel(e.target.value)}
                                data-testid="taxonomy-try-input"
                            />
                            <button type="button" className="btn btn-secondary" onClick={() => void tryIt()} data-testid="taxonomy-try-btn">Resolve</button>
                        </div>
                        {resolution && (
                            <div className="form-hint" data-testid="taxonomy-try-result">
                                {resolution.id
                                    ? <>{resolution.raw} resolves to <code>{resolution.id}</code>{resolution.title ? ` (${resolution.title})` : ''}, matched by {resolution.matched_by}.</>
                                    : resolution.matched_by === 'ambiguous'
                                        ? <>{resolution.raw} is ambiguous: it could be {resolution.candidates.map((c) => <code key={c}>{c} </code>)}</>
                                        : <>{resolution.raw} matches no label.</>}
                            </div>
                        )}
                    </div>

                    <div className="lt-section">
                        <h4 className="u-card-title">Edit</h4>
                        <div className="form-hint">Upload a .yaml, .yml or .json file, or paste the text. Saving replaces the taxonomy; the old one stays in the history.</div>
                        <input
                            type="file"
                            accept=".yaml,.yml,.json"
                            aria-label="Taxonomy file"
                            onChange={(e) => void onFile(e.target.files?.[0])}
                            data-testid="taxonomy-file"
                        />
                        <textarea
                            className="form-input lt-textarea u-mt-sm"
                            aria-label="Taxonomy text"
                            placeholder={'- id: LEAK-1\n  title: Data leak\n  aliases: [exposure]'}
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            data-testid="taxonomy-text"
                        />
                        <div className="lt-actions">
                            <input
                                className="form-input"
                                placeholder="Crosswalk from fields (comma list)"
                                aria-label="Crosswalk from fields"
                                value={crosswalkFrom}
                                onChange={(e) => setCrosswalkFrom(e.target.value)}
                                data-testid="taxonomy-crosswalk-from"
                            />
                        </div>
                        <div className="form-hint">Fields holding NAMESPACE:ref strings become crosswalk references.</div>
                        <div className="lt-actions">
                            <button type="button" className="btn btn-secondary" disabled={busy || !text.trim()} onClick={() => void validate()} data-testid="taxonomy-validate">Validate</button>
                            <button type="button" className="btn btn-primary" disabled={busy || !text.trim()} onClick={() => void save()} data-testid="taxonomy-save">Save</button>
                            {state?.taxonomy && (
                                <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => void remove()} data-testid="taxonomy-delete">Delete</button>
                            )}
                        </div>
                        {validNote && <div className="form-hint" role="status" data-testid="taxonomy-valid">{validNote}</div>}
                        <IssueList issues={errors} tone="error" testId="taxonomy-errors" />
                        <IssueList issues={warnings} tone="warning" testId="taxonomy-warnings" />
                    </div>

                    {state && state.history.length > 0 && (
                        <div className="lt-section">
                            <h4 className="u-card-title">History</h4>
                            <ul className="lt-history" data-testid="taxonomy-history">
                                {state.history.map((h) => (
                                    <li key={h.version} data-testid={`taxonomy-history-${h.version}`}>
                                        <code>{h.version}</code>
                                        <span className="lt-meta">{h.entry_count} entries, saved {h.saved_at}</span>
                                        {h.current
                                            ? <span className="lt-chip">current</span>
                                            : !h.restorable
                                                ? <span className="lt-meta" data-testid={`taxonomy-stub-${h.version}`}>snapshot no longer kept</span>
                                                : <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => void restore(h.version)} data-testid={`taxonomy-restore-${h.version}`}>Restore</button>}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </section>
    );
}
