/**
 * SeedSourcesSelect — multi-select of the project's data sources for the
 * playbooks that draw seeds from documents (code chunks / cleaned passages).
 * Emits the chosen ids (sent as `params.source_ids`) or null for "all sources".
 * Sources marked evaluation-only or retrieval-only stay listed but are never
 * used for training data (the licence gate applies on top of the choice).
 *
 * Below the list, a compact "Seed filters" block narrows the seed passages further
 * (`onFiltersChange`, sent as `seed_path_include` / `seed_path_exclude` globs on the chunk's
 * repo-relative path, `seed_relevance_terms` + `seed_relevance_scope`, and
 * `relevance_exempt_source_ids`). Filters apply before the 2,000-passage cap.
 */

import { useEffect, useState } from 'react';

import { listProvenanceSources, USAGE_LABEL, type ProvenanceSource } from '../../api/provenance';

/** The seed-filter params this block emits (only the keys that are set). */
export type SeedFilterParams = {
    seed_path_include?: string[];
    seed_path_exclude?: string[];
    seed_relevance_terms?: string[];
    seed_relevance_scope?: 'chunk' | 'file';
    seed_relevance_match?: 'path_or_text' | 'text';
    relevance_exempt_source_ids?: number[];
};

interface Props {
    projectId: number;
    onChange: (ids: number[] | null) => void;
    onFiltersChange?: (filters: SeedFilterParams | null) => void;
}

/** One entry per non-empty line (commas also split the terms box), trimmed, de-duplicated. */
export function splitLines(text: string, commas = false): string[] {
    const parts = text.split(commas ? /[\n,]/ : /\n/).map((x) => x.trim()).filter(Boolean);
    return Array.from(new Set(parts));
}

export default function SeedSourcesSelect({ projectId, onChange, onFiltersChange }: Props) {
    const [sources, setSources] = useState<ProvenanceSource[]>([]);
    const [picked, setPicked] = useState<number[]>([]);
    const [include, setInclude] = useState('');
    const [exclude, setExclude] = useState('');
    const [terms, setTerms] = useState('');
    const [scope, setScope] = useState<'chunk' | 'file'>('chunk');
    const [match, setMatch] = useState<'path_or_text' | 'text'>('path_or_text');
    const [exempt, setExempt] = useState<number[]>([]);

    useEffect(() => {
        let cancelled = false;
        listProvenanceSources(projectId)
            .then((res) => { if (!cancelled) setSources(res?.sources ?? []); })
            .catch(() => { /* optional: with no list the run uses every source */ });
        return () => { cancelled = true; };
    }, [projectId]);

    useEffect(() => {
        onChange(picked.length ? picked : null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [picked]);

    useEffect(() => {
        if (!onFiltersChange) return;
        const out: SeedFilterParams = {};
        const inc = splitLines(include);
        const exc = splitLines(exclude);
        const trm = splitLines(terms, true);
        if (inc.length) out.seed_path_include = inc;
        if (exc.length) out.seed_path_exclude = exc;
        if (trm.length) {
            out.seed_relevance_terms = trm;
            if (scope === 'file') out.seed_relevance_scope = 'file';
            if (match === 'text') out.seed_relevance_match = 'text';
            if (exempt.length) out.relevance_exempt_source_ids = exempt;
        }
        onFiltersChange(Object.keys(out).length ? out : null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [include, exclude, terms, scope, match, exempt]);

    if (!sources.length) return null;

    const toggle = (id: number) =>
        setPicked((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(0, 100)));

    const toggleExempt = (id: number) =>
        setExempt((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(0, 100)));
    const hasTerms = splitLines(terms, true).length > 0;

    return (
        <>
        <fieldset className="code-findings-form__group" data-testid="seed-sources">
            <legend>Seed sources</legend>
            <p className="u-hint u-mt0">
                {picked.length
                    ? `Seeds come only from ${picked.length} chosen source${picked.length === 1 ? '' : 's'}.`
                    : 'None chosen: seeds are spread across every source (large projects are sampled, not cut off).'}
            </p>
            {sources.map((s) => (
                <label key={s.id} className="seed-sources__option">
                    <input
                        type="checkbox"
                        checked={picked.includes(s.id)}
                        onChange={() => toggle(s.id)}
                        data-testid={`seed-source-${s.id}`}
                    />
                    <span>
                        {s.title || s.locator || s.source_kind} (source #{s.id}, {s.document_count} doc
                        {s.document_count === 1 ? '' : 's'}, {USAGE_LABEL[s.usage] ?? s.usage})
                    </span>
                </label>
            ))}
        </fieldset>
        {onFiltersChange && (
            <fieldset className="code-findings-form__group" data-testid="seed-filters">
                <legend>Seed filters</legend>
                <p className="u-hint u-mt0">
                    Applied before the 2,000-passage cap. Globs match the passage&apos;s repo-relative path (case-sensitive;
                    <code>*</code> crosses folders, so <code>src/**</code> and <code>**/*.test.*</code> work). One per line.
                </p>
                <label className="seed-filters__field">
                    <span>Only paths matching</span>
                    <textarea
                        rows={2}
                        value={include}
                        onChange={(e) => setInclude(e.target.value)}
                        placeholder="src/**"
                        data-testid="seed-path-include"
                    />
                </label>
                <label className="seed-filters__field">
                    <span>Skip paths matching</span>
                    <textarea
                        rows={2}
                        value={exclude}
                        onChange={(e) => setExclude(e.target.value)}
                        placeholder="**/*.lock"
                        data-testid="seed-path-exclude"
                    />
                </label>
                <label className="seed-filters__field">
                    <span>Keep only passages mentioning (any, case-insensitive; path or text)</span>
                    <textarea
                        rows={2}
                        value={terms}
                        onChange={(e) => setTerms(e.target.value)}
                        placeholder="one term per line"
                        data-testid="seed-relevance-terms"
                    />
                </label>
                {hasTerms && (
                    <>
                        <label className="seed-sources__option">
                            <input
                                type="checkbox"
                                checked={scope === 'file'}
                                onChange={(e) => setScope(e.target.checked ? 'file' : 'chunk')}
                                data-testid="seed-relevance-file-scope"
                            />
                            <span>Keep every passage of a file when any passage of it mentions a term</span>
                        </label>
                        <label className="seed-sources__option">
                            <span>Match terms in</span>
                            <select
                                value={match}
                                onChange={(e) => setMatch(e.target.value === 'text' ? 'text' : 'path_or_text')}
                                data-testid="seed-relevance-match"
                            >
                                <option value="path_or_text">Path or text</option>
                                <option value="text">Text only</option>
                            </select>
                        </label>
                        <p className="u-hint">Skip the term filter for these sources (path globs still apply):</p>
                        {sources.map((s) => (
                            <label key={s.id} className="seed-sources__option">
                                <input
                                    type="checkbox"
                                    checked={exempt.includes(s.id)}
                                    onChange={() => toggleExempt(s.id)}
                                    data-testid={`seed-exempt-${s.id}`}
                                />
                                <span>{s.title || s.locator || s.source_kind} (source #{s.id})</span>
                            </label>
                        ))}
                    </>
                )}
            </fieldset>
        )}
        </>
    );
}
