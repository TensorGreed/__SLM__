/**
 * SeedSourcesSelect — multi-select of the project's data sources for the
 * playbooks that draw seeds from documents (code chunks / cleaned passages).
 * Emits the chosen ids (sent as `params.source_ids`) or null for "all sources".
 * Sources marked evaluation-only or retrieval-only stay listed but are never
 * used for training data (the licence gate applies on top of the choice).
 */

import { useEffect, useState } from 'react';

import { listProvenanceSources, USAGE_LABEL, type ProvenanceSource } from '../../api/provenance';

interface Props {
    projectId: number;
    onChange: (ids: number[] | null) => void;
}

export default function SeedSourcesSelect({ projectId, onChange }: Props) {
    const [sources, setSources] = useState<ProvenanceSource[]>([]);
    const [picked, setPicked] = useState<number[]>([]);

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

    if (!sources.length) return null;

    const toggle = (id: number) =>
        setPicked((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(0, 100)));

    return (
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
    );
}
