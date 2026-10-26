/**
 * CodeFindingsParamsForm — the seeded-defect knobs of the `code-findings`
 * synth playbook: which taxonomy label to seed, optional attrs (prefilled
 * from the taxonomy entry), and an optional data-carrying context (a
 * dataset column whose field names/values the defect should involve).
 * Emits the `params` dict the run-playbook endpoint takes.
 */

import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import { getTaxonomy, type TaxonomyEntry } from '../../api/labelTaxonomy';
import type { PlaybookParams } from '../../api/synthPlaybook';

interface DatasetOption {
    id: number;
    name: string;
    dataset_type?: string;
    record_count?: number;
}

/** The answer key is held out for evaluation: never a context source. */
const ANSWER_KEY_TYPES = new Set(['gold_dev', 'gold_test']);

interface AttrRow { key: string; value: string }

interface Props {
    projectId: number;
    onChange: (params: PlaybookParams | null) => void;
}

export function attrRowsFromEntry(entry: TaxonomyEntry | undefined): AttrRow[] {
    return Object.entries(entry?.attrs ?? {}).map(([key, value]) => ({
        key,
        value: typeof value === 'string' ? value : JSON.stringify(value),
    }));
}

export default function CodeFindingsParamsForm({ projectId, onChange }: Props) {
    const [entries, setEntries] = useState<TaxonomyEntry[]>([]);
    const [taxonomyVersion, setTaxonomyVersion] = useState<string | null>(null);
    const [loaded, setLoaded] = useState(false);
    const [labelId, setLabelId] = useState('');
    const [attrs, setAttrs] = useState<AttrRow[]>([]);
    const [datasets, setDatasets] = useState<DatasetOption[]>([]);
    const [datasetId, setDatasetId] = useState('');
    const [column, setColumn] = useState('');
    const [columns, setColumns] = useState<string[]>([]);
    const [sampleSize, setSampleSize] = useState(20);

    useEffect(() => {
        let cancelled = false;
        getTaxonomy(projectId)
            .then((s) => {
                if (cancelled) return;
                setEntries(s?.taxonomy?.entries ?? []);
                setTaxonomyVersion(s?.version ?? null);
            })
            .catch(() => { /* no taxonomy: the empty state below explains */ })
            .finally(() => { if (!cancelled) setLoaded(true); });
        api.get<{ datasets: DatasetOption[] }>(`/projects/${projectId}/datasets`)
            .then((res) => {
                if (cancelled) return;
                const all = res?.data?.datasets ?? [];
                setDatasets(all.filter((d) => !ANSWER_KEY_TYPES.has(d.dataset_type ?? '')));
            })
            .catch(() => { /* the context is optional */ });
        return () => { cancelled = true; };
    }, [projectId]);

    useEffect(() => {
        setColumns([]);
        if (!datasetId) return;
        let cancelled = false;
        api.get<{ columns: string[] }>(`/projects/${projectId}/datasets/${datasetId}/columns`)
            .then((res) => { if (!cancelled) setColumns(res?.data?.columns ?? []); })
            .catch(() => { /* the picker stays empty: the dataset has no readable rows */ });
        return () => { cancelled = true; };
    }, [projectId, datasetId]);

    const dataset = useMemo(() => datasets.find((d) => String(d.id) === datasetId), [datasets, datasetId]);

    const pickLabel = (id: string) => {
        setLabelId(id);
        setAttrs(attrRowsFromEntry(entries.find((e) => e.id === id)));
    };

    useEffect(() => {
        if (!labelId) { onChange(null); return; }
        const params: PlaybookParams = { label_id: labelId };
        const cleaned = attrs.filter((a) => a.key.trim());
        if (cleaned.length > 0) {
            params.attrs = Object.fromEntries(cleaned.map((a) => [a.key.trim(), a.value]));
        }
        if (datasetId && column.trim()) {
            params.context_dataset_id = Number(datasetId);
            params.context_column = column.trim();
            params.context_sample_size = sampleSize;
        }
        onChange(params);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [labelId, attrs, datasetId, column, sampleSize]);

    if (loaded && entries.length === 0) {
        return (
            <div className="code-findings-form" data-testid="code-findings-form">
                <p className="u-hint" data-testid="code-findings-no-taxonomy">
                    This project has no label taxonomy, so type the label to seed. Add a taxonomy under Data to pick from its entries.
                </p>
                <div className="playbook-picker__count">
                    <label htmlFor="cf-free-label">Label to seed</label>
                    <input
                        id="cf-free-label"
                        value={labelId}
                        onChange={(e) => setLabelId(e.target.value.trim())}
                        data-testid="cf-free-label"
                    />
                </div>
            </div>
        );
    }

    return (
        <div className="code-findings-form" data-testid="code-findings-form">
            <div className="playbook-picker__count">
                <label htmlFor="cf-label">Label to seed</label>
                <select id="cf-label" value={labelId} onChange={(e) => pickLabel(e.target.value)} data-testid="cf-label">
                    <option value="">Pick a label…</option>
                    {entries.map((e) => (
                        <option key={e.id} value={e.id}>{e.title} ({e.id})</option>
                    ))}
                </select>
                {taxonomyVersion && <span className="u-hint">taxonomy {taxonomyVersion}</span>}
            </div>

            {labelId && (
                <fieldset className="code-findings-form__group" data-testid="cf-attrs">
                    <legend>Attributes (optional, prefilled from {labelId})</legend>
                    {attrs.map((row, i) => (
                        <div key={i} className="code-findings-form__attr">
                            <input
                                aria-label={`Attribute ${i + 1} name`}
                                value={row.key}
                                onChange={(e) => setAttrs(attrs.map((a, j) => (j === i ? { ...a, key: e.target.value } : a)))}
                            />
                            <input
                                aria-label={`Attribute ${i + 1} value`}
                                value={row.value}
                                onChange={(e) => setAttrs(attrs.map((a, j) => (j === i ? { ...a, value: e.target.value } : a)))}
                            />
                            <button type="button" className="btn btn-sm" onClick={() => setAttrs(attrs.filter((_, j) => j !== i))}>
                                Remove
                            </button>
                        </div>
                    ))}
                    <button type="button" className="btn btn-sm" data-testid="cf-add-attr" onClick={() => setAttrs([...attrs, { key: '', value: '' }])}>
                        Add attribute
                    </button>
                </fieldset>
            )}

            <fieldset className="code-findings-form__group" data-testid="cf-context">
                <legend>Data-carrying context (optional)</legend>
                <p className="u-hint u-mt0">
                    Sample distinct values from a dataset column so the seeded defect involves those field names and values.
                </p>
                <div className="playbook-picker__count">
                    <label htmlFor="cf-dataset">Dataset</label>
                    <select id="cf-dataset" value={datasetId} onChange={(e) => { setDatasetId(e.target.value); setColumn(''); }} data-testid="cf-dataset">
                        <option value="">None</option>
                        {datasets.map((d) => (
                            <option key={d.id} value={d.id}>{d.name} (dataset #{d.id})</option>
                        ))}
                    </select>
                </div>
                {datasetId && (
                    <div className="playbook-picker__count">
                        <label htmlFor="cf-column">Column</label>
                        <select
                            id="cf-column"
                            value={column}
                            onChange={(e) => setColumn(e.target.value)}
                            data-testid="cf-column"
                        >
                            <option value="">Pick a column…</option>
                            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                        </select>
                        <label htmlFor="cf-sample">Sample size</label>
                        <input
                            id="cf-sample"
                            type="number"
                            min={1}
                            max={200}
                            value={sampleSize}
                            onChange={(e) => setSampleSize(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
                            data-testid="cf-sample"
                        />
                    </div>
                )}
                {dataset && column.trim() && (
                    <p className="u-hint" data-testid="cf-context-provenance">
                        Context: dataset #{dataset.id} &quot;{dataset.name}&quot;, column <code>{column.trim()}</code>, {sampleSize} distinct values
                    </p>
                )}
            </fieldset>
        </div>
    );
}
