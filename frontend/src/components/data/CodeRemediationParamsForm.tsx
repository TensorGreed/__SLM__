/**
 * CodeRemediationParamsForm — the knobs of the two `code-remediation` synth
 * playbooks.
 *
 *  - `remediation_from_findings`: an optional findings dataset of THIS project
 *    (findings rows or fix pairs; answer-key datasets are never offered) and
 *    how many reference passages to attach per seed.
 *  - `code_examples`: how many tasks the teacher writes per code chunk.
 *
 * Emits the `params` dict the run-playbook endpoint takes.
 */

import { useEffect, useState } from 'react';

import api from '../../api/client';
import type { PlaybookParams, SynthMode } from '../../api/synthPlaybook';

interface DatasetOption {
    id: number;
    name: string;
    dataset_type?: string;
    record_count?: number;
}

/** The answer key is held out for evaluation: never a seed source. */
const ANSWER_KEY_TYPES = new Set(['gold_dev', 'gold_test']);

interface Props {
    projectId: number;
    mode: SynthMode;
    onChange: (params: PlaybookParams | null) => void;
}

export default function CodeRemediationParamsForm({ projectId, mode, onChange }: Props) {
    const [datasets, setDatasets] = useState<DatasetOption[]>([]);
    const [datasetId, setDatasetId] = useState('');
    const [topK, setTopK] = useState(3);
    const [perChunk, setPerChunk] = useState(2);

    useEffect(() => {
        if (mode !== 'remediation_from_findings') return;
        let cancelled = false;
        api.get<{ datasets: DatasetOption[] }>(`/projects/${projectId}/datasets`)
            .then((res) => {
                if (cancelled) return;
                const all = res?.data?.datasets ?? [];
                setDatasets(all.filter((d) => !ANSWER_KEY_TYPES.has(d.dataset_type ?? '')));
            })
            .catch(() => { /* the dataset is optional: dev answer-key rows seed the run */ });
        return () => { cancelled = true; };
    }, [projectId, mode]);

    useEffect(() => {
        if (mode === 'code_examples') {
            onChange({ per_chunk: perChunk });
            return;
        }
        const params: PlaybookParams = { top_k: topK };
        if (datasetId) params.findings_dataset_id = Number(datasetId);
        onChange(params);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [mode, datasetId, topK, perChunk]);

    if (mode === 'code_examples') {
        return (
            <div className="code-findings-form" data-testid="code-examples-form">
                <p className="u-hint u-mt0">
                    Seeds are your code documents&apos; chunks (only those whose licence allows training). The teacher writes
                    tasks and code that cites the chunk; answers that copy the chunk are rejected. These rows train the model
                    but are not part of the remediation score.
                </p>
                <div className="playbook-picker__count">
                    <label htmlFor="ce-per-chunk">Tasks per chunk</label>
                    <input
                        id="ce-per-chunk"
                        type="number"
                        min={1}
                        max={5}
                        value={perChunk}
                        onChange={(e) => setPerChunk(Math.max(1, Math.min(5, Number(e.target.value) || 1)))}
                        data-testid="ce-per-chunk"
                    />
                </div>
            </div>
        );
    }

    const dataset = datasets.find((d) => String(d.id) === datasetId);
    return (
        <div className="code-findings-form" data-testid="code-remediation-form">
            <p className="u-hint u-mt0">
                One remediation is written per reported finding. Seeds come from the dataset below and from the dev answer
                key; the test answer key is never used.
            </p>
            <div className="playbook-picker__count">
                <label htmlFor="cr-dataset">Findings dataset</label>
                <select id="cr-dataset" value={datasetId} onChange={(e) => setDatasetId(e.target.value)} data-testid="cr-dataset">
                    <option value="">None (dev answer key only)</option>
                    {datasets.map((d) => (
                        <option key={d.id} value={d.id}>{d.name} (dataset #{d.id})</option>
                    ))}
                </select>
                <label htmlFor="cr-top-k">Passages per seed</label>
                <input
                    id="cr-top-k"
                    type="number"
                    min={1}
                    max={10}
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
                    data-testid="cr-top-k"
                />
            </div>
            {dataset && (
                <p className="u-hint" data-testid="cr-provenance">
                    Seeds from dataset #{dataset.id} &quot;{dataset.name}&quot;
                    {typeof dataset.record_count === 'number' ? ` (${dataset.record_count} rows)` : ''}
                </p>
            )}
        </div>
    );
}
