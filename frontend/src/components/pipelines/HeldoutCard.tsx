/**
 * HeldoutCard — upload the pipeline's held-out rows (JSONL). Valid rows are
 * accepted; rejected rows are listed grouped by reason with line numbers.
 */

import { useEffect, useState } from 'react';

import { getHeldout, parsePipelineError, uploadHeldout, type HeldoutUploadResult } from '../../api/pipelines';
import './pipelines.css';

export default function HeldoutCard({ pipelineId, onUploaded }: { pipelineId: string; onUploaded?: () => void }) {
    const [total, setTotal] = useState<number | null>(null);
    const [result, setResult] = useState<HeldoutUploadResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const refresh = () => {
        getHeldout(pipelineId, 50)
            .then((h) => setTotal(h.total ?? h.rows.length))
            .catch(() => setTotal(null));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(refresh, [pipelineId]);

    const onFile = async (file: File | undefined) => {
        if (!file) return;
        setBusy(true);
        setError(null);
        try {
            setResult(await uploadHeldout(pipelineId, file));
            refresh();
            onUploaded?.();
        } catch (e) {
            setError(parsePipelineError(e).message);
        } finally {
            setBusy(false);
        }
    };

    return (
        <section className="pipelines__section" data-testid="pipeline-heldout">
            <h3>Held-out rows</h3>
            <p className="pipelines__note">
                JSONL, one row per line: <code>{'{"id", "input": {...}, "gold": {"<stage name>": ...}}'}</code>.
                Gold may cover any subset of stages.{total !== null && ` ${total} row${total === 1 ? '' : 's'} stored.`}
            </p>
            <input type="file" accept=".jsonl,.json,.txt" disabled={busy} data-testid="heldout-file"
                aria-label="Held-out JSONL file" onChange={(e) => void onFile(e.target.files?.[0])} />
            {error && <p className="pipelines__error" role="alert">{error}</p>}
            {result && (
                <div data-testid="heldout-result">
                    <p><strong>{result.accepted}</strong> row{result.accepted === 1 ? '' : 's'} accepted.</p>
                    {Object.entries(result.rejected_by_reason).map(([reason, rows]) => (
                        <details key={reason} open data-testid={`heldout-rejected-${reason}`}>
                            <summary>{rows.length} rejected: {reason}</summary>
                            <ul className="pipelines__list">
                                {rows.map((r, i) => <li key={i}>line {r.line}{r.id ? ` · id ${r.id}` : ''}</li>)}
                            </ul>
                        </details>
                    ))}
                </div>
            )}
        </section>
    );
}
