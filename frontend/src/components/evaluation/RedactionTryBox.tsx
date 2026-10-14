/**
 * "Try redaction" — paste a text, run it through a span-tagging run
 * (POST /projects/{id}/redact) and see it masked plus every span with its
 * score. The recall-bias slider lowers the "O" probability so borderline
 * tokens are tagged (high-recall mode); the run's thresholds.json applies
 * otherwise.
 */

import { useState } from 'react';

import { redactText, type RedactMode, type RedactResponse } from '../../api/redact';
import './RedactionTryBox.css';

interface RedactionTryBoxProps {
    projectId: number;
    /** The run to use; null → the project's newest span-tagging run. */
    experimentId: number | null;
}

function errorText(err: unknown): string {
    const data = (err as { response?: { data?: { detail?: unknown; message?: unknown } } })?.response?.data;
    const detail = data?.detail ?? data?.message;
    return typeof detail === 'string' ? detail : 'Redaction failed.';
}

export default function RedactionTryBox({ projectId, experimentId }: RedactionTryBoxProps) {
    const [text, setText] = useState('');
    const [mode, setMode] = useState<RedactMode>('mask');
    const [recallBias, setRecallBias] = useState(0);
    const [result, setResult] = useState<RedactResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const run = async () => {
        if (!text.trim()) return;
        setBusy(true);
        setError(null);
        try {
            setResult(await redactText(projectId, {
                text,
                experiment_id: experimentId,
                mode,
                recall_bias: recallBias > 0 ? recallBias : null,
            }));
        } catch (err) {
            setResult(null);
            setError(errorText(err));
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="redaction-try" data-testid="redaction-try">
            <h4 className="redaction-try__title">Try redaction</h4>
            <textarea
                className="redaction-try__input"
                aria-label="Text to redact"
                placeholder="Paste a text with names, phone numbers, account numbers…"
                rows={4}
                value={text}
                onChange={(e) => setText(e.target.value)}
            />
            <div className="redaction-try__controls">
                <label>
                    Output{' '}
                    <select aria-label="Output mode" value={mode} onChange={(e) => setMode(e.target.value as RedactMode)}>
                        <option value="mask">Mask ([TYPE])</option>
                        <option value="tag">Tag (&lt;type&gt;…&lt;/type&gt;)</option>
                        <option value="spans">Spans only</option>
                    </select>
                </label>
                <label className="redaction-try__bias">
                    Recall bias {recallBias.toFixed(2)}
                    <input
                        type="range"
                        aria-label="Recall bias"
                        min={0}
                        max={0.5}
                        step={0.05}
                        value={recallBias}
                        onChange={(e) => setRecallBias(Number(e.target.value))}
                    />
                </label>
                <button type="button" className="btn btn-secondary" disabled={busy || !text.trim()} onClick={run}>
                    {busy ? 'Redacting…' : 'Redact'}
                </button>
            </div>
            <p className="redaction-try__hint">
                Higher recall bias tags borderline tokens too: fewer misses, more false alarms.
            </p>
            {error && <p className="redaction-try__error" role="alert">{error}</p>}
            {result && (
                <div className="redaction-try__result" data-testid="redaction-try-result">
                    <p className="redaction-try__run">
                        Run #{result.experiment_id}{result.experiment_name ? ` (${result.experiment_name})` : ''}
                        {' · '}window {result.settings.max_length} tokens
                        {result.settings.thresholds_file ? ' · thresholds.json applied' : ''}
                    </p>
                    {result.mode !== 'spans' && (
                        <pre className="redaction-try__output" data-testid="redaction-try-output">{result.text}</pre>
                    )}
                    {result.spans.length === 0 ? (
                        <p className="redaction-try__hint">No spans found.</p>
                    ) : (
                        <table className="redaction-try__spans">
                            <thead>
                                <tr><th>Type</th><th>Text</th><th>Score</th><th>Weakest token</th><th>Offsets</th></tr>
                            </thead>
                            <tbody>
                                {result.spans.map((span) => (
                                    <tr key={`${span.start}-${span.end}-${span.type}`}>
                                        <td>{span.type}</td>
                                        <td>{span.text}</td>
                                        <td>{span.score.toFixed(2)}</td>
                                        <td>{span.min_score.toFixed(2)}</td>
                                        <td>{span.start}–{span.end}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            )}
        </div>
    );
}
