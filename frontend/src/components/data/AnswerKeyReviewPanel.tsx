/**
 * AnswerKeyReviewPanel — a person checks the answer key (P6 Trust).
 *
 * The key is written by a model, so a score against it inherits that
 * model's mistakes. This panel lists the key's rows unreviewed-first and,
 * within those, the rows the project's trained model disagrees with most
 * (one of the two is wrong — the reviewer decides which). Each row shows
 * the passage with the key's spans highlighted, marks spans the model
 * missed or added, and takes a verdict: Accept, Fix (edit the spans
 * inline — select text in the passage to add one) or Reject (the row
 * leaves every eval). The Eval tab then scores the checked rows apart
 * from the unchecked ones.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
    agreementText,
    fetchReviewQueue,
    reviewProgressText,
    setReviewVerdict,
    type ReviewQueue,
    type ReviewQueueRow,
    type ReviewSpan,
    type ReviewVerdict,
} from '../../api/goldReview';
import { parseErrorEnvelope } from '../../api/errors';
import './AnswerKeyReviewPanel.css';

interface Props {
    projectId: number;
    datasetType?: 'gold_test' | 'gold_dev';
    /** Called after a verdict lands so the entries list can refetch. */
    onChanged?: () => void;
}

interface Segment {
    start: number;
    end: number;
    key: ReviewSpan | null;
    model: ReviewSpan | null;
}

/** Cut ``text`` at every key / model span boundary so each piece can be
 *  coloured by who tagged it. Key spans win where the two overlap. */
export function segmentText(text: string, keySpans: ReviewSpan[], modelSpans: ReviewSpan[]): Segment[] {
    const cuts = new Set<number>([0, text.length]);
    for (const s of [...keySpans, ...modelSpans]) {
        cuts.add(Math.max(0, Math.min(text.length, s.start)));
        cuts.add(Math.max(0, Math.min(text.length, s.end)));
    }
    const points = [...cuts].sort((a, b) => a - b);
    const out: Segment[] = [];
    for (let i = 0; i + 1 < points.length; i += 1) {
        const start = points[i];
        const end = points[i + 1];
        if (end <= start) continue;
        const key = keySpans.find((s) => s.start <= start && s.end >= end) ?? null;
        const model = modelSpans.find((s) => s.start <= start && s.end >= end) ?? null;
        out.push({ start, end, key, model });
    }
    return out;
}

function sameSpan(a: ReviewSpan | null, b: ReviewSpan | null): boolean {
    return !!a && !!b && a.type === b.type && a.start === b.start && a.end === b.end;
}

function bucket(fraction: number): string {
    const pct = Math.max(0, Math.min(100, Math.round(fraction * 10) * 10));
    return `akr__bar-fill--w-${pct}`;
}

function PassageView({ row, spans, onSelect }: { row: ReviewQueueRow; spans: ReviewSpan[]; onSelect?: (start: number, end: number) => void }) {
    const segments = useMemo(() => segmentText(row.text, spans, row.predicted ?? []), [row.text, spans, row.predicted]);
    const handleMouseUp = () => {
        if (!onSelect) return;
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return;
        const range = sel.getRangeAt(0);
        const startEl = (range.startContainer.nodeType === Node.TEXT_NODE ? range.startContainer.parentElement : (range.startContainer as Element))?.closest('[data-start]') as HTMLElement | null;
        const endEl = (range.endContainer.nodeType === Node.TEXT_NODE ? range.endContainer.parentElement : (range.endContainer as Element))?.closest('[data-start]') as HTMLElement | null;
        if (!startEl || !endEl) return;
        const start = Number(startEl.dataset.start) + range.startOffset;
        const end = Number(endEl.dataset.start) + range.endOffset;
        sel.removeAllRanges();
        if (end > start) onSelect(start, end);
    };
    return (
        <div className="akr__text" data-testid={`akr-text-${row.index}`} onMouseUp={handleMouseUp}>
            {segments.map((seg) => {
                const piece = row.text.slice(seg.start, seg.end);
                if (!seg.key && !seg.model) {
                    return <span key={seg.start} data-start={seg.start}>{piece}</span>;
                }
                const endsKey = seg.key && seg.key.end === seg.end;
                const endsModel = !seg.key && seg.model && seg.model.end === seg.end;
                const modelAgrees = seg.key ? (row.predicted ?? []).some((m) => sameSpan(m, seg.key)) : false;
                const cls = seg.key
                    ? `akr__span${row.predicted && !modelAgrees ? ' akr__span--missed' : ''}`
                    : 'akr__span--extra';
                return (
                    <span key={seg.start} className={cls} data-start={seg.start} title={seg.key ? `key: ${seg.key.type}` : `model only: ${seg.model?.type}`}>
                        {piece}
                        {endsKey && <span className="akr__span-type">{seg.key!.type}</span>}
                        {endsModel && <span className="akr__span-type">{seg.model!.type}?</span>}
                    </span>
                );
            })}
        </div>
    );
}

export default function AnswerKeyReviewPanel({ projectId, datasetType = 'gold_test', onChanged }: Props) {
    const [queue, setQueue] = useState<ReviewQueue | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showReviewed, setShowReviewed] = useState(false);
    const [notes, setNotes] = useState<Record<string, string>>({});
    const [editing, setEditing] = useState<{ key: string; spans: ReviewSpan[] } | null>(null);
    const [saving, setSaving] = useState<string | null>(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            setQueue(await fetchReviewQueue(projectId, { datasetType }));
        } catch (err) {
            setError(parseErrorEnvelope(err).message || 'Could not load the answer key.');
        } finally {
            setLoading(false);
        }
    }, [projectId, datasetType]);

    useEffect(() => { void load(); }, [load]);

    const knownTypes = useMemo(() => {
        const seen = new Set<string>();
        for (const row of queue?.rows ?? []) {
            for (const s of [...(row.entities ?? []), ...(row.predicted ?? [])]) seen.add(s.type);
        }
        return [...seen].sort();
    }, [queue]);

    const apply = async (row: ReviewQueueRow, verdict: ReviewVerdict | null, corrected?: { entities: ReviewSpan[] }) => {
        setSaving(row.key);
        setError(null);
        try {
            const res = await setReviewVerdict(projectId, {
                dataset_type: datasetType, index: row.index, key: row.key, verdict, corrected, note: notes[row.key] ?? '',
            });
            setQueue((prev) => prev ? {
                ...prev,
                total: res.total, reviewed: res.reviewed, accepted: res.accepted, corrected: res.corrected,
                rejected: res.rejected, unreviewed: res.unreviewed, last_reviewed_at: res.last_reviewed_at,
                rows: prev.rows.map((r) => (r.key === row.key ? { ...res.row, predicted: res.row.predicted ?? r.predicted, agreement: res.row.agreement ?? r.agreement } : r)),
            } : prev);
            setEditing(null);
            onChanged?.();
        } catch (err) {
            setError(parseErrorEnvelope(err).message || 'Could not save the verdict.');
        } finally {
            setSaving(null);
        }
    };

    if (loading && !queue) return <div className="akr" data-testid="akr-loading">Loading the answer key…</div>;
    if (!queue || queue.total === 0) return null;

    const pending = queue.rows.filter((r) => !r.review);
    const done = queue.rows.filter((r) => r.review);
    const visible = showReviewed ? queue.rows : pending;

    return (
        <section className="akr" data-testid="answer-key-review" id="answer-key-review">
            <div className="akr__head">
                <h4>Check the answer key yourself</h4>
                <span className="akr__progress" data-testid="akr-progress">{reviewProgressText(queue)}</span>
            </div>
            <div className="akr__bar" aria-label={`${Math.round((queue.reviewed / Math.max(1, queue.total)) * 100)}% reviewed`}>
                <div className={`akr__bar-fill ${bucket(queue.reviewed / Math.max(1, queue.total))}`} />
            </div>
            <p className="akr__hint">
                {queue.model_experiment_id != null
                    ? <>Rows where run #{queue.model_experiment_id} disagrees with the key come first: one of the two is wrong. Green = in the key, orange = the model missed it, dashed = the model found it and the key doesn't have it.</>
                    : <>No trained span run to compare with yet — rows are in file order.</>}
                {' '}Accept what is right, fix what is wrong (select text in the passage to add a span), reject rows that are not a fair test. The Eval tab scores checked rows apart from unchecked ones.
            </p>
            {error && <p className="akr__error" data-testid="akr-error">{error}</p>}
            {pending.length === 0 && <p className="akr__hint" data-testid="akr-all-done">Every row has been checked.</p>}
            {visible.map((row) => {
                const isEditing = editing?.key === row.key;
                const spans = isEditing ? editing!.spans : (row.entities ?? []);
                const busy = saving === row.key;
                return (
                    <div key={row.key} className={`akr__row${row.review ? ' akr__row--reviewed' : ''}`} data-testid={`akr-row-${row.index}`}>
                        <div className="akr__row-head">
                            <span>Row {row.index + 1}{row.source ? ` · ${row.source.replace(/^answer_key:/, '')}` : ''}</span>
                            <span className={row.review ? `akr__verdict akr__verdict--${row.review.verdict}` : ''} data-testid={`akr-status-${row.index}`}>
                                {agreementText(row)}
                            </span>
                        </div>
                        {row.entities !== null ? (
                            <PassageView
                                row={row}
                                spans={spans}
                                onSelect={isEditing ? (start, end) => setEditing((e) => e && ({ ...e, spans: [...e.spans, { type: knownTypes[0] ?? 'entity', start, end }].sort((a, b) => a.start - b.start) })) : undefined}
                            />
                        ) : (
                            <div className="akr__text">{row.text}{Object.entries(row.fields).map(([k, v]) => <div key={k}><strong>{k}:</strong> {v}</div>)}</div>
                        )}
                        {isEditing && (
                            <div className="akr__edit" data-testid={`akr-edit-${row.index}`}>
                                {editing!.spans.map((s, i) => (
                                    <div key={`${s.start}-${s.end}-${i}`} className="akr__edit-row">
                                        <input
                                            className="input"
                                            list="akr-types"
                                            value={s.type}
                                            aria-label="Span type"
                                            onChange={(e) => setEditing((ed) => ed && ({ ...ed, spans: ed.spans.map((x, j) => (j === i ? { ...x, type: e.target.value } : x)) }))}
                                        />
                                        <input
                                            className="input"
                                            type="number"
                                            value={s.start}
                                            aria-label="Span start"
                                            onChange={(e) => setEditing((ed) => ed && ({ ...ed, spans: ed.spans.map((x, j) => (j === i ? { ...x, start: Number(e.target.value) } : x)) }))}
                                        />
                                        <input
                                            className="input"
                                            type="number"
                                            value={s.end}
                                            aria-label="Span end"
                                            onChange={(e) => setEditing((ed) => ed && ({ ...ed, spans: ed.spans.map((x, j) => (j === i ? { ...x, end: Number(e.target.value) } : x)) }))}
                                        />
                                        <code className="akr__edit-text">{row.text.slice(s.start, s.end)}</code>
                                        <button type="button" className="btn btn-link" onClick={() => setEditing((ed) => ed && ({ ...ed, spans: ed.spans.filter((_, j) => j !== i) }))} aria-label="Remove span">✕</button>
                                    </div>
                                ))}
                                <datalist id="akr-types">{knownTypes.map((t) => <option key={t} value={t} />)}</datalist>
                                <div className="akr__actions">
                                    {row.predicted && (
                                        <button type="button" className="btn btn-secondary" onClick={() => setEditing((ed) => ed && ({ ...ed, spans: row.predicted ?? [] }))} data-testid={`akr-use-model-${row.index}`}>
                                            Use the model's spans
                                        </button>
                                    )}
                                    <button type="button" className="btn btn-secondary" onClick={() => setEditing((ed) => ed && ({ ...ed, spans: [...ed.spans, { type: knownTypes[0] ?? 'entity', start: 0, end: Math.min(1, row.text.length) }] }))}>
                                        + span
                                    </button>
                                    <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void apply(row, 'corrected', { entities: editing!.spans })} data-testid={`akr-save-${row.index}`}>
                                        Save correction
                                    </button>
                                    <button type="button" className="btn btn-link" onClick={() => setEditing(null)}>Cancel</button>
                                </div>
                            </div>
                        )}
                        {!isEditing && (
                            <div className="akr__actions">
                                {row.review ? (
                                    <>
                                        {row.review.note && <span className="akr__legend">note: {row.review.note}</span>}
                                        <button type="button" className="btn btn-link" disabled={busy} onClick={() => void apply(row, null)} data-testid={`akr-clear-${row.index}`}>
                                            Clear verdict
                                        </button>
                                    </>
                                ) : (
                                    <>
                                        <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void apply(row, 'accepted')} data-testid={`akr-accept-${row.index}`}>Accept</button>
                                        {row.entities !== null && (
                                            <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => setEditing({ key: row.key, spans: row.entities ?? [] })} data-testid={`akr-fix-${row.index}`}>Fix</button>
                                        )}
                                        <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => void apply(row, 'rejected')} data-testid={`akr-reject-${row.index}`}>Reject</button>
                                        <input
                                            className="input akr__note"
                                            placeholder="note (optional)"
                                            value={notes[row.key] ?? ''}
                                            aria-label="Review note"
                                            onChange={(e) => setNotes((n) => ({ ...n, [row.key]: e.target.value }))}
                                        />
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                );
            })}
            {done.length > 0 && (
                <button type="button" className="btn btn-link" onClick={() => setShowReviewed((v) => !v)} data-testid="akr-toggle-reviewed">
                    {showReviewed ? 'Hide reviewed rows' : `Show ${done.length} reviewed row${done.length === 1 ? '' : 's'}`}
                </button>
            )}
        </section>
    );
}
