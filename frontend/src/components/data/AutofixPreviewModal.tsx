/**
 * AutofixPreviewModal — D3.2 + D4 of the data-quality arc.
 *
 * Preview-then-apply contract for every Data Health auto-fix. The
 * parent panel opens this modal with a fix_kind; the modal:
 *
 *   1. POSTs to /autofix/preview to fetch the per-item diff.
 *   2. Renders the items in a kind-specific layout (filenames for
 *      drops, keep-vs-drop pairs for dedup, label merge map for
 *      canonicalisation, finding counts for PII).
 *   3. Disables Apply when ``safe_to_apply`` is false — e.g. PII on
 *      a span-extraction recipe shows the explanation but the
 *      Apply button is greyed out so the user can read but not act.
 *   4. On Apply, POSTs to /autofix and surfaces the result via the
 *      ``onApplied`` callback so the parent can refresh the report
 *      + show the post-fix toast.
 *
 * The user's safety constraint that drove this:
 *   *"I do not want us to redacting data or de-duplicting without
 *   review"* — every destructive transform must show its items
 *   before commit.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import ErrorPanel from '../shared/ErrorPanel';
import './AutofixPreviewModal.css';

interface PreviewItem {
    kind: string;
    [key: string]: unknown;
}

interface AutofixPreview {
    fix_kind: string;
    would_apply_count: number;
    summary: string;
    details: Record<string, unknown>;
    items: PreviewItem[];
    safe_to_apply: boolean;
}

interface AutofixResult {
    fix_kind: string;
    applied_count: number;
    summary: string;
    details: Record<string, unknown>;
}

interface AutofixPreviewModalProps {
    projectId: number;
    fixKind: string;
    fixLabel: string;
    onClose: () => void;
    onApplied: (result: AutofixResult) => void;
}

// Per-fix copy for the modal header. Keep these short — the diff
// itself is what does the explaining, not the headline.
const FIX_HEADLINE: Record<string, string> = {
    drop_failed_docs:
        'These documents failed to parse and will be deleted. Their '
        + 'extracted text is empty, so they contribute nothing to '
        + 'training.',
    dedupe_duplicate_docs:
        'These documents share identical text. The lowest-id copy '
        + 'in each group is kept; the rest are deleted.',
    redact_pii:
        'These documents have detected PII that isn\'t yet '
        + 'redacted. Applying will re-clean them with PII masked.',
    canonicalise_labels:
        'These label groups are fragmented by case or whitespace. '
        + 'Applying merges every variant into the most common form.',
};

export default function AutofixPreviewModal({
    projectId,
    fixKind,
    fixLabel,
    onClose,
    onApplied,
}: AutofixPreviewModalProps) {
    const [preview, setPreview] = useState<AutofixPreview | null>(null);
    const [loading, setLoading] = useState(true);
    // Diagnostics Intervention B — both load + apply failures render
    // via the shared <ErrorPanel> so the user sees troubleshooting_id
    // + actionable_fix on the same surface they're acting on.
    const [error, setError] = useState<ErrorEnvelope | null>(null);
    const [applying, setApplying] = useState(false);

    useEffect(() => {
        let cancelled = false;
        async function load() {
            setLoading(true);
            setError(null);
            try {
                const res = await api.post<AutofixPreview>(
                    `/projects/${projectId}/data-health/autofix/preview`,
                    { fix_kind: fixKind },
                );
                if (!cancelled) setPreview(res.data);
            } catch (err) {
                if (!cancelled) {
                    setError(parseErrorEnvelope(err));
                }
            } finally {
                if (!cancelled) setLoading(false);
            }
        }
        void load();
        return () => { cancelled = true; };
    }, [projectId, fixKind]);

    const handleApply = useCallback(async () => {
        if (!preview || !preview.safe_to_apply) return;
        setApplying(true);
        setError(null);
        try {
            const res = await api.post<AutofixResult>(
                `/projects/${projectId}/data-health/autofix`,
                { fix_kind: fixKind },
            );
            onApplied(res.data);
            onClose();
        } catch (err) {
            setError(parseErrorEnvelope(err));
        } finally {
            setApplying(false);
        }
    }, [preview, projectId, fixKind, onApplied, onClose]);

    return (
        <div
            className="autofix-modal-backdrop"
            data-testid="autofix-modal-backdrop"
            onClick={onClose}
        >
            <div
                className="autofix-modal"
                data-testid="autofix-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="autofix-modal-title"
                onClick={(e) => e.stopPropagation()}
            >
                <header className="autofix-modal__head">
                    <h3 className="autofix-modal__title" id="autofix-modal-title">
                        Preview: {fixLabel}
                    </h3>
                    <button
                        type="button"
                        className="autofix-modal__close"
                        onClick={onClose}
                        aria-label="Close preview"
                        data-testid="autofix-modal-close"
                    >
                        ×
                    </button>
                </header>

                {loading && (
                    <div className="autofix-modal__body autofix-modal__body--loading">
                        Loading preview…
                    </div>
                )}

                {!loading && error && (
                    <div className="autofix-modal__body">
                        <ErrorPanel
                            envelope={error}
                            onDismiss={() => setError(null)}
                            testIdPrefix="autofix-modal-error"
                        />
                    </div>
                )}

                {!loading && !error && preview && (
                    <>
                        <p className="autofix-modal__intro">
                            {FIX_HEADLINE[fixKind] || ''}
                        </p>
                        <p
                            className="autofix-modal__summary"
                            data-testid="autofix-modal-summary"
                        >
                            {preview.summary}
                        </p>
                        {!preview.safe_to_apply && (
                            <div
                                className="autofix-modal__blocked"
                                data-testid="autofix-modal-blocked"
                                role="alert"
                            >
                                <strong>Safety guard:</strong> this fix is
                                blocked for the current recipe. The reason is
                                shown above — the Apply button is disabled
                                so the change can’t land by accident.
                            </div>
                        )}
                        <div
                            className="autofix-modal__items"
                            data-testid="autofix-modal-items"
                        >
                            <PreviewItems
                                fixKind={preview.fix_kind}
                                items={preview.items}
                            />
                        </div>
                    </>
                )}

                <footer className="autofix-modal__foot">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onClose}
                        disabled={applying}
                        data-testid="autofix-modal-cancel"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void handleApply()}
                        disabled={
                            loading
                            || !preview
                            || !preview.safe_to_apply
                            || preview.would_apply_count === 0
                            || applying
                        }
                        data-testid="autofix-modal-apply"
                    >
                        {applying
                            ? 'Applying…'
                            : preview && preview.would_apply_count > 0
                                ? `Apply (${preview.would_apply_count})`
                                : 'Apply'}
                    </button>
                </footer>
            </div>
        </div>
    );
}

interface PreviewItemsProps {
    fixKind: string;
    items: PreviewItem[];
}

function PreviewItems({ fixKind, items }: PreviewItemsProps) {
    if (!items || items.length === 0) {
        return (
            <p
                className="autofix-modal__items-empty"
                data-testid="autofix-modal-items-empty"
            >
                Nothing to change.
            </p>
        );
    }

    if (fixKind === 'drop_failed_docs') {
        return (
            <ul className="autofix-modal__list" data-testid="autofix-modal-list">
                {items.map((it) => (
                    <li
                        key={String(it.id ?? it.filename)}
                        className="autofix-modal__row"
                    >
                        <span className="autofix-modal__row-icon" aria-hidden="true">−</span>
                        <span className="autofix-modal__row-name">
                            {String(it.filename ?? '(unnamed)')}
                        </span>
                        {it.error && (
                            <span className="autofix-modal__row-meta">
                                {String(it.error).slice(0, 80)}
                            </span>
                        )}
                    </li>
                ))}
            </ul>
        );
    }

    if (fixKind === 'dedupe_duplicate_docs') {
        return (
            <ul className="autofix-modal__list" data-testid="autofix-modal-list">
                {items.map((it, idx) => {
                    const keep = (it.keep as { id: number; filename: string }) || null;
                    const drops = (it.drop as Array<{ id: number; filename: string }>) || [];
                    return (
                        <li
                            key={String(it.text_hash ?? idx)}
                            className="autofix-modal__group"
                        >
                            <div className="autofix-modal__group-head">
                                Duplicate set #{idx + 1}
                            </div>
                            {keep && (
                                <div className="autofix-modal__row autofix-modal__row--keep">
                                    <span className="autofix-modal__row-icon" aria-hidden="true">✓</span>
                                    <span className="autofix-modal__row-name">{keep.filename}</span>
                                    <span className="autofix-modal__row-meta">keep</span>
                                </div>
                            )}
                            {drops.map((d) => (
                                <div
                                    key={d.id}
                                    className="autofix-modal__row autofix-modal__row--drop"
                                >
                                    <span className="autofix-modal__row-icon" aria-hidden="true">−</span>
                                    <span className="autofix-modal__row-name">{d.filename}</span>
                                    <span className="autofix-modal__row-meta">drop</span>
                                </div>
                            ))}
                        </li>
                    );
                })}
            </ul>
        );
    }

    if (fixKind === 'redact_pii') {
        return (
            <ul className="autofix-modal__list" data-testid="autofix-modal-list">
                {items.map((it) => (
                    <li
                        key={String(it.id ?? it.filename)}
                        className="autofix-modal__row"
                    >
                        <span className="autofix-modal__row-icon" aria-hidden="true">⚠</span>
                        <span className="autofix-modal__row-name">
                            {String(it.filename ?? '(unnamed)')}
                        </span>
                        <span className="autofix-modal__row-meta">
                            {String(it.pii_findings ?? 0)} finding
                            {it.pii_findings === 1 ? '' : 's'}
                        </span>
                    </li>
                ))}
            </ul>
        );
    }

    if (fixKind === 'canonicalise_labels') {
        return (
            <ul className="autofix-modal__list" data-testid="autofix-modal-list">
                {items.map((it, idx) => {
                    const canonical = String(it.canonical ?? '');
                    const canonicalCount = Number(it.canonical_count ?? 0);
                    const mergeIn = (it.merge_in as Array<{ label: string; count: number }>) || [];
                    return (
                        <li
                            key={`${canonical}-${idx}`}
                            className="autofix-modal__group"
                        >
                            <div className="autofix-modal__group-head">
                                Label group #{idx + 1}
                            </div>
                            {mergeIn.map((m) => (
                                <div
                                    key={m.label}
                                    className="autofix-modal__row autofix-modal__row--drop"
                                >
                                    <span className="autofix-modal__row-icon" aria-hidden="true">−</span>
                                    <span className="autofix-modal__row-name">
                                        “{m.label}”
                                    </span>
                                    <span className="autofix-modal__row-meta">
                                        {m.count} row{m.count === 1 ? '' : 's'}
                                    </span>
                                </div>
                            ))}
                            <div className="autofix-modal__row autofix-modal__row--keep">
                                <span className="autofix-modal__row-icon" aria-hidden="true">→</span>
                                <span className="autofix-modal__row-name">
                                    “{canonical}”
                                </span>
                                <span className="autofix-modal__row-meta">
                                    canonical · {canonicalCount} row
                                    {canonicalCount === 1 ? '' : 's'}
                                </span>
                            </div>
                        </li>
                    );
                })}
            </ul>
        );
    }

    return (
        <ul className="autofix-modal__list" data-testid="autofix-modal-list">
            {items.map((it, idx) => (
                <li key={idx} className="autofix-modal__row">
                    <span className="autofix-modal__row-name">
                        {JSON.stringify(it)}
                    </span>
                </li>
            ))}
        </ul>
    );
}
