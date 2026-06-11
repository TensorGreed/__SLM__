/**
 * TrainingConfigPatchPreviewModal — Coach-stage-2 phase 2.
 *
 * Preview-then-apply for a single training-config patch. Parent panel
 * (TrainingConfigGapsPanel) opens this with the signal id; the modal:
 *
 *   1. POSTs to /training-config-gaps/patch/preview for the before →
 *      after diff.
 *   2. Renders the field-level diff in a small "<before> → <after>"
 *      table (3 rows max — eval_steps / num_epochs / warmup_ratio
 *      are the only patches today).
 *   3. On Apply, POSTs to /patch/apply, surfaces the result via
 *      onApplied so the parent can refresh the gap report and dispatch
 *      a DOM event the TrainingPanel listens for.
 *
 * No safety-gating logic in phase 2 — every patch is a numeric field
 * write, allow-listed at the service layer. `safe_to_apply` is always
 * true; we keep the flag in the contract for forward-compatibility
 * with phase 3 (which may introduce patches requiring confirmation).
 */

import { useCallback, useEffect, useState } from 'react';

import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import ErrorPanel from '../shared/ErrorPanel';
import {
    applyPatch,
    previewPatch,
    type TrainingConfigPatchPreview,
    type TrainingConfigPatchResult,
} from '../../api/trainingConfigGaps';
import './TrainingConfigPatchPreviewModal.css';

interface TrainingConfigPatchPreviewModalProps {
    projectId: number;
    signalId: string;
    onClose: () => void;
    onApplied: (result: TrainingConfigPatchResult) => void;
}

// Format the diff cell. Most patches today are integer steps / epoch
// counts; warmup_ratio is a fraction. Detect and format appropriately
// so eval_steps reads "100 → 10" not "100.0 → 10.0".
function formatValue(value: number): string {
    if (Number.isInteger(value)) return String(value);
    return value < 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
}

export default function TrainingConfigPatchPreviewModal({
    projectId,
    signalId,
    onClose,
    onApplied,
}: TrainingConfigPatchPreviewModalProps) {
    const [preview, setPreview] = useState<TrainingConfigPatchPreview | null>(
        null,
    );
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<ErrorEnvelope | null>(null);
    const [applying, setApplying] = useState(false);

    useEffect(() => {
        let cancelled = false;
        async function load() {
            setLoading(true);
            setError(null);
            try {
                const res = await previewPatch(projectId, signalId);
                if (!cancelled) setPreview(res);
            } catch (err) {
                if (!cancelled) setError(parseErrorEnvelope(err));
            } finally {
                if (!cancelled) setLoading(false);
            }
        }
        void load();
        return () => {
            cancelled = true;
        };
    }, [projectId, signalId]);

    const handleApply = useCallback(async () => {
        if (!preview || !preview.safe_to_apply) return;
        setApplying(true);
        setError(null);
        try {
            const res = await applyPatch(projectId, signalId);
            onApplied(res);
            onClose();
        } catch (err) {
            setError(parseErrorEnvelope(err));
        } finally {
            setApplying(false);
        }
    }, [preview, projectId, signalId, onApplied, onClose]);

    const patchedFields = preview ? Object.keys(preview.patch) : [];

    return (
        <div
            className="tcg-patch-modal-backdrop"
            data-testid="training-config-patch-modal"
            onClick={onClose}
            role="presentation"
        >
            <div
                className="tcg-patch-modal"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-label="Apply training-config patch"
            >
                <header className="tcg-patch-modal__head">
                    <h3 className="tcg-patch-modal__title">
                        {preview?.patch_label ?? 'Preview patch'}
                    </h3>
                    <button
                        type="button"
                        className="tcg-patch-modal__close"
                        onClick={onClose}
                        aria-label="Close preview"
                    >
                        ×
                    </button>
                </header>

                {loading && (
                    <div
                        className="tcg-patch-modal__loading"
                        data-testid="training-config-patch-loading"
                    >
                        Loading preview…
                    </div>
                )}

                {error && (
                    <ErrorPanel
                        envelope={error}
                        onDismiss={() => setError(null)}
                        testIdPrefix="training-config-patch-error"
                    />
                )}

                {preview && (
                    <>
                        {preview.plain_english && (
                            <p
                                className="tcg-patch-modal__plain"
                                data-testid="training-config-patch-plain"
                            >
                                {preview.plain_english}
                            </p>
                        )}
                        <table className="tcg-patch-modal__diff">
                            <thead>
                                <tr>
                                    <th>Field</th>
                                    <th>Current</th>
                                    <th aria-hidden="true">→</th>
                                    <th>After</th>
                                </tr>
                            </thead>
                            <tbody>
                                {patchedFields.map((field) => (
                                    <tr
                                        key={field}
                                        data-testid={`training-config-patch-row-${field}`}
                                    >
                                        <td>
                                            <code>{field}</code>
                                        </td>
                                        <td>{formatValue(preview.before[field])}</td>
                                        <td aria-hidden="true">→</td>
                                        <td>
                                            <strong>
                                                {formatValue(preview.after[field])}
                                            </strong>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </>
                )}

                <footer className="tcg-patch-modal__foot">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onClose}
                        data-testid="training-config-patch-cancel"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void handleApply()}
                        disabled={
                            !preview || !preview.safe_to_apply || applying
                        }
                        data-testid="training-config-patch-apply"
                    >
                        {applying ? 'Applying…' : 'Apply'}
                    </button>
                </footer>
            </div>
        </div>
    );
}
