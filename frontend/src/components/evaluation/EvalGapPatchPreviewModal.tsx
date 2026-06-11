/**
 * EvalGapPatchPreviewModal — Coach-stage-2 phase 5.
 *
 * Preview-then-apply for the two eval-side patches. Each patch has a
 * different diff shape (baseline-promote shows a candidate
 * checkpoint; label-KL rebalance shows per-class trim counts + KL
 * projection), so this modal renders per-kind layouts rather than
 * the simple field-diff table the training-config patch modal uses.
 *
 * After apply, parent panel re-fetches the gap report so the affected
 * signal's severity reflects the persistent change.
 */

import { useCallback, useEffect, useState } from 'react';

import type { ErrorEnvelope } from '../../api/errors';
import { parseErrorEnvelope } from '../../api/errors';
import ErrorPanel from '../shared/ErrorPanel';
import {
    applyEvalPatch,
    previewEvalPatch,
    type BaselinePromotePreview,
    type EvalGapPatchPreview,
    type EvalGapPatchResult,
    type LabelKlRebalancePreview,
} from '../../api/evalGaps';
import '../training/TrainingConfigPatchPreviewModal.css';

interface EvalGapPatchPreviewModalProps {
    projectId: number;
    signalId: string;
    onClose: () => void;
    onApplied: (result: EvalGapPatchResult) => void;
}

function isBaselinePromote(
    preview: EvalGapPatchPreview,
): preview is BaselinePromotePreview {
    return preview.patch_kind === 'regression_baseline_promote_last_green';
}

function isLabelKl(
    preview: EvalGapPatchPreview,
): preview is LabelKlRebalancePreview {
    return preview.patch_kind === 'label_kl_rebalance_eval';
}

export default function EvalGapPatchPreviewModal({
    projectId,
    signalId,
    onClose,
    onApplied,
}: EvalGapPatchPreviewModalProps) {
    const [preview, setPreview] = useState<EvalGapPatchPreview | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<ErrorEnvelope | null>(null);
    const [applying, setApplying] = useState(false);

    useEffect(() => {
        let cancelled = false;
        async function load() {
            setLoading(true);
            setError(null);
            try {
                const res = await previewEvalPatch(projectId, signalId);
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
            const res = await applyEvalPatch(projectId, signalId);
            onApplied(res);
            onClose();
        } catch (err) {
            setError(parseErrorEnvelope(err));
        } finally {
            setApplying(false);
        }
    }, [preview, projectId, signalId, onApplied, onClose]);

    return (
        <div
            className="tcg-patch-modal-backdrop"
            data-testid="eval-patch-modal"
            onClick={onClose}
            role="presentation"
        >
            <div
                className="tcg-patch-modal"
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-label="Apply eval-gap patch"
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
                        data-testid="eval-patch-loading"
                    >
                        Loading preview…
                    </div>
                )}

                {error && (
                    <ErrorPanel
                        envelope={error}
                        onDismiss={() => setError(null)}
                        testIdPrefix="eval-patch-error"
                    />
                )}

                {preview && (
                    <>
                        {preview.plain_english && (
                            <p
                                className="tcg-patch-modal__plain"
                                data-testid="eval-patch-plain"
                            >
                                {preview.plain_english}
                            </p>
                        )}

                        {isBaselinePromote(preview) && (
                            <table className="tcg-patch-modal__diff">
                                <thead>
                                    <tr>
                                        <th>Field</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr data-testid="eval-patch-row-experiment">
                                        <td>Experiment</td>
                                        <td>
                                            <strong>
                                                {preview.candidate.experiment_name}
                                            </strong>{' '}
                                            (#{preview.candidate.experiment_id})
                                        </td>
                                    </tr>
                                    <tr data-testid="eval-patch-row-checkpoint">
                                        <td>Checkpoint</td>
                                        <td>
                                            step{' '}
                                            <strong>
                                                {preview.candidate.checkpoint_step}
                                            </strong>
                                            {preview.candidate.checkpoint_is_best && (
                                                <> · best</>
                                            )}
                                        </td>
                                    </tr>
                                    <tr data-testid="eval-patch-row-pass-rate">
                                        <td>Pass rate</td>
                                        <td>
                                            <strong>
                                                {(
                                                    preview.candidate.pass_rate * 100
                                                ).toFixed(1)}
                                                %
                                            </strong>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        )}

                        {isLabelKl(preview) && (
                            <>
                                <table
                                    className="tcg-patch-modal__diff"
                                    data-testid="eval-patch-kl-table"
                                >
                                    <thead>
                                        <tr>
                                            <th>Label</th>
                                            <th>Before</th>
                                            <th aria-hidden="true">→</th>
                                            <th>After</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.keys(preview.before.counts)
                                            .sort()
                                            .map((label) => (
                                                <tr
                                                    key={label}
                                                    data-testid={`eval-patch-kl-row-${label}`}
                                                >
                                                    <td>
                                                        <code>{label}</code>
                                                    </td>
                                                    <td>
                                                        {preview.before.counts[label]}
                                                    </td>
                                                    <td aria-hidden="true">→</td>
                                                    <td>
                                                        <strong>
                                                            {preview.after.counts[label]}
                                                        </strong>
                                                    </td>
                                                </tr>
                                            ))}
                                        <tr data-testid="eval-patch-kl-row-summary">
                                            <td>KL (nats)</td>
                                            <td>
                                                {preview.before.kl_nats.toFixed(3)}
                                            </td>
                                            <td aria-hidden="true">→</td>
                                            <td>
                                                <strong>
                                                    {preview.after.kl_nats.toFixed(3)}
                                                </strong>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                                {preview.skipped_reason && (
                                    <p
                                        className="tcg-patch-modal__plain"
                                        data-testid="eval-patch-skipped"
                                    >
                                        {preview.skipped_reason}
                                    </p>
                                )}
                            </>
                        )}
                    </>
                )}

                <footer className="tcg-patch-modal__foot">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={onClose}
                        data-testid="eval-patch-cancel"
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
                        data-testid="eval-patch-apply"
                    >
                        {applying ? 'Applying…' : 'Apply'}
                    </button>
                </footer>
            </div>
        </div>
    );
}
