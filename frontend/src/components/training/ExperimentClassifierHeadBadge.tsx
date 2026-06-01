/**
 * Surfaces the "classifier head detected" signal on the experiment
 * detail row (δ + δ′ work).
 *
 * When an experiment was trained with ``task_type=classification``
 * the trainer loaded ``AutoModelForSequenceClassification`` and saved
 * a PEFT adapter carrying a SEQ_CLS head. δ's held-out eval path
 * detects this and dispatches through the head's logits at eval
 * time — *not* through ``model.generate()``. Without this badge a
 * user looking at their experiment row had no way to tell which
 * inference path their eval would take; they'd have to read the
 * eval result's runtime metadata after the fact. This badge makes
 * it visible up-front.
 *
 * The hint is intentionally short — full details live in the
 * tooltip + the δ′ smoke check's remediation copy.
 */

interface ExperimentClassifierHeadBadgeProps {
    /** Pass ``experiment.config?.task_type`` directly. */
    taskType?: string | null;
    /** ``experiment.status`` — we only show the badge for completed
     *  experiments so we don't mislead during a still-running job. */
    status?: string | null;
}

export default function ExperimentClassifierHeadBadge({
    taskType,
    status,
}: ExperimentClassifierHeadBadgeProps) {
    const normalizedTask = (taskType || '').trim().toLowerCase();
    if (normalizedTask !== 'classification') return null;
    // The badge means "this experiment trained a classifier head".
    // A still-running / failed / pending experiment hasn't saved
    // one yet — showing the badge would be premature.
    if (status !== 'completed') return null;

    return (
        <span
            className="badge badge-info"
            data-testid="experiment-classifier-head-badge"
            title={
                'This experiment trained a classifier head '
                + '(task_type=classification → AutoModelForSequenceClassification). '
                + 'δ: held-out eval dispatches through the head’s logits '
                + 'directly rather than ``model.generate()`` — the LM head '
                + 'was never trained to emit label tokens, so generation '
                + 'would produce garbage. The eval result’s runtime '
                + 'metadata shows ``head=sequence_classification`` to '
                + 'confirm which path fired.'
            }
        >
            🎯 classifier head
        </span>
    );
}
