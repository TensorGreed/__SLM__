/**
 * Quickstart card (Theme 1 Epic 4).
 *
 * Three one-click buttons that replace the first three checklist
 * descriptions on the project guide page with actual actions:
 *
 *   1. Import sample CSV   → materialize a bundled demo dataset
 *   2. Train default config → kick off a training run
 *   3. Evaluate             → run heldout eval on the latest experiment
 *
 * Each button is its own little state machine:
 *   idle → running → success | error
 *
 * Errors stay inline; toasts fire for the user-level confirmation so
 * the buttons themselves stay focused on "what happened to my step."
 *
 * On success of any button we call `refreshPipelineStatus()` so the
 * 12-step checklist below picks up the new pipeline stage on the
 * next render — the checkmarks flip without a page reload.
 */

import { useState } from 'react';

import api from '../../api/client';
import {
    quickstartImportSample,
    quickstartTrainDefault,
    quickstartBaselineEval,
    quickstartEvaluateLatest,
    type BaselineEvalResponse,
    type ImportSampleSummary,
    type TrainDefaultResponse,
    type EvaluateLatestResponse,
} from '../../api/quickstart';
import { useToastStore } from '../../stores/toastStore';

/**
 * Tour nudge ids. Keep these stable — they're persisted on
 * `project.quickstart_tour_state.dismissed_nudges` and the backend
 * doesn't enumerate them; an unknown id is just "never dismissed."
 */
const NUDGE_IMPORT_TO_TRAIN = 'import_to_train';
const NUDGE_TRAIN_TO_EVAL = 'train_to_eval';

type ButtonState<T> =
    | { status: 'idle' }
    | { status: 'running' }
    | { status: 'success'; result: T }
    | { status: 'error'; message: string };

/**
 * Pick the headline metrics out of an EvalResult.metrics dict and
 * render them as a short comma-separated string. The eval handler
 * decides the metric shape per task profile (f1 / exact_match /
 * accuracy / macro_f1 / pass_rate / ...), so we surface the
 * familiar names first and fall back to whatever's in the dict.
 */
function summarizeMetrics(metrics: Record<string, unknown> | undefined): string {
    if (!metrics || typeof metrics !== 'object') return 'no metrics';
    const priority = [
        'f1', 'exact_match', 'accuracy', 'macro_f1', 'precision', 'recall',
        'pass_rate', 'llm_judge_pass_rate', 'groundedness', 'tool_success_rate',
    ];
    const out: string[] = [];
    for (const key of priority) {
        if (out.length >= 2) break;
        const value = metrics[key];
        if (typeof value === 'number' && Number.isFinite(value)) {
            out.push(`${key} ${value.toFixed(2)}`);
        }
    }
    if (out.length === 0) {
        // Fallback: pick the first two numeric entries in insertion order.
        for (const [key, value] of Object.entries(metrics)) {
            if (out.length >= 2) break;
            if (typeof value === 'number' && Number.isFinite(value)) {
                out.push(`${key} ${value.toFixed(2)}`);
            }
        }
    }
    return out.length ? out.join(' · ') : 'no metrics';
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
        if (typeof detail === 'string' && detail.trim()) return detail;
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) return message;
    }
    return 'Unknown error';
}

interface QuickstartCardProps {
    projectId: number;
    /** True when the project already has a recipe applied (sets
     * base_model_name). Drives whether the Train button is unlocked. */
    hasBaseModel: boolean;
    /** Called after each successful action so the parent can re-poll
     * pipeline status — the checklist's checkmarks pick up the new
     * stage on the next render. */
    onRefresh?: () => void;
    /** Nudge ids already dismissed on this project (sourced from
     * `project.quickstart_tour_state.dismissed_nudges`). Dismissing
     * one here PUTs back the updated set so the nudge doesn't
     * replay on a future session. */
    initialDismissedNudges?: string[];
}

export default function QuickstartCard({
    projectId,
    hasBaseModel,
    onRefresh,
    initialDismissedNudges,
}: QuickstartCardProps) {
    const [importState, setImportState] = useState<ButtonState<ImportSampleSummary>>({
        status: 'idle',
    });
    const [trainState, setTrainState] = useState<ButtonState<TrainDefaultResponse>>({
        status: 'idle',
    });
    const [evalState, setEvalState] = useState<ButtonState<EvaluateLatestResponse>>({
        status: 'idle',
    });
    const [baselineState, setBaselineState] = useState<ButtonState<BaselineEvalResponse>>({
        status: 'idle',
    });
    const [dismissedNudges, setDismissedNudges] = useState<Set<string>>(
        () => new Set(initialDismissedNudges ?? []),
    );

    const { addToast } = useToastStore();

    // Persist a nudge dismissal to the project record so it doesn't
    // replay across sessions. Best-effort: local state still flips
    // even if the PUT errors (the user shouldn't see the nudge again
    // this session regardless), but we surface a quiet warning toast
    // so a persistent failure isn't fully silent.
    const dismissNudge = (nudgeId: string) => {
        if (dismissedNudges.has(nudgeId)) return;
        const next = new Set(dismissedNudges);
        next.add(nudgeId);
        setDismissedNudges(next);
        api
            .put(`/projects/${projectId}`, {
                quickstart_tour_state: { dismissed_nudges: Array.from(next) },
            })
            .catch(() => {
                addToast(
                    "Couldn't save tour preference (will re-show on refresh).",
                    'warning',
                    3500,
                );
            });
    };

    const showImportToTrainNudge =
        importState.status === 'success'
        && trainState.status === 'idle'
        && !dismissedNudges.has(NUDGE_IMPORT_TO_TRAIN);
    const showTrainToEvalNudge =
        trainState.status === 'success'
        && evalState.status === 'idle'
        && !dismissedNudges.has(NUDGE_TRAIN_TO_EVAL);

    const runImport = async () => {
        setImportState({ status: 'running' });
        try {
            const res = await quickstartImportSample(projectId);
            setImportState({ status: 'success', result: res.summary });
            addToast(
                `Imported ${res.summary.source_row_count} rows from ${res.summary.slug}`,
                'success',
                4000,
            );
            onRefresh?.();
        } catch (err) {
            const message = extractErrorMessage(err);
            setImportState({ status: 'error', message });
            addToast(`Import failed: ${message}`, 'error', 5000);
        }
    };

    const runTrain = async () => {
        setTrainState({ status: 'running' });
        try {
            const res = await quickstartTrainDefault(projectId);
            setTrainState({ status: 'success', result: res });
            addToast(
                `Training started · experiment #${res.experiment_id} on ${res.base_model}`,
                'success',
                4000,
            );
            onRefresh?.();
        } catch (err) {
            const message = extractErrorMessage(err);
            setTrainState({ status: 'error', message });
            addToast(`Train failed: ${message}`, 'error', 5000);
        }
    };

    const runEval = async () => {
        setEvalState({ status: 'running' });
        try {
            const res = await quickstartEvaluateLatest(projectId);
            setEvalState({ status: 'success', result: res });
            addToast(
                `Eval finished · experiment #${res.experiment_id}`,
                'success',
                4000,
            );
            onRefresh?.();
        } catch (err) {
            const message = extractErrorMessage(err);
            setEvalState({ status: 'error', message });
            addToast(`Eval failed: ${message}`, 'error', 5000);
        }
    };

    const runBaseline = async () => {
        setBaselineState({ status: 'running' });
        try {
            const res = await quickstartBaselineEval(projectId);
            setBaselineState({ status: 'success', result: res });
            addToast(
                `Baseline established · ${summarizeMetrics(res.result.metrics)}`,
                'success',
                4500,
            );
            onRefresh?.();
        } catch (err) {
            const message = extractErrorMessage(err);
            setBaselineState({ status: 'error', message });
            addToast(`Baseline failed: ${message}`, 'error', 5000);
        }
    };

    return (
        <section
            className="card"
            data-testid="quickstart-card"
            style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-md)',
            }}
        >
            <div>
                <h3 style={{ margin: 0 }}>Quickstart</h3>
                <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)' }}>
                    Three one-click actions to take a fresh project from
                    empty → trained → evaluated. Each step also has a full
                    UI for power users — see the checklist below.
                </p>
            </div>

            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                    gap: 'var(--space-md)',
                }}
            >
                {/* 1. Import sample CSV */}
                <ActionTile
                    index={1}
                    icon="📂"
                    title="Import sample CSV"
                    description={
                        importState.status === 'success'
                            ? `Imported ${importState.result.source_row_count} rows from ${importState.result.slug} (${importState.result.prepared_train_rows} train · ${importState.result.prepared_val_rows} val · ${importState.result.prepared_test_rows} test)`
                            : 'Bundled demo data tailored to your recipe.'
                    }
                    state={importState}
                    onRun={runImport}
                    runLabel="Import sample"
                    runningLabel="Importing…"
                    successLabel="Imported"
                    testid="quickstart-import"
                />

                {/* 2. Baseline (untrained) — Theme 8 Epic 1 */}
                <ActionTile
                    index={2}
                    icon="📐"
                    title="Baseline (untrained)"
                    description={
                        !hasBaseModel
                            ? 'Pick a recipe first — baseline runs against the recipe\'s suggested model.'
                            : baselineState.status === 'success'
                                ? `Untrained baseline · ${summarizeMetrics(baselineState.result.result.metrics)}`
                                : 'Optional but recommended — gives your post-training numbers an anchor.'
                    }
                    state={baselineState}
                    onRun={runBaseline}
                    runLabel="Run baseline"
                    runningLabel="Evaluating…"
                    successLabel="Baseline set"
                    testid="quickstart-baseline"
                    disabledReason={!hasBaseModel ? 'no-base-model' : null}
                />

                {/* 3. Train default config */}
                <ActionTile
                    index={3}
                    icon="🧪"
                    title="Train default config"
                    description={
                        !hasBaseModel
                            ? 'Pick a recipe in the dataset-import wizard first — it sets the base model used here.'
                            : trainState.status === 'success'
                                ? `Experiment #${trainState.result.experiment_id} started on ${trainState.result.base_model}`
                                : 'Launches a training run using the recipe defaults already on this project.'
                    }
                    state={trainState}
                    onRun={runTrain}
                    runLabel="Train now"
                    runningLabel="Starting…"
                    successLabel="Started"
                    testid="quickstart-train"
                    disabledReason={!hasBaseModel ? 'no-base-model' : null}
                    nudge={
                        showImportToTrainNudge && importState.status === 'success'
                            ? {
                                testid: 'quickstart-train-nudge',
                                message: `Imported ${importState.result.source_row_count} rows + ${importState.result.gold_row_count} gold-set entries + train/val/test splits. Train a model on them next.`,
                                onDismiss: () => dismissNudge(NUDGE_IMPORT_TO_TRAIN),
                            }
                            : null
                    }
                />

                {/* 4. Evaluate */}
                <ActionTile
                    index={4}
                    icon="📊"
                    title="Evaluate"
                    description={
                        evalState.status === 'success'
                            ? (
                                baselineState.status === 'success'
                                    // Show baseline → trained side-by-side so the
                                    // lift from SFT is the headline number, not
                                    // an absolute score the user has to
                                    // contextualize on their own.
                                    ? `Baseline ${summarizeMetrics(baselineState.result.result.metrics)} → trained ${summarizeMetrics((evalState.result.result as Record<string, unknown>)?.metrics as Record<string, unknown> | undefined)}`
                                    : `Eval on experiment #${evalState.result.experiment_id} — ${evalState.result.eval_type}`
                            )
                            : 'Runs eval on the latest experiment against your gold/test split.'
                    }
                    state={evalState}
                    onRun={runEval}
                    runLabel="Evaluate"
                    runningLabel="Evaluating…"
                    successLabel="Evaluated"
                    testid="quickstart-eval"
                    nudge={
                        showTrainToEvalNudge && trainState.status === 'success'
                            ? {
                                testid: 'quickstart-eval-nudge',
                                message: `Experiment #${trainState.result.experiment_id} started on ${trainState.result.base_model}. Once it's done, evaluate against the gold set.`,
                                onDismiss: () => dismissNudge(NUDGE_TRAIN_TO_EVAL),
                            }
                            : null
                    }
                />
            </div>
        </section>
    );
}

interface TileNudge {
    /** Stable nudge id, persisted in dismissed_nudges. */
    testid: string;
    /** Single-line "what just happened + do this next" copy. */
    message: string;
    onDismiss: () => void;
}

interface ActionTileProps<T> {
    index: number;
    icon: string;
    title: string;
    description: string;
    state: ButtonState<T>;
    onRun: () => void;
    runLabel: string;
    runningLabel: string;
    successLabel: string;
    testid: string;
    disabledReason?: string | null;
    /** Optional tour nudge shown above the tile header (Theme 1 Epic 2). */
    nudge?: TileNudge | null;
}

function ActionTile<T>({
    index,
    icon,
    title,
    description,
    state,
    onRun,
    runLabel,
    runningLabel,
    successLabel,
    testid,
    disabledReason,
    nudge,
}: ActionTileProps<T>) {
    const isRunning = state.status === 'running';
    const isSuccess = state.status === 'success';
    const isError = state.status === 'error';
    const disabled = isRunning || Boolean(disabledReason);

    let badge: { label: string; tone: 'info' | 'success' | 'error' | 'warning' };
    if (isSuccess) {
        badge = { label: '✓ Done', tone: 'success' };
    } else if (isError) {
        badge = { label: 'Failed', tone: 'error' };
    } else if (isRunning) {
        badge = { label: 'Running', tone: 'warning' };
    } else {
        badge = { label: `Step ${index}`, tone: 'info' };
    }

    return (
        <div
            data-testid={testid}
            style={{
                padding: 'var(--space-md)',
                borderRadius: 'var(--radius-md)',
                border: nudge
                    ? '1px solid var(--color-warning)'
                    : '1px solid var(--border-color)',
                background: 'var(--bg-card)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-sm)',
                position: 'relative',
            }}
        >
            {nudge && (
                <div
                    role="status"
                    data-testid={nudge.testid}
                    style={{
                        margin: 'calc(-1 * var(--space-md)) calc(-1 * var(--space-md)) 0',
                        padding: 'var(--space-sm) var(--space-md)',
                        background: 'var(--color-warning-bg)',
                        color: 'var(--color-warning)',
                        borderTopLeftRadius: 'var(--radius-md)',
                        borderTopRightRadius: 'var(--radius-md)',
                        borderBottom: '1px solid var(--color-warning)',
                        fontSize: '0.82rem',
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 'var(--space-sm)',
                    }}
                >
                    <span aria-hidden="true">💡 ↓</span>
                    <span style={{ flex: 1 }}>{nudge.message}</span>
                    <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={nudge.onDismiss}
                        aria-label="Dismiss tour nudge"
                        data-testid={`${nudge.testid}-dismiss`}
                        style={{
                            padding: 0,
                            fontSize: '1rem',
                            lineHeight: 1,
                            color: 'var(--color-warning)',
                        }}
                    >
                        ✕
                    </button>
                </div>
            )}
            <div
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                }}
            >
                <span style={{ fontSize: '1.4rem' }} aria-hidden="true">
                    {icon}
                </span>
                <span
                    className={`badge badge-${badge.tone}`}
                    data-testid={`${testid}-badge`}
                >
                    {badge.label}
                </span>
            </div>
            <div style={{ fontWeight: 600 }}>{title}</div>
            <div
                style={{
                    color: 'var(--text-secondary)',
                    fontSize: '0.85rem',
                    flex: 1,
                }}
                data-testid={`${testid}-description`}
            >
                {description}
            </div>
            {isError && (
                <div
                    role="alert"
                    style={{
                        padding: 'var(--space-xs) var(--space-sm)',
                        background: 'var(--color-error-bg)',
                        color: 'var(--color-error)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.8rem',
                    }}
                    data-testid={`${testid}-error`}
                >
                    {state.message}
                </div>
            )}
            <button
                type="button"
                className={isSuccess ? 'btn btn-secondary' : 'btn btn-primary'}
                onClick={onRun}
                disabled={disabled}
                data-testid={`${testid}-button`}
            >
                {isRunning ? runningLabel : isSuccess ? `${successLabel} · run again` : runLabel}
            </button>
        </div>
    );
}
