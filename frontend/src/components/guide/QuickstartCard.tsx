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

import {
    quickstartImportSample,
    quickstartTrainDefault,
    quickstartEvaluateLatest,
    type ImportSampleSummary,
    type TrainDefaultResponse,
    type EvaluateLatestResponse,
} from '../../api/quickstart';
import { useToastStore } from '../../stores/toastStore';

type ButtonState<T> =
    | { status: 'idle' }
    | { status: 'running' }
    | { status: 'success'; result: T }
    | { status: 'error'; message: string };

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
}

export default function QuickstartCard({
    projectId,
    hasBaseModel,
    onRefresh,
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

    const { addToast } = useToastStore();

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

                {/* 2. Train default config */}
                <ActionTile
                    index={2}
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
                />

                {/* 3. Evaluate */}
                <ActionTile
                    index={3}
                    icon="📊"
                    title="Evaluate"
                    description={
                        evalState.status === 'success'
                            ? `Eval on experiment #${evalState.result.experiment_id} — ${evalState.result.eval_type}`
                            : 'Runs eval on the latest experiment against your gold/test split.'
                    }
                    state={evalState}
                    onRun={runEval}
                    runLabel="Evaluate"
                    runningLabel="Evaluating…"
                    successLabel="Evaluated"
                    testid="quickstart-eval"
                />
            </div>
        </section>
    );
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
                border: '1px solid var(--border-color)',
                background: 'var(--bg-card)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-sm)',
            }}
        >
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
