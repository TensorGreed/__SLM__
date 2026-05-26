import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import QuickstartCard from './QuickstartCard';

const SAMPLE_IMPORT_SUMMARY = {
    slug: 'support-faq',
    created: true,
    project_id: 1,
    project_name: 'Test',
    source_dataset_id: 10,
    source_row_count: 20,
    gold_set_id: 11,
    gold_version_id: 1,
    gold_row_count: 6,
    prepared_train_path: '/tmp/train.jsonl',
    prepared_train_rows: 14,
    prepared_val_rows: 3,
    prepared_test_rows: 3,
    prepared_dataset_ids: { train: 12, val: 13, test: 14 },
    adapter_id: 'qa-pair',
    task_profile: 'instruction_sft',
    suggested_brief: 'Build a support assistant…',
};

const SAMPLE_TRAIN_RESULT = {
    status: 'training_started',
    experiment_id: 7,
    experiment_name: 'Quickstart · default config',
    base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    training_mode: 'sft',
    recipe_id: 'qa-sft',
    start_result: {},
};

/**
 * Job stub mirror of ``serialize_job`` for an async eval kick-off.
 * The evaluate-latest / baseline-eval tiles now POST with
 * ``?async_job=true`` and receive a Job stub back instead of an
 * inline EvalResult — the actual eval runs in the background and
 * the notification bell surfaces its terminal status.
 */
const SAMPLE_EVAL_JOB = {
    id: 901,
    kind: 'quickstart_evaluate_latest',
    title: 'Quickstart eval-latest · experiment #7',
    status: 'queued',
    progress: null,
    progress_message: null,
    project_id: 1,
    user_id: null,
    params: { experiment_id: 7, max_samples: 100, eval_type: 'exact_match' },
    result: null,
    error: null,
    queued_at: '2026-05-26T00:00:00Z',
    started_at: null,
    completed_at: null,
    dismissed_at: null,
};

const SAMPLE_BASELINE_JOB = {
    id: 902,
    kind: 'quickstart_baseline_eval',
    title: 'Quickstart baseline-eval · SmolLM2-135M-Instruct',
    status: 'queued',
    progress: null,
    progress_message: null,
    project_id: 1,
    user_id: null,
    params: {
        experiment_id: 42,
        model_path: 'HuggingFaceTB/SmolLM2-135M-Instruct',
        max_samples: 100,
    },
    result: null,
    error: null,
    queued_at: '2026-05-26T00:00:00Z',
    started_at: null,
    completed_at: null,
    dismissed_at: null,
};

describe('QuickstartCard', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
        apiMock.get.mockReset();
        apiMock.put.mockReset();
        apiMock.delete.mockReset();
    });

    it('renders four action tiles with the correct step numbers', () => {
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);
        expect(screen.getByTestId('quickstart-import')).toBeInTheDocument();
        expect(screen.getByTestId('quickstart-baseline')).toBeInTheDocument();
        expect(screen.getByTestId('quickstart-train')).toBeInTheDocument();
        expect(screen.getByTestId('quickstart-eval')).toBeInTheDocument();
        expect(screen.getByTestId('quickstart-import-badge')).toHaveTextContent('Step 1');
        expect(screen.getByTestId('quickstart-baseline-badge')).toHaveTextContent('Step 2');
        expect(screen.getByTestId('quickstart-train-badge')).toHaveTextContent('Step 3');
        expect(screen.getByTestId('quickstart-eval-badge')).toHaveTextContent('Step 4');
    });

    it('disables Train when the project has no base model yet', () => {
        render(<QuickstartCard projectId={1} hasBaseModel={false} />);
        const trainBtn = screen.getByTestId('quickstart-train-button') as HTMLButtonElement;
        expect(trainBtn.disabled).toBe(true);
        expect(screen.getByTestId('quickstart-train-description')).toHaveTextContent(
            /Pick a recipe in the dataset-import wizard/,
        );
    });

    it('runs the import action, flips to Done, fires onRefresh + a toast', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: { status: 'ok', summary: SAMPLE_IMPORT_SUMMARY },
        });
        const onRefresh = vi.fn();
        render(
            <QuickstartCard projectId={1} hasBaseModel={true} onRefresh={onRefresh} />,
        );

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-import-button'));

        await waitFor(() => {
            expect(screen.getByTestId('quickstart-import-badge')).toHaveTextContent('✓ Done');
        });
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/quickstart/import-sample',
            {},
        );
        expect(screen.getByTestId('quickstart-import-description')).toHaveTextContent(
            /Imported 20 rows from support-faq/,
        );
        expect(onRefresh).toHaveBeenCalledTimes(1);
    });

    it('surfaces import errors inline + does NOT call onRefresh on failure', async () => {
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'project_not_found:99' } },
        });
        const onRefresh = vi.fn();
        render(
            <QuickstartCard projectId={99} hasBaseModel={true} onRefresh={onRefresh} />,
        );

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-import-button'));

        await waitFor(() => {
            expect(screen.getByTestId('quickstart-import-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('quickstart-import-error')).toHaveTextContent(
            'project_not_found:99',
        );
        expect(screen.getByTestId('quickstart-import-badge')).toHaveTextContent('Failed');
        expect(onRefresh).not.toHaveBeenCalled();
    });

    it('runs Train Default and flips its badge to Done', async () => {
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_TRAIN_RESULT });
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-train-button'));

        await waitFor(() => {
            expect(screen.getByTestId('quickstart-train-badge')).toHaveTextContent('✓ Done');
        });
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/quickstart/train-default',
        );
        expect(screen.getByTestId('quickstart-train-description')).toHaveTextContent(
            /Experiment #7 started/,
        );
    });

    it('runs Evaluate as async Job, surfaces queued state + bell hint', async () => {
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_EVAL_JOB });
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-eval-button'));

        await waitFor(() => {
            expect(screen.getByTestId('quickstart-eval-badge')).toHaveTextContent('✓ Done');
        });
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/quickstart/evaluate-latest?async_job=true',
        );
        expect(screen.getByTestId('quickstart-eval-description')).toHaveTextContent(
            /Eval queued \(job #901\) — bell will notify when ready/,
        );
    });

    // ── Tour nudges (Theme 1 Epic 2) ────────────────────────────

    it('renders the import→train nudge with row + gold counts after import succeeds', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: { status: 'ok', summary: SAMPLE_IMPORT_SUMMARY },
        });
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);

        // Nudge does NOT appear before any action runs.
        expect(screen.queryByTestId('quickstart-train-nudge')).not.toBeInTheDocument();

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-import-button'));

        const nudge = await screen.findByTestId('quickstart-train-nudge');
        expect(nudge).toHaveTextContent(/Imported 20 rows \+ 6 gold-set entries/);
        expect(nudge).toHaveTextContent(/Train a model on them next/);
    });

    it('dismissing the import→train nudge PUTs the dismissal back to the project', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: { status: 'ok', summary: SAMPLE_IMPORT_SUMMARY },
        });
        apiMock.put.mockResolvedValue({ data: { id: 1 } });
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-import-button'));
        await screen.findByTestId('quickstart-train-nudge');
        await user.click(screen.getByTestId('quickstart-train-nudge-dismiss'));

        // Nudge disappears immediately.
        expect(screen.queryByTestId('quickstart-train-nudge')).not.toBeInTheDocument();

        // Dismissal persisted via PUT /projects/{id}.
        expect(apiMock.put).toHaveBeenCalledWith(
            '/projects/1',
            expect.objectContaining({
                quickstart_tour_state: {
                    dismissed_nudges: ['import_to_train'],
                },
            }),
        );
    });

    it('does not re-show a nudge whose id is already in initialDismissedNudges', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: { status: 'ok', summary: SAMPLE_IMPORT_SUMMARY },
        });
        render(
            <QuickstartCard
                projectId={1}
                hasBaseModel={true}
                initialDismissedNudges={['import_to_train']}
            />,
        );

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-import-button'));

        // Wait for the badge to flip — confirms the import POST resolved.
        await waitFor(() => {
            expect(screen.getByTestId('quickstart-import-badge')).toHaveTextContent('✓ Done');
        });

        // Nudge stays suppressed because the user already dismissed it in a
        // previous session.
        expect(screen.queryByTestId('quickstart-train-nudge')).not.toBeInTheDocument();
    });

    it('renders the train→eval nudge after a successful train', async () => {
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_TRAIN_RESULT });
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-train-button'));

        const nudge = await screen.findByTestId('quickstart-eval-nudge');
        expect(nudge).toHaveTextContent(
            /Experiment #7 started on HuggingFaceTB\/SmolLM2-135M-Instruct/,
        );
        expect(nudge).toHaveTextContent(/evaluate against the gold set/i);
    });

    // ── Baseline tile (Theme 8 Epic 1) ──────────────────────────

    it('disables the Baseline tile when no recipe / base_model is set', () => {
        render(<QuickstartCard projectId={1} hasBaseModel={false} />);
        const btn = screen.getByTestId('quickstart-baseline-button') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
        expect(screen.getByTestId('quickstart-baseline-description')).toHaveTextContent(
            /Pick a recipe first/,
        );
    });

    it('runs the baseline action as async Job, flips to queued state + bell hint', async () => {
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_BASELINE_JOB });
        const onRefresh = vi.fn();
        render(
            <QuickstartCard projectId={1} hasBaseModel={true} onRefresh={onRefresh} />,
        );

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-baseline-button'));

        await waitFor(() => {
            expect(screen.getByTestId('quickstart-baseline-badge')).toHaveTextContent('✓ Done');
        });
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/quickstart/baseline-eval?async_job=true',
        );
        const desc = screen.getByTestId('quickstart-baseline-description');
        expect(desc).toHaveTextContent(/Baseline queued \(job #902\) — bell will notify when ready/);
        expect(onRefresh).toHaveBeenCalledTimes(1);
    });

    it('surfaces baseline error inline + does NOT call onRefresh', async () => {
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'No test split found yet.' } },
        });
        const onRefresh = vi.fn();
        render(
            <QuickstartCard projectId={1} hasBaseModel={true} onRefresh={onRefresh} />,
        );

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-baseline-button'));

        await waitFor(() => {
            expect(screen.getByTestId('quickstart-baseline-error')).toBeInTheDocument();
        });
        expect(screen.getByTestId('quickstart-baseline-error')).toHaveTextContent(
            'No test split found yet.',
        );
        expect(onRefresh).not.toHaveBeenCalled();
    });

    it('queues both baseline + eval jobs independently when run back-to-back', async () => {
        // The previous side-by-side metrics view collapsed when these
        // tiles moved to the async-Job path — actual results land on
        // the Eval tab once the bell signals completion. The two
        // tiles just both flash queued + retain their job-id hint.
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_BASELINE_JOB });
        apiMock.post.mockResolvedValueOnce({ data: SAMPLE_EVAL_JOB });
        render(<QuickstartCard projectId={1} hasBaseModel={true} />);

        const user = userEvent.setup();
        await user.click(screen.getByTestId('quickstart-baseline-button'));
        await waitFor(() => {
            expect(screen.getByTestId('quickstart-baseline-badge')).toHaveTextContent('✓ Done');
        });

        await user.click(screen.getByTestId('quickstart-eval-button'));
        await waitFor(() => {
            expect(screen.getByTestId('quickstart-eval-badge')).toHaveTextContent('✓ Done');
        });

        expect(screen.getByTestId('quickstart-baseline-description'))
            .toHaveTextContent(/Baseline queued \(job #902\)/);
        expect(screen.getByTestId('quickstart-eval-description'))
            .toHaveTextContent(/Eval queued \(job #901\)/);
    });
});
