/**
 * The Train form layout (Epic U phase U2): readiness from the preflight,
 * Train = create + launch confirmation, More options reach the panel's own
 * sections, the runs list sits below.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('./PreRunConfirmModal', () => ({
    default: ({ onConfirm, baseModel }: { onConfirm: () => void; baseModel: string }) => (
        <div data-testid="pre-run-modal">
            {baseModel}
            <button type="button" onClick={onConfirm}>Launch</button>
        </div>
    ),
}));
vi.mock('./CloudBurstPlanningSection', () => ({ default: () => <div data-testid="cloud-burst" /> }));
// After Launch the panel switches to the run view, whose plan panel expects a real estimate.
vi.mock('./WhyThisPlanPanel', () => ({ default: () => <div data-testid="why-this-plan" /> }));
vi.mock('../shared/ReadinessPanel', () => ({ ReadinessPanel: () => <div data-testid="readiness-panel" /> }));

import TrainingPanel from './TrainingPanel';

const EXPERIMENTS = [
    { id: 41, name: 'old-run', status: 'completed', base_model: 'microsoft/deberta-v3-base', config: {}, created_at: '2026-10-01T00:00:00Z' },
];

function installApi(preflight: { ok: boolean; errors: string[]; warnings: string[] }) {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/runtimes')) return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        if (url.endsWith('/training/experiments')) return { data: EXPERIMENTS };
        if (url.includes('/training/experiments/41/preflight')) return { data: { preflight: { ok: true, errors: [], warnings: [] } } };
        if (url.includes('/training/experiments/77/preflight')) return { data: { preflight: { ok: true, errors: [], warnings: [] } } };
        return { data: {} };
    });
    apiMock.post.mockImplementation(async (url: string, body?: { name?: string; config?: Record<string, unknown> }) => {
        if (url.endsWith('/training/experiments/preflight')) {
            return { data: { preflight, resolved_training_config: {}, profile_defaults_applied: [] } };
        }
        if (url.endsWith('/training/experiments')) {
            return { data: { id: 77, name: body?.name, status: 'created', base_model: body?.config?.base_model ?? 'x', config: body?.config ?? {} } };
        }
        if (url.endsWith('/training/experiments/77/start')) return { data: { task_id: 't1' } };
        return { data: {} };
    });
}

describe('TrainingPanel — form layout', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
        installApi({ ok: true, errors: [], warnings: [] });
    });

    it('checks readiness on mount and shows the runs list below the form', async () => {
        render(<TrainingPanel projectId={1} title="Runs" taskShapeLabel="Span extraction (span-extraction)" />);
        expect(screen.getByTestId('train-form')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        expect(apiMock.post).toHaveBeenCalledWith('/projects/1/training/experiments/preflight', expect.anything());
        expect(screen.getByTestId('train-form-task')).toHaveTextContent('Span extraction (span-extraction) · causal_lm');
        expect(await screen.findByText('old-run')).toBeInTheDocument();
        // No setup tabs, no journey strip.
        expect(screen.queryByRole('tab', { name: /Power Tools/i })).toBeNull();
        expect(screen.queryByText('+ New Experiment')).toBeNull();
    });

    it('Train creates the run (dated name when blank, seeds in the config) and opens the launch confirmation', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        await user.click(screen.getByTestId('train-form-seeds-3'));
        await user.click(screen.getByTestId('train-form-train'));
        await waitFor(() => expect(screen.getByTestId('pre-run-modal')).toBeInTheDocument());
        const createCall = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
        expect(createCall).toBeTruthy();
        const body = createCall![1] as { name: string; config: Record<string, unknown> };
        expect(body.name).toMatch(/^run-\d{4}-\d{2}-\d{2}-\d{4}$/);
        expect(body.config.num_seeds).toBe(3);
        await user.click(screen.getByRole('button', { name: 'Launch' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/1/training/experiments/77/start'));
    });

    it('a time limit in hours is sent as external_timeout_seconds; blank sends none', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} openSection="model" />);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        await user.click(screen.getByTestId('train-form-train'));
        await waitFor(() => expect(screen.getByTestId('pre-run-modal')).toBeInTheDocument());
        const first = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
        expect((first![1] as { config: Record<string, unknown> }).config).not.toHaveProperty('external_timeout_seconds');

        apiMock.post.mockClear();
        await user.type(await screen.findByLabelText('Time limit (hours)'), '24');
        await user.click(screen.getByTestId('train-form-train'));
        await waitFor(() => expect(apiMock.post.mock.calls.some(([url]) => url === '/projects/1/training/experiments')).toBe(true));
        const second = apiMock.post.mock.calls.find(([url]) => url === '/projects/1/training/experiments');
        expect((second![1] as { config: Record<string, unknown> }).config.external_timeout_seconds).toBe(86400);
    });

    it('a blocked preflight disables nothing but shows the issues, and Fix opens the Checks section', async () => {
        installApi({ ok: false, errors: ['No prepared split'], warnings: [] });
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('1 blocking issue.'));
        expect(screen.getByTestId('train-form-issues')).toHaveTextContent('No prepared split');
        await user.click(screen.getByTestId('train-form-fix'));
        expect(await screen.findByTestId('readiness-panel')).toBeInTheDocument();
    });

    it('More options exposes LoRA, seeds, cloud burst and the extra sections; ?open= opens one', async () => {
        const user = userEvent.setup();
        render(
            <TrainingPanel
                projectId={1}
               
                openSection="lora"
                moreOptions={[{ id: 'models', title: 'Base model registry', render: () => <div data-testid="registry-body" /> }]}
            />,
        );
        expect(await screen.findByTestId('training-config-curriculum-row')).toBeInTheDocument();
        await user.click(screen.getByTestId('advanced-train-advanced-toggle'));
        await user.click(screen.getByTestId('train-section-cloud'));
        expect(screen.getByTestId('cloud-burst')).toBeInTheDocument();
        await user.click(screen.getByTestId('train-section-seeds'));
        expect(screen.getByTestId('training-multi-seed-section')).toBeInTheDocument();
        expect(screen.queryByTestId('registry-body')).toBeNull();
        await user.click(screen.getByTestId('train-form-browse'));
        expect(await screen.findByTestId('registry-body')).toBeInTheDocument();
    });
});

describe('TrainingPanel — Train on a run that never started (phase 4: Train, Launch after a reload)', () => {
    const WAITING = { id: 90, name: 'run-2026-11-20-1200', status: 'pending', started_at: null, base_model: 'HuggingFaceTB/SmolLM2-135M-Instruct', config: { num_epochs: 3 }, created_at: '2026-11-20T12:00:00Z' };

    function withRuns(runs: unknown[]) {
        installApi({ ok: true, errors: [], warnings: [] });
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string) => (url.endsWith('/training/experiments') ? { data: runs } : base(url)));
    }

    const created = () => apiMock.post.mock.calls.filter(([url]) => url === '/projects/1/training/experiments');

    beforeEach(() => {
        window.history.replaceState({}, '', '/');
    });

    it('the newest run is waiting and the form is untouched: Train opens its confirmation, no second run', async () => {
        withRuns([WAITING, EXPERIMENTS[0]]);
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
        expect(await screen.findByTestId('train-form-waiting-run')).toHaveTextContent('Run #90 · run-2026-11-20-1200');
        await user.click(screen.getByTestId('train-form-train'));
        expect(await screen.findByTestId('pre-run-modal')).toHaveTextContent('HuggingFaceTB/SmolLM2-135M-Instruct');
        expect(created()).toHaveLength(0);
        await user.click(screen.getByRole('button', { name: 'Launch' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/1/training/experiments/90/start'));
    });

    it('any edit (here a run name) creates a new run as before', async () => {
        withRuns([WAITING]);
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} />);
        await screen.findByTestId('train-form-waiting-run');
        await user.type(screen.getByTestId('train-form-run-name'), 'second');
        expect(screen.queryByTestId('train-form-waiting-run')).toBeNull();
        await user.click(screen.getByTestId('train-form-train'));
        await waitFor(() => expect(created()).toHaveLength(1));
    });

    it('a waiting run that is not the newest, a started one, or a multi-seed one is never reused', async () => {
        for (const runs of [
            [{ ...WAITING, id: 89 }, { ...EXPERIMENTS[0], id: 95 }],
            [{ ...WAITING, started_at: '2026-11-20T12:01:00Z' }],
            [{ ...WAITING, config: { num_seeds: 3 } }],
        ]) {
            withRuns(runs);
            const user = userEvent.setup();
            const view = render(<TrainingPanel projectId={1} />);
            await waitFor(() => expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.'));
            expect(screen.queryByTestId('train-form-waiting-run')).toBeNull();
            await user.click(screen.getByTestId('train-form-train'));
            await waitFor(() => expect(created()).toHaveLength(1));
            view.unmount();
        }
    });
});
