import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../api/client', () => ({ default: apiMock }));
vi.mock('../components/layout/TopBar', () => ({ default: () => <div data-testid="topbar" /> }));
vi.mock('../stores/toastStore', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } }));
vi.mock('recharts', () => ({
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    BarChart: ({ data }: { data: unknown }) => <div data-testid="bar-data">{JSON.stringify(data)}</div>,
    Bar: () => null, Cell: () => null, CartesianGrid: () => null, XAxis: () => null, YAxis: () => null, Tooltip: () => null,
}));

import PipelineDetailPage from './PipelineDetailPage';
import PipelinesPage from './PipelinesPage';
import PipelineMembership from '../components/pipelines/PipelineMembership';

const PROJECTS = [{ id: 1, name: 'Extractor' }, { id: 2, name: 'Rewriter' }];
const SETTINGS = { servable_runs: [{ id: 7, name: 'v1', base_model: 'toy-base', passage_trained: false, adapter: true }] };

const PIPELINE = {
    id: 'pl-1', name: 'toy', description: '',
    stages: [
        { name: 'extract', project_id: 1, model: '', prompt: { mode: 'template', template: '{text}' }, inputs: { text: { from: 'row', pointer: '/text' } }, output: 'json', max_tokens: 512 },
        { name: 'rewrite', project_id: 2, model: ':run-7', prompt: { mode: 'template', template: '{line}' }, inputs: { line: { from: 'item', pointer: '/line' } },
            fan_out: { pointer: '/items/*', item_key: ['/label'] }, output: 'json', max_tokens: 512 },
    ],
};

const view = (o: Record<string, number> = {}) => ({ scored: 4, correct: 3, accuracy: 0.75, unscored: 1, spurious: 1, missed_upstream: 2, errors: 0, ...o });
const RUN = {
    run_id: 'r-1', pipeline_id: 'pl-1', started_at: '2026-10-27', finished_at: null, rows: 4, truncated: true, calls: 9, errors_by_reason: { missing_input: 1 },
    stages: [
        { name: 'extract', project_id: 1, project_name: 'Extractor', model_id: 'extractor', run_id: 3, handler: 'field_match', end_to_end: view(), isolated: null },
        { name: 'rewrite', project_id: 2, project_name: 'Rewriter', model_id: 'rewriter:run-7', run_id: 7, end_to_end: view({ accuracy: 0.5 }),
            isolated: { ...view({ accuracy: 0.9 }), skipped_rows: 2, unscored_by_reason: { needs_judge: 1 } },
            judge: { kind: 'answer', label: null, ran: false, units: 0, reason: 'no_judge_available' }, judge_not_run: true },
    ],
    end_to_end: { gold_final_items: 10, fully_correct: 6, share_fully_correct: 0.6 },
    attribution: { first_wrong_stage: { extract: 1, rewrite: 3 }, correct: 6, missed_upstream_at: { rewrite: 2 } },
    rows_detail: [{ id: 'row-1', stages: [{ name: 'extract', inputs_count: 1, outputs: [{ item_key_values: null, correct: true, error: null }] }], final: [{ gold_item_key: 'a', fully_correct: false, first_wrong_stage: 'rewrite' }] }],
};

function wire(extra: (url: string) => unknown = () => undefined) {
    apiMock.get.mockImplementation(async (url: string) => {
        const x = extra(url);
        if (x !== undefined) return { data: x };
        if (url === '/projects') return { data: { projects: PROJECTS } };
        if (url.endsWith('/gateway/settings')) return { data: SETTINGS };
        if (url === '/pipelines/pl-1') return { data: PIPELINE };
        if (url === '/pipelines/pl-1/heldout') return { data: { rows: [], total: 2 } };
        if (url === '/pipelines/pl-1/runs') return { data: { runs: [{ id: 'r-1', started_at: '2026-10-27', rows: 4, share_fully_correct: 0.6, truncated: true }] } };
        if (url === '/pipelines/pl-1/runs/r-1') return { data: RUN };
        throw new Error(`unexpected ${url}`);
    });
}

function renderAt(path: string) {
    return render(
        <MemoryRouter initialEntries={[path]}>
            <Routes>
                <Route path="/pipelines" element={<PipelinesPage />} />
                <Route path="/pipelines/:id" element={<PipelineDetailPage />} />
            </Routes>
        </MemoryRouter>,
    );
}

beforeEach(() => { for (const m of Object.values(apiMock)) m.mockReset(); });

describe('Pipelines list', () => {
    it('lists pipelines with stages and last run, and links to new', async () => {
        wire((u) => (u === '/pipelines' ? { pipelines: [{ id: 'pl-1', name: 'toy', stages: [{ name: 'extract', project_id: 1, project_name: 'Extractor' }], heldout_rows: 2, last_run: { id: 'r-1', started_at: null, share_fully_correct: 0.6 } }] } : undefined));
        renderAt('/pipelines');
        expect(await screen.findByTestId('pipeline-item-pl-1')).toHaveTextContent('extract (#1 Extractor)');
        expect(screen.getByTestId('pipeline-item-pl-1')).toHaveTextContent('60% fully correct');
        expect(screen.getByTestId('pipeline-new')).toHaveAttribute('href', '/pipelines/new');
    });
});

describe('Pipeline editor', () => {
    it('creates a pipeline and renders per-field 400 errors beside the stage', async () => {
        wire();
        apiMock.post.mockRejectedValueOnce({ response: { data: { detail: 'Invalid pipeline', errors: [{ field: 'stages[0].project_id', message: 'project does not exist' }, { field: 'name', message: 'required' }] } } });
        renderAt('/pipelines/new');
        await userEvent.type(await screen.findByTestId('pipeline-name'), 'toy');
        await userEvent.click(screen.getByTestId('pipeline-save'));
        const stage = await screen.findByTestId('pipeline-stage-0');
        expect(within(stage).getByTestId('pipeline-error-stages[0].project_id')).toHaveTextContent('project does not exist');
        expect(screen.getByTestId('pipeline-error-name')).toBeInTheDocument();
        expect(apiMock.post.mock.calls[0][0]).toBe('/pipelines');
        expect(apiMock.post.mock.calls[0][1].stages[0].name).toBe('stage_1');
    });

    it('reorders stages and offers servable runs as model options', async () => {
        wire();
        renderAt('/pipelines/pl-1');
        await waitFor(() => expect(screen.getByTestId('stage-0-name')).toHaveValue('extract'));
        await waitFor(() => expect(within(screen.getByTestId('stage-1-model')).getByText(/run #7 · v1 · toy-base/)).toBeInTheDocument());
        await userEvent.click(screen.getByTestId('stage-1-up'));
        expect(screen.getByTestId('stage-0-name')).toHaveValue('rewrite');
        expect(screen.getByTestId('stage-1-name')).toHaveValue('extract');
        apiMock.put.mockResolvedValueOnce({ data: PIPELINE });
        await userEvent.click(screen.getByTestId('pipeline-save'));
        const body = apiMock.put.mock.calls[0][1];
        expect(body.stages.map((s: { name: string }) => s.name)).toEqual(['rewrite', 'extract']);
        expect(body.stages[0].fan_out).toEqual({ pointer: '/items/*', item_key: ['/label'] });
    });
});

describe('Held-out upload, runs and results', () => {
    it('shows accepted rows and rejected rows grouped by reason with line numbers', async () => {
        wire();
        apiMock.put.mockResolvedValueOnce({ data: { accepted: 3, rejected_by_reason: { duplicate_id: [{ line: 4, id: 'x' }, { line: 9, id: 'x' }], invalid_json: [{ line: 6 }] } } });
        renderAt('/pipelines/pl-1');
        const input = await screen.findByTestId('heldout-file');
        fireEvent.change(input, { target: { files: [new File(['{}'], 'h.jsonl')] } });
        const result = await screen.findByTestId('heldout-result');
        expect(result).toHaveTextContent('3 rows accepted');
        expect(screen.getByTestId('heldout-rejected-duplicate_id')).toHaveTextContent('2 rejected: duplicate_id');
        expect(screen.getByTestId('heldout-rejected-duplicate_id')).toHaveTextContent('line 9');
        expect(screen.getByTestId('heldout-rejected-invalid_json')).toHaveTextContent('line 6');
    });

    it('runs the pipeline as a job', async () => {
        wire();
        apiMock.post.mockResolvedValueOnce({ data: { id: 41, kind: 'pipeline_eval' } });
        renderAt('/pipelines/pl-1');
        await userEvent.click(await screen.findByTestId('run-start'));
        expect(apiMock.post).toHaveBeenCalledWith('/pipelines/pl-1/runs?async_job=true', { max_rows: 50 });
        expect(await screen.findByTestId('run-started')).toHaveTextContent('job #41');
    });

    it('renders both views, headline, attribution data, provenance and the truncated notice', async () => {
        wire();
        renderAt('/pipelines/pl-1?run=r-1');
        expect(await screen.findByTestId('results-headline')).toHaveTextContent('60% fully correct end to end (6 of 10');
        expect(screen.getByTestId('results-truncated')).toBeInTheDocument();
        expect(screen.getByTestId('view-e2e-extract')).toHaveTextContent('Accuracy 75%');
        expect(screen.queryByTestId('view-isolated-extract')).toBeNull();
        expect(screen.getByTestId('view-e2e-rewrite')).toHaveTextContent('Accuracy 50%');
        expect(screen.getByTestId('view-isolated-rewrite')).toHaveTextContent('Accuracy 90%');
        expect(screen.getByTestId('view-isolated-rewrite')).toHaveTextContent('2 missed upstream');
        expect(screen.getByTestId('result-provenance-rewrite')).toHaveTextContent('project #2 Rewriter · model rewriter:run-7 · run #7');
        expect(JSON.parse(screen.getByTestId('bar-data').textContent!)).toEqual([
            { name: 'extract', count: 1 }, { name: 'rewrite', count: 3 }, { name: 'correct', count: 6 },
        ]);
        expect(screen.getByTestId('rows-table')).toHaveTextContent('wrong at rewrite');
        expect(screen.getByTestId('view-isolated-rewrite')).toHaveTextContent('2 rows skipped');
        expect(screen.getByTestId('view-isolated-rewrite')).toHaveTextContent('1 needs a judge');
        expect(screen.getByTestId('result-scoring-rewrite')).toHaveTextContent('Answer judge applies but did not run (no_judge_available)');
    });

    it('shows a failed placeholder run as a status panel with its error', async () => {
        wire((u) => (u === '/pipelines/pl-1/runs/r-9'
            ? { run_id: 'r-9', pipeline_id: 'pl-1', status: 'failed', started_at: '2026-10-27', finished_at: '2026-10-27', error: 'nothing was scored' }
            : undefined));
        renderAt('/pipelines/pl-1?run=r-9');
        const panel = await screen.findByTestId('run-status');
        expect(panel).toHaveTextContent('Run r-9 · failed');
        expect(screen.getByTestId('run-status-error')).toHaveTextContent('nothing was scored');
        expect(screen.queryByTestId('pipeline-results')).toBeNull();
    });

    it('shows a running placeholder run without results, and run status in the list', async () => {
        wire((u) => {
            if (u === '/pipelines/pl-1/runs/r-8') return { run_id: 'r-8', pipeline_id: 'pl-1', status: 'running', started_at: '2026-10-27' };
            if (u === '/pipelines/pl-1/runs') return { runs: [{ id: 'r-8', status: 'running', started_at: '2026-10-27', rows: null, share_fully_correct: null, truncated: false }] };
            return undefined;
        });
        renderAt('/pipelines/pl-1?run=r-8');
        expect(await screen.findByTestId('run-status')).toHaveTextContent('still going');
        expect(screen.queryByTestId('pipeline-results')).toBeNull();
        expect(await screen.findByTestId('run-status-r-8')).toHaveTextContent('running');
    });

    it('refreshes the runs list after starting a run', async () => {
        wire();
        apiMock.post.mockResolvedValueOnce({ data: { id: 42, kind: 'pipeline_eval' } });
        renderAt('/pipelines/pl-1');
        await userEvent.click(await screen.findByTestId('run-start'));
        await screen.findByTestId('run-started');
        const runListCalls = apiMock.get.mock.calls.filter((c) => c[0] === '/pipelines/pl-1/runs').length;
        expect(runListCalls).toBeGreaterThanOrEqual(2);
    });
});

describe('Editor validation hints', () => {
    it('keeps max tokens 0 as typed (no silent 512) and flags it', async () => {
        wire();
        renderAt('/pipelines/pl-1');
        const field = await screen.findByTestId('stage-0-max-tokens');
        await waitFor(() => expect(field).toHaveValue(512));
        await userEvent.clear(field);
        await userEvent.type(field, '0');
        expect(screen.getByTestId('stage-0-max-tokens-problem')).toHaveTextContent('1 to 4096');
        expect(screen.getByTestId('stage-0-output-hint')).toHaveTextContent('Use text for bare-label');
        apiMock.put.mockResolvedValueOnce({ data: PIPELINE });
        await userEvent.click(screen.getByTestId('pipeline-save'));
        expect(apiMock.put.mock.calls[0][1].stages[0].max_tokens).toBe(0);
    });
});

describe('Pipelines list badges', () => {
    it('flags a member project that no longer exists and a stale last run', async () => {
        wire((u) => (u === '/pipelines' ? { pipelines: [{ id: 'pl-2', name: 'gone', heldout_rows: 0,
            stages: [{ name: 'extract', project_id: 9, project_name: null, project_missing: true }],
            last_run: { id: 'r-2', started_at: null, share_fully_correct: null, status: 'stale' } }] } : undefined));
        renderAt('/pipelines');
        expect(await screen.findByTestId('pipeline-missing-pl-2-extract')).toHaveTextContent('project missing');
        expect(screen.getByTestId('pipeline-item-pl-2')).toHaveTextContent('last run stale');
    });
});

describe('Results page membership links', () => {
    it('lists pipelines for the project and hides when none', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { pipelines: [{ id: 'pl-1', name: 'toy' }] } });
        const { unmount } = render(<PipelineMembership projectId={1} />);
        const box = await screen.findByTestId('pipeline-membership');
        expect(box).toHaveTextContent('Part of pipelines: toy');
        expect(within(box).getByRole('link')).toHaveAttribute('href', '/pipelines/pl-1');
        unmount();
        apiMock.get.mockResolvedValueOnce({ data: { pipelines: [] } });
        render(<PipelineMembership projectId={2} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
        expect(screen.queryByTestId('pipeline-membership')).toBeNull();
    });
});
