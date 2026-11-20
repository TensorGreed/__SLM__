/**
 * Capture harness for ``src/components/evaluation/EvalRunForm.legacyBodies.json`` (UI simplification pass,
 * phase 2): drives the LONG eval form (``EvalPanel`` at e845d0a, before the short "Score this run" form) and
 * records the ``run-heldout`` bodies it posts for each scenario.
 *
 * Not a test of the current tree: ``capture.sh`` (next to this file) checks e845d0a out into a temporary git
 * worktree, copies this file into it as a vitest file and runs it there, writing the JSON to
 * ``LEGACY_CAPTURE_OUT``. The file name keeps it out of this tree's vitest run and ``tsc -b`` (only ``src``).
 * Re-run after adding a scenario: ``bash frontend/scripts/legacy-eval-capture/capture.sh``.
 */

import { writeFileSync } from 'node:fs';

import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalPanel from './EvalPanel';
import { useProjectStore } from '../../stores/projectStore';

type User = ReturnType<typeof userEvent.setup>;

const SETS = [{
    id: 1, slug: 'fresh', name: 'Fresh', current_version: 2, rows: 3, is_locked: false,
    versions: [{ version: 1, rows: 4 }, { version: 2, rows: 3 }],
    variants: [{ name: 'native', spec: { system: 'strip' }, label: 'system message stripped' }],
    dataset_name: 'eval_set:fresh@v2', last_evals: {},
}];
const BASE_MODEL = 'Org/Base-0.5B';
const RUN_URL = '/projects/4/evaluation/run-heldout?async_job=true';

function mockApi() {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/eval-sets/variant-support')) return { data: { experiment_id: 21, supported: true, reason: null } };
        if (url.includes('/eval-sets')) return { data: { eval_sets: SETS } };
        if (url.includes('/training/experiments')) return { data: [{ id: 21, name: 'exp-21', base_model: BASE_MODEL, status: 'completed' }] };
        if (url.includes('/evaluation/results/21')) return { data: [] };
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
        if (url.includes('/evaluation/pack-preference')) {
            return { data: { preferred_pack_id: null, active_pack_id: null, active_pack_source: 'default', active_pack: { display_name: 'Auto' } } };
        }
        if (url.includes('/local-judge/serve-runs')) {
            return { data: { runs: [{ run_id: 'serve-abc123', status: 'running', template_id: 'runner.vllm' }] } };
        }
        if (url.includes('/settings/null-as-absent')) return { data: { enabled: false } };
        if (url.includes('/evaluation/safety-scorecard/21')) return { data: { overall_risk: 'low', red_flags: [] } };
        if (url.includes('/evaluation/gates/21')) {
            return { data: { captured_at: '2026-03-26T00:00:00Z', passed: true, failed_gate_ids: [], missing_required_metrics: [], checks: [] } };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return { data: { experiment_id: 21, is_ship: true, decision: 'SHIP', reasons: [], failed_gates: [], missing_metrics: [],
                gate_report: { passed: true, checks: [], missing_required_metrics: [], failed_gate_ids: [] } } };
        }
        if (url.includes('remediation-plans')) return { data: { project_id: 4, count: 0, plans: [] } };
        if (url.includes('/gold/review')) return { data: { total: 0, rows: [] } };
        return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: { id: 7 } });
}

const captured: Record<string, Array<{ url: string; body: unknown }>> = {};

async function openForm(): Promise<User> {
    const user = userEvent.setup();
    render(<MemoryRouter><EvalPanel projectId={4} workflow="verdict" /></MemoryRouter>);
    await user.click(await screen.findByRole('button', { name: /exp-21/ }));
    await user.click(await screen.findByRole('button', { name: /Evaluate on test examples/ }));
    await waitFor(() => expect(screen.getByTestId('eval-dataset-select')).toHaveTextContent('Fresh · v2 · 3 rows'));
    return user;
}

/** The long form's labels have no ``htmlFor``: the field is the input / select in the label's form group. */
function field(label: string): HTMLInputElement | HTMLSelectElement {
    const el = screen.getAllByText(label, { selector: 'label' })[0];
    const control = el.closest('.form-group')?.querySelector('input, select');
    if (!control) throw new Error(`no control for ${label}`);
    return control as HTMLInputElement | HTMLSelectElement;
}

async function fill(user: User, label: string, value: string) {
    const el = field(label);
    await user.clear(el);
    await user.type(el, value);
}

async function run(user: User, name: string) {
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
    captured[name] = apiMock.post.mock.calls
        .filter(([url]) => url === RUN_URL)
        .map(([url, body]) => ({ url: String(url), body }));
    expect(captured[name].length).toBe(1);
}

async function pickEvalSetV1Native(user: User) {
    await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
    await user.selectOptions(screen.getByTestId('eval-set-version-select'), '1');
    await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
}

async function judge(user: User, provider: string) {
    await user.selectOptions(field('Metric'), 'llm_judge');
    await user.selectOptions(field('Judge Provider'), provider);
}

describe('legacy eval form bodies (capture)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        useProjectStore.setState({ activeProject: null });
        mockApi();
    });

    afterAll(() => {
        const out = process.env.LEGACY_CAPTURE_OUT;
        if (!out) throw new Error('LEGACY_CAPTURE_OUT is not set');
        writeFileSync(out, `${JSON.stringify(captured, null, 2)}\n`);
    });

    it('ft_builtin_defaults', async () => {
        const user = await openForm();
        await run(user, 'ft_builtin_defaults');
    });

    it('base_builtin_defaults', async () => {
        const user = await openForm();
        await user.type(field('Model Path (optional)'), BASE_MODEL);
        await run(user, 'base_builtin_defaults');
    });

    it('ft_evalset_v1_native', async () => {
        const user = await openForm();
        await pickEvalSetV1Native(user);
        await run(user, 'ft_evalset_v1_native');
    });

    it('base_evalset_v1_native', async () => {
        const user = await openForm();
        await pickEvalSetV1Native(user);
        await user.type(field('Model Path (optional)'), BASE_MODEL);
        await run(user, 'base_evalset_v1_native');
    });

    it('ft_evalset_current', async () => {
        const user = await openForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await run(user, 'ft_evalset_current');
    });

    it('ft_validation_advanced', async () => {
        const user = await openForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'validation');
        await user.selectOptions(field('Metric'), 'f1');
        await fill(user, 'Max Samples', '250');
        await fill(user, 'Max New Tokens', '512');
        await fill(user, 'Temperature', '0.7');
        await user.click(screen.getByTestId('eval-dialogue-judge-toggle'));
        await user.selectOptions(screen.getByTestId('null-as-absent-run-select'), 'on');
        await run(user, 'ft_validation_advanced');
    });

    it('base_gold_test_null_off', async () => {
        const user = await openForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'gold_test');
        await user.selectOptions(screen.getByTestId('null-as-absent-run-select'), 'off');
        await user.type(field('Model Path (optional)'), BASE_MODEL);
        await run(user, 'base_gold_test_null_off');
    });

    it('ft_llm_judge_local_serve', async () => {
        const user = await openForm();
        await judge(user, 'local_serve');
        await user.selectOptions(field('Auto Start Missing Run'), 'yes');
        await user.type(field('Export ID (auto-start)'), '12');
        await fill(user, 'Host (auto-start)', '0.0.0.0');
        await fill(user, 'Port (auto-start)', '9001');
        await run(user, 'ft_llm_judge_local_serve');
    });

    it('ft_llm_judge_openai', async () => {
        const user = await openForm();
        await judge(user, 'openai');
        await run(user, 'ft_llm_judge_openai');
    });

    it('custom_model_path', async () => {
        const user = await openForm();
        await user.type(field('Model Path (optional)'), '/models/other');
        await run(user, 'custom_model_path');
    });

    it('ft_llm_judge_local_serve_registry', async () => {
        const user = await openForm();
        await judge(user, 'local_serve');
        await user.selectOptions(field('Auto Start Missing Run'), 'yes');
        await user.selectOptions(field('Auto Start Source'), 'registry');
        await user.type(field('Registry Model ID (auto-start)'), '7');
        await fill(user, 'Template ID (optional)', 'runner.tgi');
        await user.selectOptions(field('Auto Stop After Eval'), 'no');
        await run(user, 'ft_llm_judge_local_serve_registry');
    });

    it('ft_llm_judge_local_serve_existing_run', async () => {
        const user = await openForm();
        await judge(user, 'local_serve');
        await waitFor(() => expect(field('Serve Run')).toHaveTextContent('serve-abc123'));
        await user.selectOptions(field('Serve Run'), 'serve-abc123');
        await run(user, 'ft_llm_judge_local_serve_existing_run');
    });

    it('ft_llm_judge_ollama', async () => {
        const user = await openForm();
        await judge(user, 'ollama');
        await run(user, 'ft_llm_judge_ollama');
    });

    it('ft_llm_judge_hf', async () => {
        const user = await openForm();
        // HuggingFace is the default provider: pick another, then come back (the model resets to its default).
        await judge(user, 'openai');
        await user.selectOptions(field('Judge Provider'), 'hf');
        await run(user, 'ft_llm_judge_hf');
    });
});
