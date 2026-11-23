/**
 * "Score this run" (UI simplification pass, phase 2): payload equality with the long form, the model choice
 * (fine-tuned / base / both), the Advanced section, and the UI-only prefills.
 *
 * ``EvalRunForm.legacyBodies.json`` holds the request bodies the LONG form (``EvalPanel`` at e845d0a, before
 * this phase) posted for each scenario, captured by driving that form in vitest (dataset select, Model Path
 * typed, Advanced fields filled). Regenerate it with ``bash frontend/scripts/legacy-eval-capture/capture.sh``
 * (checks e845d0a out into a temporary worktree and runs ``legacyEvalBodies.capture.tsx`` there). Each test
 * below makes the same choices in the short form and asserts the posted bodies are exactly those. The only
 * intentional differences are the prefills, asserted on their own: ``max_samples`` = the set's row count
 * ("All N", sets of ≤ 2,000 rows) and ``max_new_tokens`` = the suggestion from the set's longest answer (floor
 * 128), the base result's rows / tokens when "Fine-tuned" is scored on a set that has one, plus remembered
 * last-used values.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import legacyRaw from './EvalRunForm.legacyBodies.json?raw';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import EvalRunForm from './EvalRunForm';
import { readLastUsed, suggestMaxNewTokens } from './evalRunPayload';
import type { EvalSet } from '../../api/evalSets';

type Body = Record<string, unknown>;
const LEGACY = JSON.parse(legacyRaw) as Record<string, Array<{ url: string; body: Body }>>;
const legacy = (name: string) => LEGACY[name].map((c) => c.body);
const URL_RUN = '/projects/4/evaluation/run-heldout?async_job=true';

const SETS: EvalSet[] = [{
    id: 1, slug: 'fresh', name: 'Fresh', current_version: 2, rows: 3, is_locked: false,
    versions: [{ version: 1, rows: 4 }, { version: 2, rows: 3 }],
    variants: [{ name: 'native', spec: { system: 'strip' }, label: 'system message stripped' }],
    dataset_name: 'eval_set:fresh@v2', last_evals: {},
}];
const RUN = { id: 21, name: 'exp-21', base_model: 'Org/Base-0.5B', status: 'completed' };

interface MockOptions {
    splits?: Record<string, number>;
    noManifest?: boolean;
    testJsonl?: string;
    evalSetRows?: Array<Record<string, unknown>>;
    outputSchema?: Record<string, unknown>;
    /** The lift summary's baseline (the stored base result) and that run's results. */
    lift?: Record<string, unknown>;
    baseResults?: Array<Record<string, unknown>>;
}

function mockApi(opts: MockOptions = {}) {
    apiMock.get.mockImplementation(async (url: string, config?: { params?: { offset?: number } }) => {
        if (url.endsWith('/prepared-manifest')) {
            // Default: a prepared test split of 100 rows (so the one eval set in SETS is never prefilled over it);
            // `noManifest` answers {} as the server does for a project with no prepared data.
            if (opts.noManifest) return { data: {} };
            const splits = opts.splits ?? { test: 100 };
            return { data: { splits, ...(opts.outputSchema ? { output_schema: opts.outputSchema } : {}) } };
        }
        if (url.endsWith('/evaluation/sft-lift-summary')) return { data: opts.lift ?? {} };
        if (url.endsWith('/evaluation/results/30')) return { data: opts.baseResults ?? [] };
        if (url.includes('/local-judge/serve-runs')) {
            return { data: { runs: [{ run_id: 'serve-abc123', status: 'running', template_id: 'runner.vllm' }] } };
        }
        if (url.includes('/data-studio/dataset-versions/test/export')) return { data: opts.testJsonl ?? '' };
        if (url.includes('/eval-sets/variant-support')) return { data: { experiment_id: 21, supported: true, reason: null } };
        if (url.includes('/eval-sets/fresh/rows')) {
            const offset = config?.params?.offset ?? 0;
            const rows = (opts.evalSetRows ?? []).slice(offset, offset + 200).map((row, i) => ({ position: offset + i, row }));
            return { data: { slug: 'fresh', version: 1, total: opts.evalSetRows?.length ?? 0, offset, limit: 200, rows } };
        }
        if (url.includes('/settings/null-as-absent')) return { data: { enabled: false } };
        return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: { id: 7 } });
}

interface RenderExtras {
    evalSetsLoading?: boolean;
    results?: Array<{ id: number; dataset_name: string; eval_type: string; pass_rate: number | null; metrics?: Record<string, unknown> }>;
}

function renderForm(run: typeof RUN & { config?: Record<string, unknown> } = RUN, sets: EvalSet[] = SETS, extras: RenderExtras = {}) {
    const user = userEvent.setup();
    const onQueued = vi.fn();
    const view = render(<EvalRunForm projectId={4} run={run} evalSets={sets} onQueued={onQueued} {...extras} />);
    return { user, onQueued, ...view };
}

async function openAdvanced(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByTestId('advanced-score-run-advanced-toggle'));
}

async function submitAndGetBodies(user: ReturnType<typeof userEvent.setup>, expected = 1): Promise<Body[]> {
    // Run waits for the stored-base-result check (and a remembered eval set while the sets load).
    await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
    await user.click(screen.getByTestId('eval-run-submit'));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(expected));
    for (const call of apiMock.post.mock.calls) expect(call[0]).toBe(URL_RUN);
    return apiMock.post.mock.calls.map((c) => c[1] as Body);
}

/** These blocks test the form with Fine-tuned as the starting choice: a remembered pick (phase 7c: with no pick, a
 *  run with no stored base result starts on Both; that default has its own block). */
function rememberFineTuned() {
    window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ model: 'finetuned' }));
}

async function pickEvalSetV1Native(user: ReturnType<typeof userEvent.setup>) {
    await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
    await user.selectOptions(screen.getByTestId('eval-set-version-select'), '1');
    await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
}

describe('EvalRunForm payload equality with the long form', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        mockApi();
        rememberFineTuned();
    });

    it('fine-tuned, built-in test set, nothing touched: the long form\'s default body', async () => {
        const { user, onQueued } = renderForm();
        expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked();
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_builtin_defaults'));
        expect(onQueued).toHaveBeenCalled();
    });

    it('base model (picked from the Base model radio): the long form with Model Path = the run\'s base model', async () => {
        const { user } = renderForm();
        expect(screen.getByTestId('eval-run-model-base').closest('label')).toHaveTextContent('Base model');
        expect(screen.getByTestId('eval-run-model-base').closest('label')).toHaveTextContent('Org/Base-0.5B');
        await user.click(screen.getByTestId('eval-run-model-base'));
        expect(await submitAndGetBodies(user)).toEqual(legacy('base_builtin_defaults'));
    });

    it('both: the fine-tuned body, then the base body', async () => {
        const { user } = renderForm();
        await user.click(screen.getByTestId('eval-run-model-both'));
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Run both');
        expect(await submitAndGetBodies(user, 2)).toEqual([...legacy('ft_builtin_defaults'), ...legacy('base_builtin_defaults')]);
    });

    it('named eval set (version + prompt format), fine-tuned / base / both, First 100 rows', async () => {
        for (const [choice, expected] of [
            ['finetuned', legacy('ft_evalset_v1_native')],
            ['base', legacy('base_evalset_v1_native')],
            ['both', [...legacy('ft_evalset_v1_native'), ...legacy('base_evalset_v1_native')]],
        ] as const) {
            apiMock.post.mockClear();
            const { user, unmount } = renderForm();
            await pickEvalSetV1Native(user);
            // A 4-row set prefills "All 4"; "First 100" is the long form's choice.
            expect(screen.getByTestId('eval-run-rows-all')).toBeChecked();
            await user.click(screen.getByTestId('eval-run-rows-first'));
            await user.click(screen.getByTestId(`eval-run-model-${choice}`));
            expect(await submitAndGetBodies(user, expected.length)).toEqual(expected);
            unmount();
            window.localStorage.clear();
        }
    });

    it('named eval set on its current version, no variant', async () => {
        const { user } = renderForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await user.click(screen.getByTestId('eval-run-rows-first'));
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_evalset_current'));
    });

    it('Advanced touched (validation, F1, 250 rows, 512 tokens, temperature 0.7, every turn judged, null as absent on)', async () => {
        const { user } = renderForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'validation');
        await user.clear(screen.getByTestId('eval-run-rows-first-n'));
        await user.type(screen.getByTestId('eval-run-rows-first-n'), '250');
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'f1');
        await user.clear(screen.getByTestId('eval-run-max-new-tokens'));
        await user.type(screen.getByTestId('eval-run-max-new-tokens'), '512');
        await user.clear(screen.getByTestId('eval-run-temperature'));
        await user.type(screen.getByTestId('eval-run-temperature'), '0.7');
        await user.click(screen.getByTestId('eval-dialogue-judge-toggle'));
        await user.selectOptions(screen.getByTestId('null-as-absent-run-select'), 'on');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_validation_advanced'));
    });

    it('Advanced touched + both: each leg carries the same Advanced fields (base = + model_path)', async () => {
        const { user } = renderForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'validation');
        await user.clear(screen.getByTestId('eval-run-rows-first-n'));
        await user.type(screen.getByTestId('eval-run-rows-first-n'), '250');
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'f1');
        await user.clear(screen.getByTestId('eval-run-max-new-tokens'));
        await user.type(screen.getByTestId('eval-run-max-new-tokens'), '512');
        await user.clear(screen.getByTestId('eval-run-temperature'));
        await user.type(screen.getByTestId('eval-run-temperature'), '0.7');
        await user.click(screen.getByTestId('eval-dialogue-judge-toggle'));
        await user.selectOptions(screen.getByTestId('null-as-absent-run-select'), 'on');
        await user.click(screen.getByTestId('eval-run-model-both'));
        const [ft] = legacy('ft_validation_advanced');
        expect(await submitAndGetBodies(user, 2)).toEqual([ft, { ...ft, model_path: 'Org/Base-0.5B' }]);
    });

    it('base model on the final answer key with null as absent off', async () => {
        const { user } = renderForm();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'gold_test');
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('null-as-absent-run-select'), 'off');
        await user.click(screen.getByTestId('eval-run-model-base'));
        expect(await submitAndGetBodies(user)).toEqual(legacy('base_gold_test_null_off'));
    });

    it('LLM judge on a local serve runtime with auto-start host / port (all under Advanced)', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
        await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'local_serve');
        await user.selectOptions(screen.getByLabelText('Auto Start Missing Run'), 'yes');
        await user.type(screen.getByLabelText('Export ID (auto-start)'), '12');
        await user.clear(screen.getByLabelText('Host (auto-start)'));
        await user.type(screen.getByLabelText('Host (auto-start)'), '0.0.0.0');
        await user.clear(screen.getByLabelText('Port (auto-start)'));
        await user.type(screen.getByLabelText('Port (auto-start)'), '9001');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_llm_judge_local_serve'));
    });

    it('LLM judge on a cloud provider', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
        await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'openai');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_llm_judge_openai'));
    });

    it('a raw Model path under Advanced wins: one eval, as the long form sent it', async () => {
        const { user } = renderForm();
        await user.click(screen.getByTestId('eval-run-model-both'));
        await openAdvanced(user);
        await user.type(screen.getByTestId('eval-run-model-path'), '/models/other');
        expect(screen.getByTestId('eval-run-model-raw-path')).toHaveTextContent('one eval answered by /models/other');
        expect(screen.getByTestId('eval-run-model-both')).toBeDisabled();
        expect(await submitAndGetBodies(user)).toEqual(legacy('custom_model_path'));
    });

    it('Advanced fields stay mounted while collapsed and keep sending their values', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        await user.click(screen.getByTestId('eval-dialogue-judge-toggle'));
        await openAdvanced(user); // collapse again
        expect(screen.getByTestId('advanced-score-run-advanced-toggle')).toHaveAttribute('aria-expanded', 'false');
        const [body] = await submitAndGetBodies(user);
        expect(body).toEqual({ ...legacy('ft_builtin_defaults')[0], dialogue_judge: true });
    });

    it('a baseline run offers no model choice and sends the run\'s own body', async () => {
        const { user } = renderForm({ ...RUN, config: { is_baseline: true } });
        expect(screen.queryByTestId('eval-run-model-base')).toBeNull();
        expect(screen.getByTestId('eval-run-model-baseline')).toHaveTextContent('Org/Base-0.5B');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_builtin_defaults'));
    });

    it('a run with no recorded base model disables Base / Both', () => {
        renderForm({ ...RUN, base_model: '' });
        expect(screen.getByTestId('eval-run-model-base')).toBeDisabled();
        expect(screen.getByTestId('eval-run-model-both')).toBeDisabled();
    });

    it('when the base leg fails after the fine-tuned leg was queued, says so and keeps the form open', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { id: 31 } }).mockRejectedValueOnce({ response: { data: { detail: 'no GPU' } } });
        const { user, onQueued } = renderForm();
        await user.click(screen.getByTestId('eval-run-model-both'));
        await user.click(screen.getByTestId('eval-run-submit'));
        expect(await screen.findByTestId('eval-run-error')).toHaveTextContent('Queued fine-tuned (job #31); the next eval (base model) failed: no GPU');
        expect(onQueued).not.toHaveBeenCalled();
        // Something was queued, so the choices are remembered (phase 7b, gap N2).
        expect(JSON.parse(window.localStorage.getItem('brewslm.scoreRun.4') ?? '{}').model).toBe('both');
    });

    it('LLM judge on a local serve runtime auto-started from the registry (model_id, template, no auto stop)', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
        await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'local_serve');
        await user.selectOptions(screen.getByLabelText('Auto Start Missing Run'), 'yes');
        await user.selectOptions(screen.getByLabelText('Auto Start Source'), 'registry');
        await user.type(screen.getByLabelText('Registry Model ID (auto-start)'), '7');
        await user.clear(screen.getByLabelText('Template ID (optional)'));
        await user.type(screen.getByLabelText('Template ID (optional)'), 'runner.tgi');
        await user.selectOptions(screen.getByLabelText('Auto Stop After Eval'), 'no');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_llm_judge_local_serve_registry'));
    });

    it('LLM judge on an existing local serve run', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
        await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'local_serve');
        await waitFor(() => expect(screen.getByLabelText('Serve Run')).toHaveTextContent('serve-abc123'));
        await user.selectOptions(screen.getByLabelText('Serve Run'), 'serve-abc123');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_llm_judge_local_serve_existing_run'));
    });

    it('LLM judge on Ollama and on HuggingFace (each provider\'s default judge model)', async () => {
        for (const [provider, name] of [['ollama', 'ft_llm_judge_ollama'], ['hf', 'ft_llm_judge_hf']] as const) {
            apiMock.post.mockClear();
            const { user, unmount } = renderForm();
            await openAdvanced(user);
            await user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
            await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'openai');
            await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), provider);
            expect(await submitAndGetBodies(user)).toEqual(legacy(name));
            unmount();
        }
    });

    it('when the base leg fails after the fine-tuned leg was queued, "Retry base only" posts the base body alone', async () => {
        apiMock.post
            .mockResolvedValueOnce({ data: { id: 31 } })
            .mockRejectedValueOnce({ response: { data: { detail: 'no GPU' } } })
            .mockResolvedValueOnce({ data: { id: 32 } });
        const { user, onQueued } = renderForm();
        await user.click(screen.getByTestId('eval-run-model-both'));
        await user.click(screen.getByTestId('eval-run-submit'));
        await screen.findByTestId('eval-run-error');
        await user.click(screen.getByTestId('eval-run-retry-base'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(3));
        expect(apiMock.post.mock.calls.map((c) => c[1])).toEqual([
            ...legacy('ft_builtin_defaults'), ...legacy('base_builtin_defaults'), ...legacy('base_builtin_defaults'),
        ]);
        expect(onQueued).toHaveBeenCalled();
        expect(screen.queryByTestId('eval-run-error')).toBeNull();
    });
});

describe('EvalRunForm prefills (UI only; listed differences from the long form)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        rememberFineTuned();
    });

    const JSONL = [
        { question: 'q1', answer: 'short' },
        { messages: [{ role: 'user', content: 'q2' }, { role: 'assistant', content: 'x'.repeat(70) }] },
        { question: 'q3', answer: 'mid answer' },
    ].map((r) => JSON.stringify(r)).join('\n');

    it('a ≤ 2,000-row set prefills All N rows and max new tokens from its longest answer, with where each came from', async () => {
        mockApi({ splits: { train: 26, val: 7, test: 7 }, testJsonl: JSONL });
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-run-rows-all')).toBeChecked());
        expect(screen.getByTestId('eval-run-rows-provenance')).toHaveTextContent('Test examples: 7 rows (prepared split) · all rows prefilled');
        await waitFor(() => expect(screen.getByTestId('eval-run-sends')).toHaveTextContent(
            `max ${suggestMaxNewTokens(70)} new tokens (suggested from the longest answer in this set, 70 characters)`,
        ));
        const [body] = await submitAndGetBodies(user);
        // The ONLY differences from the long form's default body: max_samples and max_new_tokens.
        expect(body).toEqual({ ...legacy('ft_builtin_defaults')[0], max_samples: 7, max_new_tokens: suggestMaxNewTokens(70) });
        // Floor 128: a 70-character answer key must not truncate a preamble or a thinking block.
        expect(suggestMaxNewTokens(70)).toBe(128);
    });

    it('the suggestion has a floor of 128 and the cap of 1,024; above the floor it follows the longest answer', () => {
        expect(suggestMaxNewTokens(0)).toBe(128);
        expect(suggestMaxNewTokens(250)).toBe(128);
        expect(suggestMaxNewTokens(600)).toBe(304);
        expect(suggestMaxNewTokens(100_000)).toBe(1024);
    });

    it('a set over 2,000 rows keeps First 100 and today\'s 128 tokens, and does not download the rows', async () => {
        mockApi({ splits: { test: 2500 }, testJsonl: JSONL });
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-run-rows-provenance')).toHaveTextContent('2,500 rows'));
        expect(screen.getByTestId('eval-run-rows-first')).toBeChecked();
        expect(screen.getByTestId('eval-run-rows-provenance')).toHaveTextContent('partial: 100 of 2,500 rows');
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('no suggestion for a set over 2,000 rows');
        expect(apiMock.get.mock.calls.some(([url]) => String(url).includes('/export'))).toBe(false);
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_builtin_defaults'));
    });

    it('an eval set reads its version\'s rows (paged) for the suggestion', async () => {
        mockApi({ evalSetRows: [{ messages: [{ role: 'user', content: 'q' }, { role: 'assistant', content: '', tool_calls: [{ name: 'f', arguments: { a: 1 } }] }] }] });
        const { user } = renderForm();
        await pickEvalSetV1Native(user);
        const toolCalls = JSON.stringify([{ name: 'f', arguments: { a: 1 } }]);
        await waitFor(() => expect(screen.getByTestId('eval-run-sends')).toHaveTextContent(`max ${suggestMaxNewTokens(toolCalls.length)} new tokens`));
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/eval-sets/fresh/rows', { params: { version: 1, offset: 0, limit: 200 } });
        const [body] = await submitAndGetBodies(user);
        expect(body).toEqual({ ...legacy('ft_evalset_v1_native')[0], max_samples: 4, max_new_tokens: suggestMaxNewTokens(toolCalls.length) });
    });

    it('review S5: a base-only run is never remembered; the remembered Fine-tuned / Both choice stays', async () => {
        mockApi({ splits: { test: 7, val: 7 }, testJsonl: JSONL });
        const first = renderForm();
        await first.user.click(screen.getByTestId('eval-run-model-both'));
        await submitAndGetBodies(first.user, 2);
        first.unmount();

        apiMock.post.mockClear();
        const second = renderForm();
        expect(screen.getByTestId('eval-run-model-both')).toBeChecked();
        await second.user.click(screen.getByTestId('eval-run-model-base'));
        await submitAndGetBodies(second.user, 1);
        second.unmount();

        apiMock.post.mockClear();
        renderForm();
        expect(screen.getByTestId('eval-run-model-both')).toBeChecked();
        // A base-only choice an earlier build stored reads as today's default.
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ model: 'base' }));
        expect(readLastUsed(4).model).toBeUndefined();
    });

    it('remembers the last-used choices per project (and a tokens value only when you set it)', async () => {
        mockApi({ splits: { test: 7, val: 7 }, testJsonl: JSONL });
        const first = renderForm();
        await first.user.selectOptions(screen.getByTestId('eval-dataset-select'), 'validation');
        await first.user.click(screen.getByTestId('eval-run-model-both'));
        await first.user.click(screen.getByTestId('eval-run-rows-first'));
        await openAdvanced(first.user);
        await first.user.clear(screen.getByTestId('eval-run-temperature'));
        await first.user.type(screen.getByTestId('eval-run-temperature'), '0.3');
        await submitAndGetBodies(first.user, 2);
        first.unmount();

        apiMock.post.mockClear();
        const second = renderForm();
        expect(screen.getByTestId('eval-dataset-select')).toHaveValue('validation');
        expect(screen.getByTestId('eval-run-model-both')).toBeChecked();
        await waitFor(() => expect(screen.getByTestId('eval-run-rows-provenance')).toHaveTextContent('your last choice'));
        expect(screen.getByTestId('eval-run-rows-first')).toBeChecked();
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('temperature 0.3');
        const bodies = await submitAndGetBodies(second.user, 2);
        expect(bodies.map((b) => [b.dataset_name, b.max_samples, b.max_new_tokens, b.temperature, b.model_path ?? ''])).toEqual([
            ['validation', 100, 128, 0.3, ''],
            ['validation', 100, 128, 0.3, 'Org/Base-0.5B'],
        ]);
    });

    it('works with storage blocked (today\'s defaults)', async () => {
        mockApi();
        const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
        const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
        try {
            const { user, onQueued } = renderForm();
            // Nothing remembered and no base result stored: the first-eval default, Both (phase 7c).
            await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
            expect(await submitAndGetBodies(user, 2)).toEqual([...legacy('ft_builtin_defaults'), ...legacy('base_builtin_defaults')]);
            expect(onQueued).toHaveBeenCalled();
        } finally {
            getItem.mockRestore();
            setItem.mockRestore();
        }
    });
});

describe('EvalRunForm comparability with the stored base result (Fine-tuned)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    const JSONL = Array.from({ length: 5 }, (_, i) => JSON.stringify({ question: `q${i}`, answer: 'a'.repeat(30) })).join('\n');
    const LIFT = {
        status: 'no_trained',
        baseline: { experiment_id: 30, experiment_name: 'Baseline', base_model: 'Org/Base-0.5B', eval_result_id: 501, dataset_name: 'test' },
        trained: null,
    };
    const BASE_RESULTS = [{
        id: 501, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.2, metrics: {},
        details: { dataset: { max_samples: 100, rows_available: 600 }, inference: { samples: 100, max_new_tokens: 256, max_new_tokens_requested: 256 } },
    }];

    it('defaults rows and max new tokens to the base result\'s and says so', async () => {
        mockApi({ splits: { test: 600 }, testJsonl: JSONL, lift: LIFT, baseResults: BASE_RESULTS });
        const { user } = renderForm();
        expect(await screen.findByTestId('eval-run-base-match')).toHaveTextContent('Matching the base result #501 (run #30): 100 rows, 256 max new tokens');
        expect(screen.getByTestId('eval-run-rows-first')).toBeChecked();
        expect(screen.getByTestId('eval-run-rows-first-n')).toHaveValue(100);
        expect(screen.getByTestId('eval-run-rows-provenance')).toHaveTextContent('matching the base result #501');
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('max 256 new tokens (matching the base result #501');
        expect(screen.queryByTestId('eval-run-base-mismatch')).toBeNull();
        const [body] = await submitAndGetBodies(user);
        expect(body).toEqual({ ...legacy('ft_builtin_defaults')[0], max_samples: 100, max_new_tokens: 256 });
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/evaluation/sft-lift-summary', { params: { experiment_id: 21 } });
    });

    it('warns when rows or tokens move away from the base\'s, and "Match the base" puts them back', async () => {
        mockApi({ splits: { test: 600 }, testJsonl: JSONL, lift: LIFT, baseResults: BASE_RESULTS });
        const { user } = renderForm();
        await screen.findByTestId('eval-run-base-match');
        await user.click(screen.getByTestId('eval-run-rows-all'));
        expect(screen.getByTestId('eval-run-base-mismatch')).toHaveTextContent(
            'The base scored 100 rows at 256 max new tokens (result #501, run #30); this eval sends 600 rows at 256. Base vs fine-tuned would compare different settings: pick Both, or match them.',
        );
        await user.click(screen.getByTestId('eval-run-base-match-button'));
        expect(screen.queryByTestId('eval-run-base-mismatch')).toBeNull();
        await openAdvanced(user);
        await user.clear(screen.getByTestId('eval-run-max-new-tokens'));
        await user.type(screen.getByTestId('eval-run-max-new-tokens'), '64');
        expect(screen.getByTestId('eval-run-base-mismatch')).toHaveTextContent('this eval sends 100 rows at 64');
        const [body] = await submitAndGetBodies(user);
        expect(body).toMatchObject({ max_samples: 100, max_new_tokens: 64 });
    });

    it('"Both" keeps the All N / suggestion prefills (both sides get the same settings) and shows no warning', async () => {
        mockApi({ splits: { test: 600 }, testJsonl: JSONL, lift: LIFT, baseResults: BASE_RESULTS });
        const { user } = renderForm();
        await screen.findByTestId('eval-run-base-match');
        await user.click(screen.getByTestId('eval-run-model-both'));
        expect(screen.getByTestId('eval-run-rows-all')).toBeChecked();
        expect(screen.queryByTestId('eval-run-base-match')).toBeNull();
        expect(screen.queryByTestId('eval-run-base-mismatch')).toBeNull();
        const bodies = await submitAndGetBodies(user, 2);
        expect(bodies.map((b) => [b.max_samples, b.max_new_tokens])).toEqual([[600, 128], [600, 128]]);
    });

    it('a base result on another split, or of another base model, is not matched', async () => {
        rememberFineTuned();
        for (const baseline of [
            { ...LIFT.baseline, dataset_name: 'validation' },
            { ...LIFT.baseline, base_model: 'Other/Model' },
        ]) {
            mockApi({ splits: { test: 600 }, testJsonl: JSONL, lift: { ...LIFT, baseline }, baseResults: BASE_RESULTS });
            const { user, unmount } = renderForm();
            await waitFor(() => expect(screen.getByTestId('eval-run-rows-all')).toBeChecked());
            const [body] = await submitAndGetBodies(user);
            expect(body).toMatchObject({ max_samples: 600, max_new_tokens: 128 });
            expect(screen.queryByTestId('eval-run-base-match')).toBeNull();
            unmount();
            apiMock.post.mockClear();
        }
    });

    it('an eval set looks up the base result on exactly that set, version and prompt format', async () => {
        mockApi({
            evalSetRows: [{ question: 'q', answer: 'a' }],
            lift: { ...LIFT, baseline: { ...LIFT.baseline, dataset_name: 'eval_set:fresh@v1+native' } },
            baseResults: [{ ...BASE_RESULTS[0], dataset_name: 'eval_set:fresh@v1+native', details: { dataset: { max_samples: 2 }, inference: { samples: 2, max_new_tokens_requested: 200 } } }],
        });
        const { user } = renderForm();
        await pickEvalSetV1Native(user);
        expect(await screen.findByTestId('eval-run-base-match')).toHaveTextContent('2 rows, 200 max new tokens');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/evaluation/sft-lift-summary', { params: { experiment_id: 21, dataset_name: 'eval_set:fresh@v1+native' } });
        const [body] = await submitAndGetBodies(user);
        expect(body).toEqual({ ...legacy('ft_evalset_v1_native')[0], max_samples: 2, max_new_tokens: 200 });
    });

    it('Run waits for the base-result check', async () => {
        mockApi({ splits: { test: 7 }, testJsonl: JSONL });
        let release: (v: unknown) => void = () => {};
        const pending = new Promise((resolve) => { release = resolve; });
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string, config?: unknown) => {
            if (url.endsWith('/evaluation/sft-lift-summary')) { await pending; return { data: {} }; }
            return base(url, config);
        });
        renderForm();
        expect(screen.getByTestId('eval-run-submit')).toBeDisabled();
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Checking for a base result');
        release(null);
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
    });
});

describe('EvalRunForm: what it says it sends', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        rememberFineTuned();
    });

    it('the prepared manifest\'s scoring mode alone does not say "Scored as" (a Q&A split is stamped field_match yet scored as free text)', async () => {
        mockApi({ outputSchema: { scoring_mode: 'field_match', required: [] } });
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
        expect(screen.queryByTestId('eval-run-scored-as')).toBeNull();
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('Also sent: metric Exact match');
        expect(await submitAndGetBodies(user)).toEqual(legacy('ft_builtin_defaults'));
    });

    it('says "Scored as" from the newest result on the run that stored a scoring mode, and drops the metric from "Also sent"', async () => {
        mockApi();
        renderForm(RUN, SETS, { results: [
            { id: 12, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5, metrics: { scoring_mode: 'tool_call' } },
            { id: 11, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5, metrics: { scoring_mode: 'field_match' } },
        ] });
        expect(await screen.findByTestId('eval-run-scored-as')).toHaveTextContent('Scored as: tool call (as result #12 on this run was');
        expect(screen.getByTestId('eval-run-sends')).not.toHaveTextContent('metric Exact match');
    });

    it('plain free-text projects keep "metric Exact match" and no Scored as line', async () => {
        mockApi();
        renderForm(RUN, SETS, { results: [{ id: 12, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5, metrics: {} }] });
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
        expect(screen.queryByTestId('eval-run-scored-as')).toBeNull();
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('Also sent: metric Exact match');
    });

    it('an LLM judge shows its provider and model on the "Also sent" line', async () => {
        mockApi();
        const { user } = renderForm();
        await openAdvanced(user);
        await user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
        await user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'openai');
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('metric LLM judge (judge OpenAI gpt-4o)');
    });

    it('never remembers the LLM judge (metric, provider, model) across runs', async () => {
        mockApi();
        const first = renderForm();
        await openAdvanced(first.user);
        await first.user.selectOptions(screen.getByTestId('eval-run-metric'), 'llm_judge');
        await first.user.selectOptions(screen.getByTestId('eval-run-judge-provider'), 'openai');
        await submitAndGetBodies(first.user);
        first.unmount();
        const stored = JSON.parse(window.localStorage.getItem('brewslm.scoreRun.4') ?? '{}');
        expect(stored).not.toHaveProperty('evalType');
        expect(stored).not.toHaveProperty('provider');
        expect(stored).not.toHaveProperty('judgeModel');

        apiMock.post.mockClear();
        // Even a value an earlier build stored is ignored.
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ ...stored, evalType: 'llm_judge', provider: 'openai', judgeModel: 'gpt-4o' }));
        const second = renderForm();
        expect(await submitAndGetBodies(second.user)).toEqual(legacy('ft_builtin_defaults'));
    });

    it('a remembered max new tokens applies to its own set only, and "use suggestion" goes back', async () => {
        const JSONL = JSON.stringify({ question: 'q', answer: 'x'.repeat(600) });
        mockApi({ splits: { test: 1, val: 1 }, testJsonl: JSONL });
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ model: 'finetuned', maxNewTokensBySet: { test: 512 } }));
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('max 512 new tokens (last used for this set'));
        await user.click(screen.getByTestId('eval-run-tokens-use-suggestion'));
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('max 304 new tokens (suggested from the longest answer in this set, 600 characters');
        expect(screen.queryByTestId('eval-run-tokens-use-suggestion')).toBeNull();
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'validation');
        expect(screen.getByTestId('eval-run-sends')).not.toHaveTextContent('last used');
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'test');
        await openAdvanced(user);
        await user.clear(screen.getByTestId('eval-run-max-new-tokens'));
        await user.type(screen.getByTestId('eval-run-max-new-tokens'), '400');
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('max 400 new tokens (set under Advanced · use suggestion (304)');
        const [body] = await submitAndGetBodies(user);
        expect(body).toMatchObject({ dataset_name: 'test', max_new_tokens: 400 });
        expect(JSON.parse(window.localStorage.getItem('brewslm.scoreRun.4') ?? '{}').maxNewTokensBySet).toEqual({ test: 400 });
    });

    it('a remembered eval set that is still loading disables Run (no fast click scores the test split)', async () => {
        mockApi();
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ datasetName: 'eval_set:fresh' }));
        const { rerender } = render(<EvalRunForm projectId={4} run={RUN} evalSets={[]} evalSetsLoading />);
        expect(screen.getByTestId('eval-run-submit')).toBeDisabled();
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Loading eval sets');
        expect(screen.getByTestId('eval-run-sets-loading')).toHaveTextContent('fresh');
        rerender(<EvalRunForm projectId={4} run={RUN} evalSets={SETS} evalSetsLoading={false} />);
        expect(screen.getByTestId('eval-dataset-select')).toHaveValue('eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
    });
});

describe('EvalRunForm phase 7b: both legs, every prompt format, cross-prompt base, memory, tool-call sizing', () => {
    const TWO_FORMATS: EvalSet[] = [{
        ...SETS[0],
        variants: [
            { name: 'native', spec: { system: 'strip' }, label: 'system message stripped' },
            { name: 'instructed', spec: { system: 'replace', text: 'Be exact.' }, label: 'system message replaced' },
        ],
    }];
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        mockApi();
        rememberFineTuned();
    });
    const summary = (bodies: Body[]) => bodies.map((b) => [b.dataset_name, b.prompt_variant ?? '', b.model_path ?? '']);

    it('"Both" posts both legs and says they run one after the other (no forced retry)', async () => {
        apiMock.post.mockResolvedValueOnce({ data: { id: 41 } }).mockResolvedValueOnce({ data: { id: 42 } });
        const { user, onQueued } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
        await user.click(screen.getByTestId('eval-run-model-both'));
        expect(screen.getByTestId('eval-run-model-both').closest('label')).toHaveTextContent('two evals, one after the other');
        expect(summary(await submitAndGetBodies(user, 2))).toEqual([
            ['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'native', 'Org/Base-0.5B'],
        ]);
        expect(onQueued).toHaveBeenCalled();
        expect(screen.queryByTestId('eval-run-error')).toBeNull();
    });

    it('"Every prompt format" with Both queues one eval per format and model in one Run', async () => {
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), '*');
        await user.click(screen.getByTestId('eval-run-model-both'));
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Run 4 evals');
        expect(screen.queryByTestId('eval-run-base-variant')).toBeNull();
        expect(summary(await submitAndGetBodies(user, 4))).toEqual([
            ['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'native', 'Org/Base-0.5B'],
            ['eval_set:fresh@v2', 'instructed', ''], ['eval_set:fresh@v2', 'instructed', 'Org/Base-0.5B'],
        ]);
    });

    it('"Base model on" another prompt format runs the cross-prompt pair in one Run (fine-tuned native, base instructed)', async () => {
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
        await user.click(screen.getByTestId('eval-run-model-both'));
        await user.selectOptions(screen.getByTestId('eval-run-base-variant'), 'instructed');
        expect(screen.getByTestId('eval-run-base-variant')).toHaveTextContent('cross-prompt');
        expect(summary(await submitAndGetBodies(user, 2))).toEqual([
            ['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'instructed', 'Org/Base-0.5B'],
        ]);
    });

    it('remembers the set, prompt format, base prompt format and model choice for the next visit', async () => {
        const first = renderForm(RUN, TWO_FORMATS);
        await first.user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await first.user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
        await first.user.click(screen.getByTestId('eval-run-model-both'));
        await first.user.selectOptions(screen.getByTestId('eval-run-base-variant'), 'instructed');
        await submitAndGetBodies(first.user, 2);
        first.unmount();

        apiMock.post.mockClear();
        const second = renderForm(RUN, TWO_FORMATS);
        expect(screen.getByTestId('eval-dataset-select')).toHaveValue('eval_set:fresh');
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('native');
        expect(screen.getByTestId('eval-run-model-both')).toBeChecked();
        expect(screen.getByTestId('eval-run-base-variant')).toHaveValue('instructed');
        expect(summary(await submitAndGetBodies(second.user, 2))).toEqual([
            ['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'instructed', 'Org/Base-0.5B'],
        ]);
    });

    it('a remembered prompt format the picked set does not have reads (and sends) "as in the rows"', async () => {
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ datasetName: 'eval_set:fresh', variant: 'gone', model: 'finetuned' }));
        const { user } = renderForm(RUN, TWO_FORMATS);
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('');
        const [body] = await submitAndGetBodies(user);
        expect(body).not.toHaveProperty('prompt_variant');
    });

    it('sizes max new tokens from a tool-call row\'s expected call (tool_name + tool_args)', async () => {
        const args = { city: 'x'.repeat(400) };
        mockApi({ evalSetRows: [{ prompt: 'p', tool_name: 'get_weather', tool_args: args }, { prompt: 'q', tool_name: null }] });
        const { user } = renderForm();
        await pickEvalSetV1Native(user);
        const call = JSON.stringify({ tool: 'get_weather', args });
        await waitFor(() => expect(screen.getByTestId('eval-run-sends')).toHaveTextContent(`max ${suggestMaxNewTokens(call.length)} new tokens (suggested from the longest answer in this set, ${call.length} characters`));
    });

    it('a set with no answer field under a scoring mode starts at 256 and says it could not size', async () => {
        mockApi({ evalSetRows: [{ prompt: 'p' }] });
        const { user } = renderForm(RUN, SETS, { results: [
            { id: 12, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5, metrics: { scoring_mode: 'tool_call' } },
        ] });
        await pickEvalSetV1Native(user);
        await waitFor(() => expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('max 256 new tokens (could not size: no answer field found in the rows; 256 for tool call scoring, check max new tokens'));
    });
});

describe('EvalRunForm phase 7c: the first eval defaults to Both; the only eval set is prefilled when unambiguous', () => {
    const TEST_LIFT = {
        status: 'no_trained',
        baseline: { experiment_id: 30, experiment_name: 'Baseline', base_model: 'Org/Base-0.5B', eval_result_id: 501, dataset_name: 'test' },
        trained: null,
    };
    const TEST_BASE_RESULTS = [{
        id: 501, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.2, metrics: {},
        details: { dataset: { max_samples: 100, rows_available: 100 }, inference: { samples: 100, max_new_tokens: 128, max_new_tokens_requested: 128 } },
    }];
    const TWO_FORMATS: EvalSet[] = [{
        ...SETS[0],
        variants: [
            { name: 'native', spec: { system: 'strip' }, label: 'system message stripped' },
            { name: 'instructed', spec: { system: 'replace', text: 'Be exact.' }, label: 'system message replaced' },
        ],
    }];
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('no choice picked and no base-model score on the set: Both, says why and that it means two evals; the default is not remembered', async () => {
        mockApi();
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        const note = screen.getByTestId('eval-run-model-default-both');
        expect(note).toHaveTextContent('Both is the default for a first eval: no base-model score on this set yet');
        expect(note).toHaveTextContent('two evals, one after the other');
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Run both');
        expect(await submitAndGetBodies(user, 2)).toEqual([...legacy('ft_builtin_defaults'), ...legacy('base_builtin_defaults')]);
        expect(JSON.parse(window.localStorage.getItem('brewslm.scoreRun.4') ?? '{}')).not.toHaveProperty('model');
    });

    it('a base-model score already stored on the set: stays on Fine-tuned (matching it), no default note', async () => {
        mockApi({ lift: TEST_LIFT, baseResults: TEST_BASE_RESULTS });
        const { user } = renderForm();
        expect(await screen.findByTestId('eval-run-base-match')).toBeInTheDocument();
        expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked();
        expect(screen.queryByTestId('eval-run-model-default-both')).toBeNull();
        expect(await submitAndGetBodies(user, 1)).toEqual(legacy('ft_builtin_defaults'));
    });

    it('a stored base result that names no rows / tokens still counts: Fine-tuned, not Both', async () => {
        mockApi({ lift: TEST_LIFT, baseResults: [{ id: 501, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.2, metrics: {}, details: {} }] });
        const { user } = renderForm();
        expect(await submitAndGetBodies(user, 1)).toEqual(legacy('ft_builtin_defaults'));
        expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked();
    });

    it('the base-result check failed: stays on Fine-tuned (an unknown is not "no score")', async () => {
        mockApi();
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string, config?: unknown) => {
            if (url.endsWith('/evaluation/sft-lift-summary')) throw new Error('down');
            return base(url, config);
        });
        const { user } = renderForm();
        expect(await screen.findByTestId('eval-run-base-unknown')).toBeInTheDocument();
        expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked();
        expect(await submitAndGetBodies(user, 1)).toEqual(legacy('ft_builtin_defaults'));
    });

    it('a remembered choice wins over the default', async () => {
        mockApi();
        rememberFineTuned();
        const { user } = renderForm();
        expect(await submitAndGetBodies(user, 1)).toEqual(legacy('ft_builtin_defaults'));
        expect(screen.queryByTestId('eval-run-model-default-both')).toBeNull();
    });

    it('picking Fine-tuned over the default runs one eval and is remembered', async () => {
        mockApi();
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        await user.click(screen.getByTestId('eval-run-model-finetuned'));
        expect(screen.queryByTestId('eval-run-model-default-both')).toBeNull();
        expect(await submitAndGetBodies(user, 1)).toEqual(legacy('ft_builtin_defaults'));
        expect(JSON.parse(window.localStorage.getItem('brewslm.scoreRun.4') ?? '{}').model).toBe('finetuned');
    });

    it('a baseline run gets no default (it is the base model)', async () => {
        mockApi();
        renderForm({ ...RUN, config: { is_baseline: true } });
        expect(screen.getByTestId('eval-run-model-baseline')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
        expect(screen.queryByTestId('eval-run-model-default-both')).toBeNull();
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent(/^Run$/);
    });

    it('per set + prompt format: Both where the base has no score on that format, Fine-tuned where it has one', async () => {
        mockApi({ evalSetRows: [{ question: 'q', answer: 'a' }] });
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string, config?: { params?: { dataset_name?: string } }) => {
            if (url.endsWith('/evaluation/sft-lift-summary')) {
                const name = config?.params?.dataset_name;
                if (name === 'eval_set:fresh@v2+native') {
                    return { data: { ...TEST_LIFT, baseline: { ...TEST_LIFT.baseline, dataset_name: name } } };
                }
                return { data: {} };
            }
            if (url.endsWith('/evaluation/results/30')) {
                return { data: [{ ...TEST_BASE_RESULTS[0], dataset_name: 'eval_set:fresh@v2+native' }] };
            }
            return base(url, config);
        });
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'instructed');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        expect(screen.getByTestId('eval-run-model-default-both')).toHaveTextContent('no base-model score on this set and prompt format yet');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked());
        expect(screen.queryByTestId('eval-run-model-default-both')).toBeNull();
    });

    it('prefills the only eval set when there is no prepared test split, and says why', async () => {
        mockApi({ noManifest: true, evalSetRows: [{ question: 'q', answer: 'a' }] });
        rememberFineTuned();
        const { user } = renderForm();
        await waitFor(() => expect(screen.getByTestId('eval-dataset-select')).toHaveValue('eval_set:fresh'));
        expect(screen.getByTestId('eval-run-set-prefill')).toHaveTextContent('Prefilled: Fresh is the only eval set in this project, and there is no prepared test split.');
        const [body] = await submitAndGetBodies(user, 1);
        expect(body.dataset_name).toBe('eval_set:fresh@v2');
    });

    it('never prefills over a prepared test split, with two eval sets, over a remembered set, or when the manifest is unknown', async () => {
        const second: EvalSet = { ...SETS[0], id: 2, slug: 'other', name: 'Other', dataset_name: 'eval_set:other@v2' };
        const cases: Array<{ opts: MockOptions; sets: EvalSet[]; remembered?: { model: string; datasetName?: string }; failManifest?: boolean }> = [
            { opts: { splits: { train: 20, test: 5 } }, sets: SETS },
            { opts: { noManifest: true }, sets: [...SETS, second] },
            { opts: { noManifest: true }, sets: SETS, remembered: { model: 'finetuned', datasetName: 'validation' } },
            { opts: {}, sets: SETS, failManifest: true },
        ];
        for (const c of cases) {
            mockApi(c.opts);
            if (c.failManifest) {
                const base = apiMock.get.getMockImplementation()!;
                apiMock.get.mockImplementation(async (url: string, config?: unknown) => {
                    if (url.endsWith('/prepared-manifest')) throw new Error('down');
                    return base(url, config);
                });
            }
            window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify(c.remembered ?? { model: 'finetuned' }));
            const { unmount } = renderForm(RUN, c.sets);
            await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
            expect(screen.getByTestId('eval-dataset-select')).toHaveValue(c.remembered?.datasetName ?? 'test');
            expect(screen.queryByTestId('eval-run-set-prefill')).toBeNull();
            unmount();
            window.localStorage.clear();
        }
    });
});

describe('EvalRunForm phase 7e: Every prompt format keeps Both and is the default for a set with several formats', () => {
    const TWO_FORMATS: EvalSet[] = [{
        ...SETS[0],
        variants: [
            { name: 'native', spec: { system: 'strip' }, label: 'system message stripped' },
            { name: 'instructed', spec: { system: 'replace', text: 'Be exact.' }, label: 'system message replaced' },
        ],
    }];
    const BASE = 'Org/Base-0.5B';
    const summary = (bodies: Body[]) => bodies.map((b) => [b.dataset_name, b.prompt_variant ?? '', b.model_path ?? '']);
    const EVERY_BOTH = [
        ['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'native', BASE],
        ['eval_set:fresh@v2', 'instructed', ''], ['eval_set:fresh@v2', 'instructed', BASE],
    ];
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        mockApi({ evalSetRows: [{ question: 'q', answer: 'a' }] });
    });

    it('a set with 2 prompt formats starts on Every prompt format + Both: one Run queues base + fine-tuned per format, says where the default came from, and does not remember it', async () => {
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        // The default waits for the variant-support answer (review S3).
        await waitFor(() => expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('*'));
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        expect(screen.getByTestId('eval-run-model-default-both')).toHaveTextContent('no base-model score on any prompt format of this set yet');
        expect(screen.getByTestId('eval-run-model-default-both')).toHaveTextContent('this runs 4 evals');
        expect(screen.getByTestId('eval-run-model-default-both')).toHaveTextContent('Pick Fine-tuned for 2 evals.');
        expect(screen.getByTestId('eval-run-rows-provenance')).toHaveTextContent(
            'Every prompt format is the default for a set with 2 prompt formats (native, instructed); pick one for fewer evals');
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Run 4 evals: fine-tuned + base, 2 prompt formats');
        expect(summary(await submitAndGetBodies(user, 4))).toEqual(EVERY_BOTH);
        const stored = readLastUsed(4);
        expect(stored.variant).toBeUndefined();
        expect(stored.model).toBeUndefined();
    });

    it('picking Every prompt format keeps the first-eval Both default (gap G1: it used to drop to Fine-tuned)', async () => {
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ variant: 'native' }));
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('native');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), '*');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        // Picked, not defaulted: no "default" provenance on the rows line.
        expect(screen.getByTestId('eval-run-rows-provenance')).not.toHaveTextContent('is the default for a set');
        expect(summary(await submitAndGetBodies(user, 4))).toEqual(EVERY_BOTH);
        expect(readLastUsed(4).variant).toBe('*');
    });

    it('the Run label says what is queued: Fine-tuned or Base model under Every prompt format is not "Run both"', async () => {
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        await user.click(screen.getByTestId('eval-run-model-finetuned'));
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Run 2 evals: fine-tuned, 2 prompt formats');
        await user.click(screen.getByTestId('eval-run-model-base'));
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Run 2 evals: base model, 2 prompt formats');
        await user.selectOptions(screen.getByTestId('eval-set-variant-select'), 'native');
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent(/^Run$/);
        await user.click(screen.getByTestId('eval-run-model-both'));
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent(/^Run both$/);
        expect(summary(await submitAndGetBodies(user, 2))).toEqual([['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'native', BASE]]);
    });

    it('a single-format set is unchanged: as in the rows, Both by default, two evals, "Run both"', async () => {
        const { user } = renderForm(RUN, SETS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        expect(screen.getByTestId('eval-run-rows-provenance')).not.toHaveTextContent('Every prompt format');
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent(/^Run both$/);
        expect(summary(await submitAndGetBodies(user, 2))).toEqual([['eval_set:fresh@v2', '', ''], ['eval_set:fresh@v2', '', BASE]]);
    });

    it('a remembered prompt format and model choice beat the new default', async () => {
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ datasetName: 'eval_set:fresh', variant: 'instructed', model: 'finetuned' }));
        const { user } = renderForm(RUN, TWO_FORMATS);
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('instructed');
        expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked();
        expect(screen.getByTestId('eval-run-rows-provenance')).not.toHaveTextContent('is the default for a set');
        expect(summary(await submitAndGetBodies(user, 1))).toEqual([['eval_set:fresh@v2', 'instructed', '']]);
    });

    it('a remembered "as in the rows" also beats the new default', async () => {
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ datasetName: 'eval_set:fresh', variant: '' }));
        renderForm(RUN, TWO_FORMATS);
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        expect(screen.getByTestId('eval-run-submit')).toHaveTextContent(/^Run both$/);
    });

    it('under Every prompt format, a base result on any format means Fine-tuned (no base re-run), and Run waits for every check', async () => {
        const base = apiMock.get.getMockImplementation()!;
        let release: () => void = () => {};
        const gate = new Promise<void>((resolve) => { release = resolve; });
        apiMock.get.mockImplementation(async (url: string, config?: { params?: { dataset_name?: string } }) => {
            if (url.endsWith('/evaluation/sft-lift-summary')) {
                const name = config?.params?.dataset_name;
                if (name === 'eval_set:fresh@v2+instructed') {
                    await gate;
                    return { data: { status: 'no_trained', trained: null, baseline: {
                        experiment_id: 30, base_model: BASE, eval_result_id: 501, dataset_name: name } } };
                }
                return { data: {} };
            }
            if (url.endsWith('/evaluation/results/30')) {
                return { data: [{ id: 501, dataset_name: 'eval_set:fresh@v2+instructed', eval_type: 'exact_match', pass_rate: 0.2, metrics: {},
                    details: { dataset: { max_samples: 3 }, inference: { samples: 3, max_new_tokens: 128 } } }] };
            }
            return base(url, config);
        });
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toHaveTextContent('Checking for a base result'));
        expect(screen.getByTestId('eval-run-submit')).toBeDisabled();
        release();
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
        expect(screen.getByTestId('eval-run-model-finetuned')).toBeChecked();
        expect(screen.queryByTestId('eval-run-model-default-both')).toBeNull();
        expect(summary(await submitAndGetBodies(user, 2))).toEqual([['eval_set:fresh@v2', 'native', ''], ['eval_set:fresh@v2', 'instructed', '']]);
    });
    it('review S2: each base-result lookup runs once (a lookup that lands does not cancel and refetch the others)', async () => {
        const base = apiMock.get.getMockImplementation()!;
        // The lookups land one after the other (the order that used to drop and refetch the rest).
        const delay: Record<string, number> = { 'eval_set:fresh@v2': 5, 'eval_set:fresh@v2+native': 25, 'eval_set:fresh@v2+instructed': 45 };
        apiMock.get.mockImplementation(async (url: string, config?: { params?: { dataset_name?: string } }) => {
            if (url.endsWith('/evaluation/sft-lift-summary')) {
                await new Promise((r) => setTimeout(r, delay[config?.params?.dataset_name ?? ''] ?? 0));
                return { data: {} };
            }
            return base(url, config);
        });
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-run-model-both')).toBeChecked());
        await new Promise((r) => setTimeout(r, 80));
        const lifts = apiMock.get.mock.calls
            .filter(([url]) => String(url).endsWith('/evaluation/sft-lift-summary'))
            .map(([, cfg]) => (cfg as { params?: { dataset_name?: string } } | undefined)?.params?.dataset_name ?? '');
        const counts = lifts.reduce<Record<string, number>>((acc, k) => ({ ...acc, [k]: (acc[k] ?? 0) + 1 }), {});
        expect(counts['eval_set:fresh@v2+native']).toBe(1);
        expect(counts['eval_set:fresh@v2+instructed']).toBe(1);
        expect(counts['eval_set:fresh@v2']).toBe(1);
        expect(summary(await submitAndGetBodies(user, 4))).toEqual(EVERY_BOTH);
    });

    it('review S3: until the server says this run takes prompt formats, there is no Every default (Run queues the rows as they are)', async () => {
        const base = apiMock.get.getMockImplementation()!;
        let answer: () => void = () => {};
        const support = new Promise<void>((resolve) => { answer = resolve; });
        apiMock.get.mockImplementation(async (url: string, config?: unknown) => {
            if (url.includes('/eval-sets/variant-support')) {
                await support;
                return { data: { experiment_id: 21, supported: true, reason: null } };
            }
            return base(url, config);
        });
        window.localStorage.setItem('brewslm.scoreRun.4', JSON.stringify({ model: 'finetuned' }));
        const { user, unmount } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
        expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('');
        expect(screen.getByTestId('eval-run-rows-provenance')).not.toHaveTextContent('is the default for a set');
        expect(summary(await submitAndGetBodies(user, 1))).toEqual([['eval_set:fresh@v2', '', '']]);
        unmount();
        // Once it answers "supported", the default applies.
        apiMock.post.mockClear();
        answer();
        renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-set-variant-select')).toHaveValue('*'));
    });

    it('review S3: a run that renders no system message never gets the Every default', async () => {
        const base = apiMock.get.getMockImplementation()!;
        apiMock.get.mockImplementation(async (url: string, config?: unknown) => {
            if (url.includes('/eval-sets/variant-support')) return { data: { experiment_id: 21, supported: false, reason: 'No system message.' } };
            return base(url, config);
        });
        const { user } = renderForm(RUN, TWO_FORMATS);
        await user.selectOptions(screen.getByTestId('eval-dataset-select'), 'eval_set:fresh');
        await waitFor(() => expect(screen.getByTestId('eval-run-submit')).toBeEnabled());
        await new Promise((r) => setTimeout(r, 30));
        expect(screen.getByTestId('eval-run-rows-provenance')).not.toHaveTextContent('is the default for a set');
        expect(summary(await submitAndGetBodies(user, 2))).toEqual([['eval_set:fresh@v2', '', ''], ['eval_set:fresh@v2', '', BASE]]);
    });
});

describe('EvalRunForm gap G7: stop a prompt format the model never answers in (opt-in)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        mockApi();
        rememberFineTuned();
    });

    it('is off by default: the body is the long form\'s, with no stop_if_unanswered_after', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        expect(screen.getByTestId('eval-run-stop-unanswered')).not.toBeChecked();
        expect(screen.getByTestId('eval-run-stop-unanswered-hint')).toHaveTextContent('JSON / tool-call scoring only');
        const [body] = await submitAndGetBodies(user);
        expect(body).toEqual(legacy('ft_builtin_defaults')[0]);
        expect(body).not.toHaveProperty('stop_if_unanswered_after');
    });

    it('ticked: every leg sends stop_if_unanswered_after = 50 and "Also sent" says so', async () => {
        const { user } = renderForm();
        await openAdvanced(user);
        await user.click(screen.getByTestId('eval-run-stop-unanswered'));
        expect(screen.getByTestId('eval-run-sends')).toHaveTextContent('stops a prompt format with no parsable answer after 50 rows');
        await user.click(screen.getByTestId('eval-run-model-both'));
        const [ft] = legacy('ft_builtin_defaults');
        expect(await submitAndGetBodies(user, 2)).toEqual([
            { ...ft, stop_if_unanswered_after: 50 },
            { ...ft, model_path: 'Org/Base-0.5B', stop_if_unanswered_after: 50 },
        ]);
    });

    it('says it applies when the run is scored by a JSON / tool-call scoring mode', async () => {
        renderForm(RUN, SETS, { results: [
            { id: 12, dataset_name: 'test', eval_type: 'exact_match', pass_rate: 0.5, metrics: { scoring_mode: 'tool_call' } },
        ] });
        expect(await screen.findByTestId('eval-run-stop-unanswered-hint')).toHaveTextContent('Applies here: this run is scored as tool call.');
    });
});
