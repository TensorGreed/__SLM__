/**
 * "Score this run" (UI simplification pass, phase 2): payload equality with the long form, the model choice
 * (fine-tuned / base / both), the Advanced section, and the UI-only prefills.
 *
 * ``EvalRunForm.legacyBodies.json`` holds the request bodies the LONG form (``EvalPanel`` at e845d0a, before
 * this phase) posted for each scenario, captured by driving that form in vitest (dataset select, Model Path
 * typed, Advanced fields filled). Each test below makes the same choices in the short form and asserts the
 * posted bodies are exactly those. The only intentional differences are the prefills, asserted on their own:
 * ``max_samples`` = the set's row count ("All N", sets of ≤ 2,000 rows) and ``max_new_tokens`` = the
 * suggestion from the set's longest answer, plus remembered last-used values.
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
import { suggestMaxNewTokens } from './evalRunPayload';
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
    testJsonl?: string;
    evalSetRows?: Array<Record<string, unknown>>;
}

function mockApi(opts: MockOptions = {}) {
    apiMock.get.mockImplementation(async (url: string, config?: { params?: { offset?: number } }) => {
        if (url.endsWith('/prepared-manifest')) return { data: opts.splits ? { splits: opts.splits } : {} };
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

function renderForm(run: typeof RUN & { config?: Record<string, unknown> } = RUN, sets: EvalSet[] = SETS) {
    const user = userEvent.setup();
    const onQueued = vi.fn();
    const view = render(<EvalRunForm projectId={4} run={run} evalSets={sets} onQueued={onQueued} />);
    return { user, onQueued, ...view };
}

async function openAdvanced(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByTestId('advanced-score-run-advanced-toggle'));
}

async function submitAndGetBodies(user: ReturnType<typeof userEvent.setup>, expected = 1): Promise<Body[]> {
    await user.click(screen.getByTestId('eval-run-submit'));
    await waitFor(() => expect(apiMock.post).toHaveBeenCalledTimes(expected));
    for (const call of apiMock.post.mock.calls) expect(call[0]).toBe(URL_RUN);
    return apiMock.post.mock.calls.map((c) => c[1] as Body);
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
        expect(await screen.findByTestId('eval-run-error')).toHaveTextContent('The fine-tuned eval was queued (job #31); the base-model eval failed: no GPU');
        expect(onQueued).not.toHaveBeenCalled();
    });
});

describe('EvalRunForm prefills (UI only; listed differences from the long form)', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
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
        expect(suggestMaxNewTokens(70)).toBe(48);
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
            expect(await submitAndGetBodies(user)).toEqual(legacy('ft_builtin_defaults'));
            expect(onQueued).toHaveBeenCalled();
        } finally {
            getItem.mockRestore();
            setItem.mockRestore();
        }
    });
});
