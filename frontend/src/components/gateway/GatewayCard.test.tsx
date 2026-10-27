import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import GatewayCard from './GatewayCard';
import MultiLoraCard from './MultiLoraCard';

const RUNS = [
    { id: 7, name: 'findings-v1', base_model: 'Qwen/Qwen2.5-0.5B', passage_trained: false, adapter: true },
    { id: 9, name: 'findings-v2', base_model: 'Qwen/Qwen2.5-0.5B', passage_trained: true, adapter: true },
];

function settings(over: Record<string, unknown> = {}, tools: unknown[] = []) {
    return {
        settings: { upstream: 'in_process', base_url: null, upstream_kind: null, upstream_model_map: {}, default_run_id: 9, tools, ...over },
        slug: 'toy', model_ids: ['toy', 'toy:base', 'toy:run-7', 'toy:run-9'],
        base_model: 'Qwen/Qwen2.5-0.5B', default_run_resolved: 9, default_resolves_to: { kind: 'run', run_id: 9 },
        guided_json_available: true, servable_runs: RUNS,
    };
}

const SNIPPETS = {
    gateway_url: 'http://localhost:8000', model_ids: ['toy'],
    mcp_json: { mcpServers: { brewslm: { type: 'stdio', env: { BREWSLM_API_KEY: '${BREWSLM_API_KEY}' } } } },
    curl: 'curl -H "Authorization: Bearer ${BREWSLM_API_KEY}"', openai_python: 'from openai import OpenAI',
};

const KEY = { id: 3, name: 'laptop', key_prefix: 'bsk_abcd1234', project_ids: [5], rate_limit_per_minute: 60, is_active: true, created_at: null, last_used_at: null, revoked_at: null };

function wire(opts: { settings?: unknown; keys?: unknown[]; telemetry?: unknown } = {}) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url === '/gateway/keys') return { data: { keys: opts.keys ?? [KEY] } };
        if (url.endsWith('/gateway/settings')) return { data: opts.settings ?? settings() };
        if (url.endsWith('/gateway/snippets')) return { data: SNIPPETS };
        if (url.endsWith('/gateway/telemetry')) return { data: opts.telemetry ?? { summary: { by_model: [], total: 0 }, recent: [] } };
        throw new Error(`unexpected ${url}`);
    });
}

describe('GatewayCard', () => {
    beforeEach(() => {
        for (const m of Object.values(apiMock)) m.mockReset();
        Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('renders model ids with project, run and base provenance', async () => {
        wire();
        render(<GatewayCard projectId={5} />);
        const run = await screen.findByTestId('gateway-model-toy:run-7');
        expect(run).toHaveTextContent('project 5');
        expect(run).toHaveTextContent('run #7 · findings-v1 · Qwen/Qwen2.5-0.5B');
        expect(screen.getByTestId('gateway-model-toy')).toHaveTextContent('default run #9');
        expect(screen.getByTestId('gateway-model-toy:base')).toHaveTextContent('base model');
        expect(screen.getByTestId('gateway-model-toy:base')).toHaveTextContent('Qwen/Qwen2.5-0.5B');
        expect(screen.getByTestId('gateway-telemetry-empty')).toBeInTheDocument();
        expect(screen.getByTestId('gateway-guided-line')).toHaveTextContent('constrained to the project');
    });

    it('copies the snippets', async () => {
        wire();
        render(<GatewayCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('gateway-copy-curl'));
        await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith(SNIPPETS.curl));
        expect(screen.getByTestId('gateway-snippet-mcp')).toHaveTextContent('"mcpServers"');
    });

    it('shows a created key once and hides it after dismiss', async () => {
        wire();
        apiMock.post.mockResolvedValue({ data: { key: 'bsk_plaintext', record: KEY } });
        render(<GatewayCard projectId={5} />);
        fireEvent.change(await screen.findByTestId('gateway-key-name'), { target: { value: 'ci' } });
        fireEvent.click(screen.getByTestId('gateway-key-create'));
        expect(await screen.findByTestId('gateway-secret-value')).toHaveTextContent('bsk_plaintext');
        expect(apiMock.post).toHaveBeenCalledWith('/gateway/keys', { name: 'ci', project_ids: [5], rate_limit_per_minute: 60 });
        expect(screen.getByTestId('gateway-secret')).toHaveTextContent("won't see it again");
        fireEvent.click(screen.getByTestId('gateway-secret-dismiss'));
        expect(screen.queryByText('bsk_plaintext')).toBeNull();
    });

    it('revokes a key with DELETE', async () => {
        wire();
        apiMock.delete.mockResolvedValue({ data: { record: { ...KEY, is_active: false } } });
        render(<GatewayCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('gateway-revoke-3'));
        await waitFor(() => expect(apiMock.delete).toHaveBeenCalledWith('/gateway/keys/3'));
    });

    it('saves settings with PUT and renders field errors from a 400', async () => {
        wire();
        render(<GatewayCard projectId={5} />);
        fireEvent.change(await screen.findByTestId('gateway-upstream'), { target: { value: 'openai_compatible' } });
        fireEvent.change(screen.getByTestId('gateway-base-url'), { target: { value: 'nope' } });
        apiMock.put.mockRejectedValueOnce({ response: { data: { detail: 'Invalid gateway settings', errors: [{ field: 'base_url', message: 'must start with http' }] } } });
        fireEvent.click(screen.getByTestId('gateway-settings-save'));
        expect(await screen.findByTestId('gateway-error-base_url')).toHaveTextContent('must start with http');
        expect(apiMock.put).toHaveBeenCalledWith('/projects/5/gateway/settings', {
            upstream: 'openai_compatible', default_run_id: 9, base_url: 'nope', upstream_kind: 'vllm',
        });
    });

    it('reads errors from metadata.errors too', async () => {
        wire();
        render(<GatewayCard projectId={5} />);
        fireEvent.change(await screen.findByTestId('gateway-upstream'), { target: { value: 'openai_compatible' } });
        apiMock.put.mockRejectedValueOnce({ response: { data: { detail: 'bad', metadata: { errors: [{ field: 'base_url', message: 'required' }] } } } });
        fireEvent.click(screen.getByTestId('gateway-settings-save'));
        expect(await screen.findByTestId('gateway-error-base_url')).toHaveTextContent('required');
    });

    it('rejects an invalid JSON schema client-side; @recipe checkbox sets the template', async () => {
        wire();
        render(<GatewayCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('gateway-tool-add'));
        fireEvent.change(screen.getByTestId('gateway-tool-name'), { target: { value: 'detect_issues' } });
        fireEvent.change(screen.getByTestId('gateway-tool-schema'), { target: { value: '{not json' } });
        fireEvent.click(screen.getByTestId('gateway-tool-save'));
        expect(await screen.findByTestId('gateway-tool-local-error')).toBeInTheDocument();
        expect(apiMock.put).not.toHaveBeenCalled();

        fireEvent.click(screen.getByTestId('gateway-tool-recipe'));
        expect(screen.getByTestId('gateway-tool-template')).toHaveValue('@recipe');
        fireEvent.change(screen.getByTestId('gateway-tool-schema'), { target: { value: '{"type":"object"}' } });
        apiMock.put.mockResolvedValueOnce({ data: settings({}, []) });
        fireEvent.click(screen.getByTestId('gateway-tool-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        const body = apiMock.put.mock.calls[0][1] as { tools: { prompt_template: string; name: string }[] };
        expect(body.tools[0]).toMatchObject({ name: 'detect_issues', prompt_template: '@recipe' });
    });

    it('says what the bare model id resolves to when no run is servable', async () => {
        wire({ settings: { ...settings({ default_run_id: null }), servable_runs: [], model_ids: ['toy', 'toy:base'],
            default_run_resolved: null, default_resolves_to: { kind: 'base_with_retrieval', run_id: null } } });
        render(<GatewayCard projectId={5} />);
        expect(await screen.findByTestId('gateway-slug-note')).toHaveTextContent('base model + your passages');
        expect(screen.getByTestId('gateway-model-toy:base')).toHaveTextContent('Qwen/Qwen2.5-0.5B');
    });

    it('saves a tool with retrieval default (field omitted), on or off', async () => {
        wire();
        render(<GatewayCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('gateway-tool-add'));
        fireEvent.change(screen.getByTestId('gateway-tool-name'), { target: { value: 'ask' } });
        expect(screen.getByTestId('gateway-tool-retrieval')).toHaveValue('default');
        apiMock.put.mockResolvedValue({ data: settings({}, []) });
        fireEvent.click(screen.getByTestId('gateway-tool-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledTimes(1));
        expect(apiMock.put.mock.calls[0][1].tools[0]).not.toHaveProperty('retrieval');

        fireEvent.click(await screen.findByTestId('gateway-tool-add'));
        fireEvent.change(screen.getByTestId('gateway-tool-name'), { target: { value: 'ask_off' } });
        fireEvent.change(screen.getByTestId('gateway-tool-retrieval'), { target: { value: 'off' } });
        fireEvent.click(screen.getByTestId('gateway-tool-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledTimes(2));
        expect(apiMock.put.mock.calls[1][1].tools[0]).toMatchObject({ retrieval: false });
    });

    it('deletes a tool through the same PUT', async () => {
        wire({ settings: settings({}, [{ name: 'detect_issues', description: 'd', input_schema: { type: 'object' }, prompt_template: '{code}' }]) });
        apiMock.put.mockResolvedValueOnce({ data: settings() });
        render(<GatewayCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('gateway-tool-delete-detect_issues'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/5/gateway/settings', { tools: [] }));
    });
});

describe('MultiLoraCard', () => {
    beforeEach(() => {
        for (const m of Object.values(apiMock)) m.mockReset();
        Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
        wire();
    });

    it('builds a plan, starts it and routes the gateway', async () => {
        const hint = { upstream: 'openai_compatible', base_url: 'http://localhost:8001', upstream_kind: 'vllm' };
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.endsWith('multi-lora-plan')) return { data: {
                base_model: 'Qwen/Qwen2.5-0.5B', template: { command: 'python -m vllm ...' }, gateway_settings_hint: hint,
                adapters: [{ run_id: 7, name: 'findings-v1', module_name: 'toy:run-7', path: '/a/7', rank: 16 }],
            } };
            return { data: { status: 'starting' } };
        });
        apiMock.put.mockResolvedValue({ data: settings() });
        render(<MultiLoraCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('multi-lora-run-7'));
        fireEvent.click(screen.getByTestId('multi-lora-build'));
        expect(await screen.findByTestId('multi-lora-command')).toHaveTextContent('vllm');
        expect(apiMock.post).toHaveBeenCalledWith('/projects/5/serve/multi-lora-plan', { experiment_ids: [7] });
        expect(screen.getByTestId('multi-lora-plan')).toHaveTextContent('toy:run-7');
        fireEvent.click(screen.getByTestId('multi-lora-start'));
        expect(await screen.findByTestId('multi-lora-status')).toHaveTextContent('starting');
        fireEvent.click(screen.getByTestId('multi-lora-route'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/5/gateway/settings', hint));
    });

    it('shows the 400 message when runs do not share a base', async () => {
        apiMock.post.mockRejectedValue({ response: { data: { detail: 'Runs use different base models: #7 A, #9 B' } } });
        render(<MultiLoraCard projectId={5} />);
        fireEvent.click(await screen.findByTestId('multi-lora-run-7'));
        fireEvent.click(screen.getByTestId('multi-lora-build'));
        expect(await screen.findByTestId('multi-lora-error')).toHaveTextContent('different base models');
    });
});
