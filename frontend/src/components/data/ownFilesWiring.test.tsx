/**
 * Payload equivalence for the own-files import (UI simplification pass, phase 3): Data › Sources
 * (OwnFilesSourcesCard) sends exactly what Data › Split (the split form's card) sends for the same
 * server state, so moving the entry to step 1 changes no request.
 */
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../utils/workflowGraphPrefill', () => ({
    loadWorkflowStagePrefill: vi.fn().mockResolvedValue(null),
}));

import DatasetPrepPanel from './DatasetPrepPanel';
import OwnFilesSourcesCard from './OwnFilesSourcesCard';
import { wiringFromServer } from './ownFilesWiring';

const IMPORT_URL = '/projects/42/dataset/predefined-split';
const OK = {
    split_source: 'predefined', predefined_test_active: true, total_entries: 3, splits: { train: 1, val: 1, test: 1 },
    file_hashes: {}, dataset_versions: {}, prepared_version: 1, source_files: {}, duplicate_report: {},
    leak_report: { on_leak: 'fail', pairs: [], dropped_from_train: 0 },
};

const file = (name: string) => new File(['{"question":"q","answer":"a"}\n'], name, { type: 'application/json' });

interface ServerState {
    pref: Record<string, unknown> | null;
    manifest: Record<string, unknown> | null;
}

function mockServer(state: ServerState) {
    apiMock.get.mockImplementation((url: string) => {
        if (url.endsWith('/prepared-manifest')) {
            return state.manifest ? Promise.resolve({ data: state.manifest }) : Promise.reject({ response: { status: 404 } });
        }
        if (url.includes('/dataset/adapter-preference')) {
            return state.pref ? Promise.resolve({ data: state.pref }) : Promise.reject({ response: { status: 404 } });
        }
        if (url.includes('/adapters/catalog')) return Promise.resolve({ data: { adapters: {}, default_adapter: 'default-canonical' } });
        if (url.includes('/data-health')) return Promise.resolve({ data: { groups: [], overall: 'ok', severity_summary: { ok: 0, warn: 0, block: 0 }, total_signals: 0 } });
        return Promise.resolve({ data: {} });
    });
    apiMock.post.mockImplementation((url: string) => (url === IMPORT_URL ? Promise.resolve({ data: OK }) : Promise.resolve({ data: {} })));
}

function importBody(): Array<[string, string]> {
    const call = apiMock.post.mock.calls.find(([url]) => url === IMPORT_URL);
    expect(call, 'an import was posted').toBeTruthy();
    return [...(call![1] as FormData).entries()].map(([k, v]) => [k, v instanceof File ? `file:${v.name}` : String(v)]);
}

async function pickAndImport(user: ReturnType<typeof userEvent.setup>, mapping?: string) {
    await user.upload(await screen.findByTestId('pdsplit-files'), [file('train.jsonl'), file('val.jsonl'), file('test.jsonl')]);
    if (mapping) {
        await user.click(screen.getByTestId('pdsplit-options-toggle'));
        await user.type(screen.getByLabelText(/Column mapping/), mapping.replace(/[{[]/g, (c) => c + c));
    }
    await user.click(screen.getByTestId('pdsplit-submit'));
    await waitFor(() => expect(apiMock.post.mock.calls.some(([url]) => url === IMPORT_URL)).toBe(true));
}

/** The Split step: the split form mounts, hydrates, then the user opens its own-files card. */
async function bodyFromSplitStep(state: ServerState, mapping?: string) {
    mockServer(state);
    const user = userEvent.setup();
    render(<MemoryRouter><DatasetPrepPanel projectId={42} onNextStep={() => undefined} /></MemoryRouter>);
    await waitFor(() => expect(apiMock.get.mock.calls.some(([u]) => String(u).endsWith('/prepared-manifest'))).toBe(true));
    await waitFor(() => expect(apiMock.get.mock.calls.some(([u]) => String(u).includes('/adapter-preference'))).toBe(true));
    // Let both hydration GETs settle.
    await new Promise((r) => setTimeout(r, 20));
    await user.click(screen.getByTestId('split-own-files'));
    await pickAndImport(user, mapping);
    const body = importBody();
    cleanup();
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    try { window.localStorage.clear(); } catch { /* storage unavailable */ }
    return body;
}

async function bodyFromSources(state: ServerState, mapping?: string) {
    mockServer(state);
    const user = userEvent.setup();
    render(<OwnFilesSourcesCard projectId={42} onContinueToTrain={() => undefined} />);
    await pickAndImport(user, mapping);
    const body = importBody();
    cleanup();
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    try { window.localStorage.clear(); } catch { /* storage unavailable */ }
    return body;
}

const STATES: Record<string, ServerState> = {
    'fresh project (no preference, no prepared version)': { pref: null, manifest: null },
    'saved adapter preference, no prepared version': {
        pref: { adapter_id: 'qa-pair', task_profile: 'qa', adapter_config: { field_mapping: { question: 'q' } }, field_mapping: {} },
        manifest: null,
    },
    'active prepared version with explicit ratios / seed / template': {
        pref: { adapter_id: 'default-canonical', adapter_config: {}, field_mapping: {} },
        manifest: { seed: 7, chat_template: 'chatml', ratios: { train: 0.7, val: 0.15, test: 0.15 }, splits: { test: 9 } },
    },
    'active own test file, canonical resolved_split_config': {
        pref: { adapter_id: 'default-canonical', adapter_config: {}, field_mapping: {} },
        manifest: { split_source: 'predefined', predefined_test_active: true, resolved_split_config: { train_ratio: 0.8, seed: 42, chat_template: 'zephyr' }, splits: { test: 5 } },
    },
    'prepared version with a template only (profile defaults still apply)': {
        pref: { adapter_id: 'default-canonical', adapter_config: {}, field_mapping: {} },
        manifest: { chat_template: 'phi3', splits: { test: 3 } },
    },
};

describe('own-files import: Sources sends what Split sends', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    for (const [name, state] of Object.entries(STATES)) {
        it(`same body: ${name}`, async () => {
            const split = await bodyFromSplitStep(state);
            const sources = await bodyFromSources(state);
            expect(sources).toEqual(split);
        });
        it(`same body with a column mapping: ${name}`, async () => {
            const mapping = '{"question":"prompt"}';
            const split = await bodyFromSplitStep(state, mapping);
            const sources = await bodyFromSources(state, mapping);
            expect(sources).toEqual(split);
            expect(sources.find(([k]) => k === 'field_mapping')?.[1]).toBe(mapping);
        });
    }

    it('wiringFromServer: defaults with nothing on the server', () => {
        expect(wiringFromServer(null, null)).toEqual({
            chatTemplate: 'llama3', sendChatTemplate: false,
            adapter: { adapterId: 'default-canonical', taskProfile: 'auto', adapterConfig: {} },
            predefinedActive: false, activeTestRows: 0,
        });
    });

    it('Sources offers Continue to Train after a successful import', async () => {
        mockServer(STATES['fresh project (no preference, no prepared version)']);
        const onContinue = vi.fn();
        const user = userEvent.setup();
        render(<OwnFilesSourcesCard projectId={42} onContinueToTrain={onContinue} />);
        expect(screen.queryByTestId('own-files-continue-train')).toBeNull();
        await pickAndImport(user);
        const button = await screen.findByTestId('own-files-continue-train');
        // 7h follow-up (G3): recordings find it by its id as well as its test id.
        expect(document.getElementById('own-files-continue-train')).toBe(button);
        await user.click(button);
        expect(onContinue).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('own-files-continue')).toHaveTextContent('optional');
    });
});
