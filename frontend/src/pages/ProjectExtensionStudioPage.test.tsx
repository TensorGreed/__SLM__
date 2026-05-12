import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type {
    ExtensionListResponse,
    PluginContractReport,
    ReloadResponse,
    ScaffoldResponse,
} from '../types/extensions';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../api/client', () => ({ default: apiMock }));

import ProjectExtensionStudioPage from './ProjectExtensionStudioPage';

const CATALOG: ExtensionListResponse = {
    known_kinds: ['data_adapter', 'training_runtime', 'domain_pack', 'eval_pack'],
    kinds: [
        {
            kind: 'data_adapter',
            contract_version: 'slm.data_adapter/v3',
            supports_safe_reload: true,
            has_module_loader: true,
            settings_key: 'DATA_ADAPTER_PLUGIN_MODULES',
            configured_modules: [],
            loaded_modules: ['example.adapters.builtin'],
            load_errors: {},
            registered_count: 4,
            recognized_exports: [
                'register_data_adapters',
                'get_data_adapters',
                'DATA_ADAPTERS',
            ],
        },
        {
            kind: 'training_runtime',
            contract_version: 'slm.training_runtime/v1',
            supports_safe_reload: true,
            has_module_loader: true,
            settings_key: 'TRAINING_RUNTIME_PLUGIN_MODULES',
            configured_modules: [],
            loaded_modules: [],
            load_errors: { 'bad.module': 'ImportError: nope' },
            registered_count: 1,
            recognized_exports: ['register_training_runtime_plugins'],
        },
        {
            kind: 'domain_pack',
            contract_version: 'slm.domain-pack/v1',
            supports_safe_reload: false,
            has_module_loader: false,
            settings_key: null,
            configured_modules: [],
            loaded_modules: [],
            load_errors: {},
            registered_count: 0,
            recognized_exports: ['register_domain_packs'],
            note: 'Module loader for this kind is planned for P38.',
        },
        {
            kind: 'eval_pack',
            contract_version: 'slm.evaluation-pack/v2',
            supports_safe_reload: false,
            has_module_loader: false,
            settings_key: null,
            configured_modules: [],
            loaded_modules: [],
            load_errors: {},
            registered_count: 0,
            recognized_exports: ['register_evaluation_packs'],
        },
    ],
};

function renderPage() {
    return render(
        <MemoryRouter initialEntries={['/project/7/extensions']}>
            <Routes>
                <Route
                    path="/project/:id/extensions"
                    element={<ProjectExtensionStudioPage />}
                />
            </Routes>
        </MemoryRouter>,
    );
}

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('ProjectExtensionStudioPage', () => {
    it('renders the catalog and reflects load errors on a kind row', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        renderPage();

        // Detail block surfaces the first kind by default.
        expect(
            await screen.findByText(/Plugin kinds/i),
        ).toBeInTheDocument();
        expect(
            screen.getAllByText('Data adapter').length,
        ).toBeGreaterThanOrEqual(1);
        // Training-runtime kind has a load error → badge surfaces in the row.
        expect(screen.getByText(/1 error\(s\)/i)).toBeInTheDocument();
    });

    it('switching kind in the sidebar updates the detail panel', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        renderPage();

        await screen.findByText(/Plugin kinds/i);
        const user = userEvent.setup();
        // Click the domain pack kind row.
        await user.click(
            screen.getByRole('button', {
                name: /Select Domain pack kind/i,
            }),
        );
        // Detail card shows the domain pack note.
        expect(
            await screen.findByText(/planned for P38/i),
        ).toBeInTheDocument();
        // The "Reload kind" button is disabled when reload is not supported.
        const reloadKindButton = screen.getByRole('button', {
            name: /Reload kind/i,
        });
        expect(reloadKindButton).toBeDisabled();
    });

    it('Generate scaffold POSTs the canonical body and renders the preview', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        const scaffoldResponse: ScaffoldResponse = {
            kind: 'data_adapter',
            plugin_id: 'phase87-ui-test',
            plugin_id_raw: 'phase87-ui-test',
            module_basename: 'phase87_ui_test',
            display_name: 'Phase87 Ui Test',
            description: 'Generated from the UI.',
            author: 'BrewSLM author',
            version: '0.1.0',
            contract_version: 'slm.data_adapter/v3',
            output_dir: '/tmp/phase87',
            written_files: ['/tmp/phase87/phase87_ui_test.py'],
            files: {
                'phase87_ui_test.py': 'def register_data_adapters(register):\n    pass\n',
                'test_phase87_ui_test.py': 'import unittest\n',
                'README.md': '# Phase87 Ui Test\n',
            },
        };
        apiMock.post.mockResolvedValueOnce({ data: scaffoldResponse });

        renderPage();
        await screen.findByText(/Plugin kinds/i);

        const user = userEvent.setup();
        await user.type(
            screen.getByLabelText(/Plugin id/i),
            'phase87-ui-test',
        );
        await user.type(
            screen.getByLabelText(/Description/i),
            'Generated from the UI.',
        );
        await user.click(
            screen.getByRole('button', { name: /Generate scaffold/i }),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/extensions/scaffold',
                expect.objectContaining({
                    kind: 'data_adapter',
                    plugin_id: 'phase87-ui-test',
                    description: 'Generated from the UI.',
                    write: true,
                }),
            );
        });

        expect(
            await screen.findByText(/scaffold ready/i),
        ).toBeInTheDocument();
        // Each filename surfaces in both the tab and the active-pane toolbar.
        expect(
            screen.getAllByText('phase87_ui_test.py').length,
        ).toBeGreaterThanOrEqual(1);
        expect(
            screen.getByRole('tab', { name: 'README.md' }),
        ).toBeInTheDocument();
    });

    it('Validate posts the active kind and renders pass/fail check rows', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        const report: PluginContractReport = {
            kind: 'data_adapter',
            module: 'example.bad.module',
            contract_version: 'slm.data_adapter/v3',
            declared_version: null,
            declared_ids: ['phase87-decl'],
            ok: false,
            import_error: null,
            checks: [
                {
                    name: 'module_importable',
                    ok: true,
                    message: "Imported 'example.bad.module'.",
                },
                {
                    name: 'module_interface',
                    ok: true,
                    message: 'Module exports: register_data_adapters.',
                },
                {
                    name: 'schema_compliance',
                    ok: false,
                    message: 'register_data_adapters(register) must take exactly one positional parameter (found 2).',
                },
                {
                    name: 'version_metadata',
                    ok: true,
                    message: 'matches.',
                },
                {
                    name: 'safe_reload',
                    ok: true,
                    message: 'ok.',
                },
            ],
        };
        apiMock.post.mockResolvedValueOnce({ data: report });

        renderPage();
        await screen.findByText(/Plugin kinds/i);

        const user = userEvent.setup();
        await user.type(
            screen.getByLabelText(/Importable Python module path/i),
            'example.bad.module',
        );
        await user.click(screen.getByRole('button', { name: /^Validate$/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/extensions/validate',
                {
                    kind: 'data_adapter',
                    module: 'example.bad.module',
                    force_reload: false,
                },
            );
        });

        const reportEl = await screen.findByLabelText(/Contract validation report/i);
        expect(reportEl).toBeInTheDocument();
        expect(within(reportEl).getByText('contract failed')).toBeInTheDocument();
        expect(
            within(reportEl).getByText(/one positional parameter/i),
        ).toBeInTheDocument();
        // Pass rows render too.
        expect(within(reportEl).getAllByText('pass').length).toBeGreaterThanOrEqual(4);
    });

    it('Reload all posts an empty body and refreshes the catalog', async () => {
        apiMock.get
            .mockResolvedValueOnce({ data: CATALOG })
            .mockResolvedValueOnce({ data: CATALOG });
        const reloadResponse: ReloadResponse = {
            results: [
                { kind: 'data_adapter', status: 'ok', registered_count: 4 },
                { kind: 'training_runtime', status: 'partial', failed_modules: { x: 'boom' } },
                { kind: 'domain_pack', status: 'not_supported' },
                { kind: 'eval_pack', status: 'not_supported' },
            ],
        };
        apiMock.post.mockResolvedValueOnce({ data: reloadResponse });

        renderPage();
        await screen.findByText(/Plugin kinds/i);

        const user = userEvent.setup();
        await user.click(screen.getByRole('button', { name: /Reload all/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/extensions/reload', {});
        });

        // Re-fetch the catalog after the reload completes.
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledTimes(2);
        });

        // Status badges appear inside each kind row.
        expect(screen.getAllByText('ok').length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText('partial')).toBeInTheDocument();
        expect(screen.getAllByText('not_supported').length).toBeGreaterThanOrEqual(2);
    });

    it('surfaces an error when the catalog request fails', async () => {
        apiMock.get.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'catalog_unavailable' } },
        });
        renderPage();
        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('catalog_unavailable');
    });
});
