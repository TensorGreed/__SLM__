import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ToolCatalogCard from './ToolCatalogCard';

const schema = (props: Record<string, unknown>, required: string[] = []) => ({ type: 'object', properties: props, required });
const STATE = {
    catalog: {
        version: 'a1b2c3d4e5f6',
        flags_version: '0f1e2d3c4b5a',
        imported_at: '2026-10-26T12:00:00Z',
        source: { kind: 'mcp_stdio', label: 'deploy lab', command: ['/opt/bin/deploy-mcp', '--stdio'], env_names: ['API_TOKEN'], secret_provider: 'mcp_deploy_lab' },
        server: { name: 'deploy-service', version: '1.2.0', protocol_version: '2025-06-18' },
        tools: [
            { name: 'get_service', description: 'Fetch one service.', input_schema: schema({ name: { type: 'string' } }, ['name']), read_only: true, destructive: null, allowed: true, read_only_effective: true, mutating: false },
            { name: 'create_service', description: 'Create a service.', input_schema: schema({ name: { type: 'string' }, tier: { enum: ['free', 'pro'] } }, ['name']), read_only: false, destructive: false, allowed: true, read_only_effective: false, mutating: true },
            { name: 'delete_service', description: 'Delete a service.', input_schema: schema({ name: { type: 'string' } }, ['name']), read_only: false, destructive: true, allowed: false, read_only_effective: false, mutating: true },
        ],
    },
    history: [
        { version: 'a1b2c3d4e5f6', imported_at: '2026-10-26', source_kind: 'mcp_stdio', tool_count: 3, current: true, restorable: true },
        { version: '111111111111', imported_at: '2026-10-25', source_kind: 'openapi', tool_count: 2, current: false, restorable: true },
    ],
};

async function openCard() {
    const user = userEvent.setup();
    render(<ToolCatalogCard projectId={7} />);
    await user.click(await screen.findByTestId('tc-toggle'));
    return user;
}

describe('ToolCatalogCard', () => {
    beforeEach(() => {
        for (const fn of Object.values(apiMock)) fn.mockReset();
        apiMock.get.mockResolvedValue({ data: STATE });
        apiMock.patch.mockResolvedValue({ data: STATE });
        apiMock.post.mockResolvedValue({ data: STATE });
    });

    it('collapsed header shows version and counts, then provenance and the table', async () => {
        render(<ToolCatalogCard projectId={7} />);
        expect(await screen.findByTestId('tc-summary')).toHaveTextContent('3 tools · 2 allowed · version a1b2c3d4e5f6');
        await userEvent.setup().click(screen.getByTestId('tc-toggle'));
        const prov = screen.getByTestId('tc-provenance');
        expect(prov).toHaveTextContent('MCP server (local process) "deploy lab"');
        expect(prov).toHaveTextContent('server deploy-service 1.2.0');
        expect(prov).toHaveTextContent('imported 2026-10-26T12:00:00Z');
        expect(screen.getByTestId('tc-risk-get_service')).toHaveTextContent('read-only');
        expect(screen.getByTestId('tc-risk-create_service')).toHaveTextContent('mutating');
        expect(screen.getByTestId('tc-risk-delete_service')).toHaveTextContent('destructive');
        expect(screen.getByTestId('tc-tool-create_service')).toHaveTextContent('create_service(name: string, tier?: free|pro)');
    });

    it('disables the read-only override for a destructive tool', async () => {
        await openCard();
        expect(screen.getByTestId('tc-override-delete_service')).toBeDisabled();
        expect(screen.getByTestId('tc-override-create_service')).not.toBeDisabled();
    });

    it('toggling allowed and the override patch the flags', async () => {
        const user = await openCard();
        await user.click(screen.getByTestId('tc-allowed-delete_service'));
        await waitFor(() => expect(apiMock.patch).toHaveBeenCalledWith('/projects/7/tool-catalog/flags', { flags: { delete_service: { allowed: true } } }));
        await user.selectOptions(screen.getByTestId('tc-override-create_service'), 'yes');
        await waitFor(() => expect(apiMock.patch).toHaveBeenLastCalledWith('/projects/7/tool-catalog/flags', { flags: { create_service: { read_only: true } } }));
    });

    it('bulk deny sends every tool', async () => {
        const user = await openCard();
        await user.click(screen.getByTestId('tc-deny-all'));
        await waitFor(() => expect(apiMock.patch).toHaveBeenCalled());
        expect(apiMock.patch.mock.calls[0][1].flags).toEqual({ get_service: { allowed: false }, create_service: { allowed: false }, delete_service: { allowed: false } });
    });

    it('prefills the stored env names with blank password inputs; blank imports send null', async () => {
        const user = await openCard();
        expect(screen.getByTestId('tc-env-name-0')).toHaveValue('API_TOKEN');
        const value = screen.getByTestId('tc-env-value-0');
        expect(value).toHaveAttribute('type', 'password');
        expect(value).toHaveValue('');
        await user.click(screen.getByTestId('tc-import'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        const [url, body] = apiMock.post.mock.calls[0];
        expect(url).toBe('/projects/7/tool-catalog/import/mcp');
        expect(body).toEqual({ kind: 'mcp_stdio', command: ['/opt/bin/deploy-mcp', '--stdio'], env: { API_TOKEN: null }, label: 'deploy lab' });
    });

    it('sends a typed secret value and never renders it', async () => {
        const user = await openCard();
        await user.type(screen.getByTestId('tc-env-value-0'), 'hunter2');
        await user.click(screen.getByTestId('tc-import'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1].env).toEqual({ API_TOKEN: 'hunter2' });
        expect(document.body.textContent).not.toContain('hunter2');
    });

    it('imports from an HTTP server with headers', async () => {
        const user = await openCard();
        await user.click(screen.getByTestId('tc-tab-mcp_http'));
        await user.clear(screen.getByTestId('tc-url'));
        await user.type(screen.getByTestId('tc-url'), 'http://localhost:9000/mcp');
        await user.click(screen.getByTestId('tc-header-add'));
        await user.type(screen.getByTestId('tc-header-name-0'), 'Authorization');
        await user.type(screen.getByTestId('tc-header-value-0'), 'Bearer abc');
        await user.click(screen.getByTestId('tc-import'));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][1]).toMatchObject({ kind: 'mcp_http', url: 'http://localhost:9000/mcp', headers: { Authorization: 'Bearer abc' } });
        expect(screen.getByTestId('tc-header-value-0')).toHaveAttribute('type', 'password');
    });

    it('uploads an OpenAPI file', async () => {
        const user = await openCard();
        await user.click(screen.getByTestId('tc-tab-openapi'));
        const file = new File(['{"openapi":"3.0.0"}'], 'api.json', { type: 'application/json' });
        await user.upload(screen.getByTestId('tc-openapi-upload'), file);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalled());
        expect(apiMock.post.mock.calls[0][0]).toBe('/projects/7/tool-catalog/import/openapi');
        expect(apiMock.post.mock.calls[0][1]).toEqual({ filename: 'api.json', content: '{"openapi":"3.0.0"}' });
    });

    it('shows the stdio-disabled error with its message', async () => {
        apiMock.post.mockRejectedValue({ response: { data: { error_code: 'MCP_STDIO_DISABLED', detail: 'Set MCP_STDIO_ALLOWED_BINARIES to allow deploy-mcp.' } } });
        const user = await openCard();
        await user.click(screen.getByTestId('tc-import'));
        const err = await screen.findByTestId('tc-error');
        expect(err).toHaveTextContent('MCP_STDIO_DISABLED');
        expect(err).toHaveTextContent('MCP_STDIO_ALLOWED_BINARIES');
    });

    it('lists invalid-catalog errors', async () => {
        apiMock.post.mockRejectedValue({ response: { data: { error_code: 'INVALID_TOOL_CATALOG', detail: 'invalid', metadata: { errors: ['tools[0]: name is required'] } } } });
        const user = await openCard();
        await user.click(screen.getByTestId('tc-import'));
        expect(await screen.findByTestId('tc-error')).toHaveTextContent('tools[0]: name is required');
    });

    it('restores a history entry and clears the catalog', async () => {
        const user = await openCard();
        const row = screen.getByTestId('tc-history-111111111111');
        await user.click(within(row).getByRole('button', { name: 'Restore' }));
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/7/tool-catalog/restore', { version: '111111111111' }));
        apiMock.delete.mockResolvedValue({ data: {} });
        await user.click(screen.getByTestId('tc-clear'));
        await waitFor(() => expect(apiMock.delete).toHaveBeenCalledWith('/projects/7/tool-catalog'));
        expect(await screen.findByText('no catalog')).toBeInTheDocument();
    });
});
