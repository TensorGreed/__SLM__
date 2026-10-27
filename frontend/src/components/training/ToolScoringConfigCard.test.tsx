import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ToolScoringConfigCard from './ToolScoringConfigCard';

const STATE = {
    config: {},
    effective: { enabled: true, modes: ['tool_call_valid', 'tool_args_match', 'plan_order_valid'], steps_pointer: '/steps', tool_field: 'tool', args_field: 'args', id_field: 'step', depends_field: 'depends_on', arg_keys: [], dry_run: false },
    catalog: { version: 'a1b2c3d4e5f6', tool_count: 4, source_kind: 'mcp_stdio' },
};

describe('ToolScoringConfigCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset(); apiMock.put.mockReset();
        apiMock.get.mockResolvedValue({ data: STATE });
        apiMock.put.mockResolvedValue({ data: STATE });
    });

    it('shows the catalog provenance and the dry-run safety note', async () => {
        render(<ToolScoringConfigCard projectId={3} isPlan />);
        expect(await screen.findByTestId('ts-catalog-line')).toHaveTextContent('catalog version a1b2c3d4e5f6 (4 tools, source mcp_stdio)');
        expect(screen.getByTestId('ts-dry-run-note')).toHaveTextContent('never calls mutating tools');
        expect(screen.getByTestId('ts-steps-pointer')).toHaveValue('/steps');
    });

    it('hides the plan fields and the order check for a single tool call', async () => {
        render(<ToolScoringConfigCard projectId={3} isPlan={false} />);
        await screen.findByTestId('ts-config');
        expect(screen.queryByTestId('ts-steps-pointer')).toBeNull();
        expect(screen.queryByTestId('ts-mode-plan_order_valid')).toBeNull();
    });

    it('saves the chosen checks, pointer, keys and dry run', async () => {
        const user = userEvent.setup();
        render(<ToolScoringConfigCard projectId={3} isPlan />);
        await screen.findByTestId('ts-config');
        await user.click(screen.getByTestId('ts-mode-plan_order_valid'));
        await user.clear(screen.getByTestId('ts-steps-pointer'));
        await user.type(screen.getByTestId('ts-steps-pointer'), '/plan/steps');
        await user.type(screen.getByTestId('ts-arg-keys'), 'name, tier');
        await user.click(screen.getByTestId('ts-dry-run'));
        await user.click(screen.getByTestId('ts-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        const [url, body] = apiMock.put.mock.calls[0];
        expect(url).toBe('/projects/3/tool-scoring-config');
        expect(body).toMatchObject({ modes: ['tool_call_valid', 'tool_args_match'], steps_pointer: '/plan/steps', arg_keys: ['name', 'tier'], dry_run: true });
    });

    it('shows the 422 messages', async () => {
        apiMock.put.mockRejectedValue({ response: { data: { detail: 'bad', metadata: { errors: ['dry_run needs MCP_DRY_RUN_ENABLED on the server'] } } } });
        const user = userEvent.setup();
        render(<ToolScoringConfigCard projectId={3} isPlan />);
        await screen.findByTestId('ts-config');
        await user.click(screen.getByTestId('ts-save'));
        expect(await screen.findByTestId('ts-save-errors')).toHaveTextContent('MCP_DRY_RUN_ENABLED');
    });
});
