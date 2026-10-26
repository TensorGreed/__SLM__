import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import GenerationModelPicker from './GenerationModelPicker';

function mockApi(backends: unknown[]) {
    apiMock.get.mockImplementation((url: string) => {
        if (url.endsWith('/cloud/models')) return Promise.resolve({ data: { providers: [] } });
        return Promise.resolve({ data: { project_id: 1, backends } });
    });
}

const CLI = { name: 'command', available: true, describe: 'command', billing: 'subscription', detail: 'claude CLI (subscription) · batch 8 · concurrency 1' };

describe('GenerationModelPicker', () => {
    beforeEach(() => { apiMock.get.mockReset(); });

    it('offers the subscription CLI backend and emits its pin', async () => {
        mockApi([CLI, { name: 'ollama', available: true, describe: 'ollama', billing: 'local' }]);
        const onChange = vi.fn();
        render(<GenerationModelPicker projectId={1} value={null} onChange={onChange} testIdPrefix="t" />);
        const select = screen.getByTestId('t-provider');
        await waitFor(() => expect(select.textContent).toContain('Local CLI (subscription) — claude CLI (subscription) · batch 8'));
        await userEvent.selectOptions(select, '__subscription');
        expect(onChange).toHaveBeenCalledWith('command');
    });

    it('labels the chosen CLI backend as subscription-billed', async () => {
        mockApi([CLI]);
        render(<GenerationModelPicker projectId={1} value="command" onChange={vi.fn()} testIdPrefix="t" />);
        expect((await screen.findByTestId('t-subscription-note')).textContent).toContain('not billed per call');
    });

    it('omits the option when no subscription backend is available', async () => {
        mockApi([{ ...CLI, available: false }]);
        render(<GenerationModelPicker projectId={1} value={null} onChange={vi.fn()} testIdPrefix="t" />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledTimes(2));
        expect(screen.getByTestId('t-provider').textContent).not.toContain('subscription');
    });
});
