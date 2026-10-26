import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import JudgeModelCard from './JudgeModelCard';

const STATE = {
    enabled: true,
    backend: null,
    env_override: null,
    resolved: { describe: 'ollama:qwen3:8b', local: true, billing: 'local' },
    options: [
        { value: '', label: 'Auto', billing: 'local', available: true },
        { value: 'command', label: 'command', billing: 'subscription', available: true },
        { value: 'cloud:anthropic', label: 'anthropic', billing: 'metered', available: false },
    ],
};

describe('JudgeModelCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('shows the resolved judge with its billing and the options', async () => {
        apiMock.get.mockResolvedValue({ data: STATE });
        render(<JudgeModelCard projectId={3} />);
        expect((await screen.findByTestId('judge-resolved')).textContent).toContain('ollama:qwen3:8b');
        expect(screen.getByTestId('judge-billing').textContent).toContain('local');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/3/eval-judge');
        expect(screen.getByTestId('judge-select').textContent).toContain('anthropic (not available)');
    });

    it('saves a new pick and shows the billing of the new resolved judge', async () => {
        apiMock.get.mockResolvedValue({ data: STATE });
        apiMock.put.mockResolvedValue({
            data: { ...STATE, backend: 'command', resolved: { describe: 'command', local: false, billing: 'subscription' } },
        });
        render(<JudgeModelCard projectId={3} />);
        await userEvent.selectOptions(await screen.findByTestId('judge-select'), 'command');
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/3/eval-judge', { enabled: true, backend: 'command' }));
        await waitFor(() => expect(screen.getByTestId('judge-billing').textContent).toContain('subscription'));
    });

    it('toggles enabled and shows the env override notice and save errors', async () => {
        apiMock.get.mockResolvedValue({ data: { ...STATE, env_override: 'ollama:x' } });
        apiMock.put.mockRejectedValue({ response: { status: 422, data: { detail: 'bad backend', error_code: 'invalid', message: 'bad backend' } } });
        render(<JudgeModelCard projectId={3} />);
        expect((await screen.findByTestId('judge-env-override')).textContent).toContain('ollama:x');
        await userEvent.click(screen.getByTestId('judge-enabled'));
        expect((await screen.findByTestId('judge-error')).textContent).toContain('bad backend');
        expect(apiMock.put).toHaveBeenCalledWith('/projects/3/eval-judge', { enabled: false, backend: null });
    });

    it('renders nothing when the judge settings cannot load', async () => {
        apiMock.get.mockRejectedValue(new Error('down'));
        const { container } = render(<JudgeModelCard projectId={3} />);
        await waitFor(() => expect(container.querySelector('[data-testid="judge-model-card"]')).toBeNull());
    });
});
