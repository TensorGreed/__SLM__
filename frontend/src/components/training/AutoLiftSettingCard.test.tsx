import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({ apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() } }));
vi.mock('../../api/client', () => ({ default: apiMock }));

import AutoLiftSettingCard from './AutoLiftSettingCard';

const DEFAULT_ON = { enabled: null, global_enabled: true, effective: true, reason: null };
const OFF = { enabled: false, global_enabled: true, effective: false, reason: 'disabled_for_project' };
const GLOBAL_OFF = { enabled: true, global_enabled: false, effective: false, reason: 'disabled_globally' };

describe('AutoLiftSettingCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
        apiMock.get.mockResolvedValue({ data: DEFAULT_ON });
        apiMock.put.mockResolvedValue({ data: OFF });
    });

    it('is on by default and says so', async () => {
        render(<AutoLiftSettingCard projectId={4} />);
        expect(await screen.findByTestId('al-enabled')).toBeChecked();
        expect(screen.getByTestId('al-status')).toHaveTextContent('On:');
        expect(screen.queryByTestId('al-reset')).toBeNull();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/4/settings/auto-lift-eval');
    });

    it('turns the check off for the project and offers to go back to the default', async () => {
        const user = userEvent.setup();
        render(<AutoLiftSettingCard projectId={4} />);
        await user.click(await screen.findByTestId('al-enabled'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/4/settings/auto-lift-eval', { enabled: false }));
        expect(await screen.findByTestId('al-status')).toHaveTextContent('Off for this project');
        expect(screen.getByTestId('al-enabled')).not.toBeChecked();

        apiMock.put.mockResolvedValue({ data: DEFAULT_ON });
        await user.click(screen.getByTestId('al-reset'));
        await waitFor(() => expect(apiMock.put).toHaveBeenLastCalledWith('/projects/4/settings/auto-lift-eval', { enabled: null }));
        expect(await screen.findByTestId('al-enabled')).toBeChecked();
    });

    it('is disabled with the reason when the server-wide switch is off', async () => {
        apiMock.get.mockResolvedValue({ data: GLOBAL_OFF });
        render(<AutoLiftSettingCard projectId={4} />);
        const box = await screen.findByTestId('al-enabled');
        expect(box).toBeDisabled();
        expect(box).not.toBeChecked();
        expect(screen.getByTestId('al-status')).toHaveTextContent('AUTO_LIFT_EVAL_ENABLED');
    });

    it('shows a load error', async () => {
        apiMock.get.mockRejectedValue(new Error('boom'));
        render(<AutoLiftSettingCard projectId={4} />);
        expect(await screen.findByTestId('al-error')).toBeInTheDocument();
    });
});
