/**
 * Token classification: the "OCR-noise copies" control (Power Tools tab)
 * appears only for that task type and starts off.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import TrainingPanel from './TrainingPanel';

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/runtimes')) return { data: { project_id: 1, default_runtime_id: 'auto', runtimes: [] } };
        if (url.includes('/training/experiments')) return { data: [] };
        return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: {} });
    window.history.replaceState({}, '', '/');
});

describe('TrainingPanel — OCR-noise copies', () => {
    it('shows only for token classification, off by default, copies enabled once a rate is picked', async () => {
        const user = userEvent.setup();
        render(<TrainingPanel projectId={1} forceCreateVisible hideExperimentList />);
        await user.click(screen.getByRole('tab', { name: /Power Tools/i }));
        expect(screen.queryByTestId('ocr-noise-augment')).toBeNull();
        const taskSelect = (await screen.findAllByRole('combobox')).find(
            (el) => el.querySelector('option[value="token_classification"]'),
        ) as HTMLSelectElement;
        await user.selectOptions(taskSelect, 'token_classification');
        const rate = await screen.findByTestId('ocr-noise-rate');
        expect(rate).toHaveValue('0');
        expect(screen.getByTestId('ocr-noise-copies')).toBeDisabled();
        await user.selectOptions(rate, '0.1');
        expect(screen.getByTestId('ocr-noise-copies')).toBeEnabled();
        expect(screen.getByTestId('ocr-noise-augment')).toHaveTextContent('Offsets stay exact');
    });
});
