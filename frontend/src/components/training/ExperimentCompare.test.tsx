/** Results > Compare shows the epochs a run trained, not the form value auto epochs replaced (phase 7i fix pass, B). */

import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('recharts', () => {
    const Stub = ({ children }: { children?: unknown }) => <div>{children as never}</div>;
    return { LineChart: Stub, Line: Stub, XAxis: Stub, YAxis: Stub, CartesianGrid: Stub, Tooltip: Stub, Legend: Stub, ResponsiveContainer: Stub };
});

import ExperimentCompare from './ExperimentCompare';

describe('ExperimentCompare epochs row', () => {
    it('uses the trainer plan (total_epochs), the config for older runs, else a dash', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                experiments: [
                    { id: 1, name: 'auto-run', config: { num_epochs: 3, batch_size: 4 }, total_epochs: 8, history: [] },
                    { id: 2, name: 'old-run', config: { num_epochs: 2, batch_size: 8 }, total_epochs: null, history: [] },
                    { id: 3, name: 'bare-run', config: {}, total_epochs: null, history: [] },
                ],
            },
        });
        render(<ExperimentCompare projectId={1} experimentIds={[1, 2, 3]} onClose={() => {}} />);
        const row = (await screen.findByText('Epochs / Batch')).closest('tr')!;
        const cells = within(row).getAllByRole('cell').map((cell) => cell.textContent);
        expect(cells).toEqual(['8 / 4', '2 / 8', '— / —']);
    });
});
