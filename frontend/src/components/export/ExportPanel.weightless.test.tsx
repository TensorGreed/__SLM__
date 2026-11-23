import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ExportPanel from './ExportPanel';

function mockSummary(skip: string | null) {
  apiMock.get.mockImplementation(async (url: string) => {
    if (url.includes('/training/experiments')) return { data: [{ id: 4, name: 'exp-4', status: 'completed' }] };
    if (url.includes('/evaluation/summary')) {
      return { data: { verdict: 'not_evaluated', ...(skip ? { auto_eval_skip_reason: skip } : {}) } };
    }
    if (url.includes('/export/list')) return { data: [] };
    return { data: {} };
  });
}

describe('ExportPanel weightless run', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
  });

  it.each(['simulated_run', 'no_model_weights'])('disables Run Export with a visible reason (%s)', async (reason) => {
    mockSummary(reason);
    render(<ExportPanel projectId={9} stage="package" />);
    expect(await screen.findByTestId('export-weightless-reason')).toBeTruthy();
    expect((screen.getByRole('button', { name: 'Run Export' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('keeps Run Export enabled for a run that can be evaluated', async () => {
    mockSummary(null);
    render(<ExportPanel projectId={9} stage="package" />);
    await waitFor(() => expect((screen.getByTestId('export-run-select') as HTMLSelectElement).value).toBe('4'));
    await waitFor(() => expect(apiMock.get.mock.calls.some((c) => String(c[0]).includes('/evaluation/summary'))).toBe(true));
    expect(screen.queryByTestId('export-weightless-reason')).toBeNull();
    await waitFor(() => expect((screen.getByRole('button', { name: 'Run Export' }) as HTMLButtonElement).disabled).toBe(false));
  });
});
