/**
 * Ship stages (Epic U phase U5): the export panel splits by ``stage`` —
 * Package = the export form + Releases, Publish = the Hub form over the
 * completed Hugging Face exports, Deploy = the plans; without a stage the
 * one-page layout still renders Export History and the Model Registry.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import ExportPanel from './ExportPanel';

const EXPORTS = [
    { id: 36, export_format: 'huggingface', status: 'completed', created_at: '2026-10-17T13:00:00Z', experiment_id: 135, manifest: {} },
    { id: 37, export_format: 'huggingface', status: 'completed', created_at: '2026-10-17T15:00:00Z', experiment_id: 143, manifest: {},
      hub_publish: { repo_id: 'TensorGreed/brewslm-sensitive-data-tagger', url: 'https://huggingface.co/TensorGreed/brewslm-sensitive-data-tagger', private: false, commit: 'abc1234' } },
    { id: 38, export_format: 'gguf', status: 'completed', created_at: '2026-10-17T16:00:00Z', experiment_id: 143, manifest: {} },
];

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/experiments')) return { data: [{ id: 143, name: 'run-143', status: 'completed' }] };
        if (url.includes('/export/list')) return { data: EXPORTS };
        if (url.includes('/registry/models')) return { data: { models: [{ id: 1, experiment_id: 143, name: 'tagger', version: '1', stage: 'staging', deployment_status: 'none', readiness: { metrics: {} } }] } };
        if (url.includes('/export/deployment-targets')) return { data: { default_target_ids: [], targets: [] } };
        return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: {} });
});

describe('ExportPanel — Ship stages', () => {
    it('Package: the export form, Releases and the exports + registered models; no publish card', async () => {
        render(<ExportPanel projectId={31} stage="package" />);
        expect(screen.getByTestId('export-form-card')).toHaveTextContent('New export');
        expect(await screen.findByTestId('export-releases')).toHaveTextContent('3 exports · 1 registered model');
        expect(screen.getByText('Exports')).toBeInTheDocument();
        expect(screen.getByText('Registered models')).toBeInTheDocument();
        expect(screen.queryByTestId('export-publish-card')).toBeNull();
    });

    it('Publish: one row per completed Hugging Face export — the published one shows its link, the other the form', async () => {
        render(<ExportPanel projectId={31} stage="publish" />);
        expect(await screen.findByTestId('export-publish-card')).toBeInTheDocument();
        expect(screen.getByTestId('export-publish-row-37')).toHaveTextContent('TensorGreed/brewslm-sensitive-data-tagger');
        expect(screen.getByTestId('export-publish-row-36').querySelector('input[type="text"]')).toBeTruthy();
        expect(screen.queryByTestId('export-publish-row-38')).toBeNull();
        expect(screen.queryByTestId('export-form-card')).toBeNull();
        expect(screen.queryByTestId('export-releases')).toBeNull();
    });

    it('Publish: filling the repo and pressing Publish posts to publish-hub', async () => {
        const user = userEvent.setup();
        apiMock.post.mockImplementation(async (url: string) => {
            if (url.endsWith('/export/36/publish-hub')) return { data: { repo_id: 'me/tagger', url: 'https://huggingface.co/me/tagger', private: false, commit: 'def5678' } };
            return { data: {} };
        });
        render(<ExportPanel projectId={31} stage="publish" />);
        const row = await screen.findByTestId('export-publish-row-36');
        await user.type(row.querySelector('input[type="text"]') as HTMLInputElement, 'me/tagger');
        await user.click(screen.getAllByRole('button', { name: 'Publish to Hugging Face' })[0]);
        await waitFor(() => expect(apiMock.post).toHaveBeenCalledWith('/projects/31/export/36/publish-hub', { repo_id: 'me/tagger', private: false }));
        expect(await screen.findByText('me/tagger')).toBeInTheDocument();
    });

    it('Deploy and Monitor render no export blocks until a plan exists', async () => {
        render(<ExportPanel projectId={31} stage="deploy" />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalledWith(expect.stringContaining('/export/list')));
        expect(screen.queryByTestId('export-form-card')).toBeNull();
        expect(screen.queryByTestId('export-publish-card')).toBeNull();
        expect(screen.queryByText('Exports')).toBeNull();
    });
});
