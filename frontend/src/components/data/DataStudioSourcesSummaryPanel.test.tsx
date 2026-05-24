import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioSourcesSummaryPanel from './DataStudioSourcesSummaryPanel';

const sourcesPayload = {
    project_id: 1,
    verdict: 'attention',
    totals: {
        dataset_count: 2,
        document_count: 2,
        row_count: 82,
        accepted_documents: 1,
        pending_documents: 0,
        processing_documents: 0,
        error_documents: 1,
        rejected_documents: 0,
    },
    dataset_groups: [
        {
            dataset_type: 'raw',
            dataset_count: 1,
            row_count: 42,
            locked_count: 0,
            with_file_count: 1,
        },
        {
            dataset_type: 'cleaned',
            dataset_count: 1,
            row_count: 40,
            locked_count: 0,
            with_file_count: 1,
        },
    ],
    recent_documents: [
        {
            id: 1,
            dataset_id: 1,
            dataset_name: 'Uploaded CSV',
            dataset_type: 'raw',
            filename: 'tickets.csv',
            file_type: 'csv',
            status: 'accepted',
            source: 'upload',
            sensitivity: 'internal',
            file_size_bytes: 2048,
            chunk_count: 3,
            quality_score: null,
            ingested_at: '2026-05-24T12:00:00Z',
        },
        {
            id: 2,
            dataset_id: 1,
            dataset_name: 'Uploaded CSV',
            dataset_type: 'raw',
            filename: 'bad.jsonl',
            file_type: 'jsonl',
            status: 'error',
            source: 'upload',
            sensitivity: 'internal',
            file_size_bytes: 512,
            chunk_count: 0,
            quality_score: null,
            ingested_at: '2026-05-24T12:01:00Z',
        },
    ],
    issues: [
        {
            id: 'source_errors',
            severity: 'warning',
            title: 'Source import errors',
            message: '1 source document failed ingestion.',
            action_label: 'Inspect failed sources',
            target_tab: 'data',
        },
    ],
};

describe('DataStudioSourcesSummaryPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders source health, dataset groups, and recent documents', async () => {
        apiMock.get.mockResolvedValueOnce({ data: sourcesPayload });

        render(<DataStudioSourcesSummaryPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-sources')).toBeInTheDocument();
        });

        expect(screen.getByText('Needs attention')).toBeInTheDocument();
        expect(screen.getByText('Source docs')).toBeInTheDocument();
        expect(screen.getByText('82')).toBeInTheDocument();
        expect(screen.getByText('Raw')).toBeInTheDocument();
        expect(screen.getByText('Cleaned')).toBeInTheDocument();
        expect(screen.getByText('tickets.csv')).toBeInTheDocument();
        expect(screen.getByText('bad.jsonl')).toBeInTheDocument();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/sources');
    });

    it('renders the empty-state source guidance', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...sourcesPayload,
                verdict: 'empty',
                totals: {
                    ...sourcesPayload.totals,
                    dataset_count: 0,
                    document_count: 0,
                    row_count: 0,
                    accepted_documents: 0,
                    error_documents: 0,
                },
                dataset_groups: [],
                recent_documents: [],
                issues: [],
            },
        });

        render(<DataStudioSourcesSummaryPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText('No sources')).toBeInTheDocument();
        });
        expect(screen.getByText(/Use the import controls below/i)).toBeInTheDocument();
    });
});
