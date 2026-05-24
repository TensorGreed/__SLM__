import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        post: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioAssistPanel from './DataStudioAssistPanel';

const assistPayload = {
    project_id: 1,
    focus: 'mapping',
    status: 'ok',
    provider: {
        provider: 'ollama',
        api_url: 'http://localhost:11434/v1/chat/completions',
        model_name: 'llama3',
        api_key_configured: false,
        tokens_used: 42,
    },
    source_of_truth: 'deterministic_data_studio_checks',
    auto_apply: false,
    summary: 'Category appears to be the label column.',
    suggestions: [
        {
            id: 'map-label-category',
            type: 'mapping',
            title: 'Map label to category',
            confidence: 0.91,
            rationale: 'The values are short repeated classes.',
            evidence: ['category has label-like values'],
            target_tab: 'dataprep',
            requires_user_confirmation: true,
            suggested_field_mapping: { label: 'category' },
        },
    ],
    deterministic_context: {
        verdict: 'attention',
        issues: [],
    },
    warnings: [
        'LLM suggestions are advisory and require user confirmation.',
    ],
};

describe('DataStudioAssistPanel', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('runs opt-in Ollama mapping assist and renders review-only suggestions', async () => {
        apiMock.post.mockResolvedValueOnce({ data: assistPayload });

        render(<DataStudioAssistPanel projectId={1} />);

        fireEvent.click(screen.getByRole('button', { name: /Run assist/i }));

        await waitFor(() => {
            expect(screen.getByText('Map label to category')).toBeInTheDocument();
        });

        expect(screen.getByText('Suggestions ready')).toBeInTheDocument();
        expect(screen.getByText('91%')).toBeInTheDocument();
        expect(screen.getByText(/Review required/i)).toBeInTheDocument();
        expect(screen.getAllByText(/category has label-like values/i).length).toBeGreaterThan(0);
        expect(apiMock.post).toHaveBeenCalledWith(
            '/projects/1/data-studio/assist',
            {
                focus: 'mapping',
                provider: 'ollama',
                model_name: 'llama3',
                api_url: undefined,
                api_key: undefined,
            },
        );
    });

    it('renders unavailable provider responses without suggestions', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                ...assistPayload,
                status: 'unavailable',
                summary: 'LLM assist is unavailable: connection refused',
                suggestions: [],
            },
        });

        render(<DataStudioAssistPanel projectId={1} />);

        fireEvent.click(screen.getByRole('button', { name: /Run assist/i }));

        await waitFor(() => {
            expect(screen.getByText('Unavailable')).toBeInTheDocument();
        });

        expect(screen.getAllByText(/connection refused/i).length).toBeGreaterThan(0);
        expect(screen.queryByText('Map label to category')).not.toBeInTheDocument();
    });
});
