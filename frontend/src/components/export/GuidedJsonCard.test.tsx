import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import GuidedJsonCard from './GuidedJsonCard';

describe('GuidedJsonCard', () => {
    beforeEach(() => { apiMock.get.mockReset(); });

    it('shows source, taxonomy version, label count and the schema', async () => {
        apiMock.get.mockResolvedValue({ data: { guided_json: {
            source: 'recipe:code-findings', taxonomy_version: 'v3', updated_at: '2026-10-26T00:00:00Z',
            schema: { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: { label: { enum: ['a', 'b', 'c'] } } } } } },
        } } });
        render(<GuidedJsonCard projectId={5} />);
        expect(await screen.findByTestId('guided-json-card')).toBeInTheDocument();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/5/guided-json');
        expect(screen.getByTestId('guided-json-source')).toHaveTextContent('recipe:code-findings');
        expect(screen.getByTestId('guided-json-taxonomy')).toHaveTextContent('v3');
        expect(screen.getByTestId('guided-json-labels')).toHaveTextContent('3');
        expect(screen.getByTestId('guided-json-schema')).toHaveTextContent('"findings"');
    });

    it('renders nothing when the hint is null', async () => {
        apiMock.get.mockResolvedValue({ data: { guided_json: null } });
        const { container } = render(<GuidedJsonCard projectId={5} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(container).toBeEmptyDOMElement();
    });

    it('renders nothing when the call fails', async () => {
        apiMock.get.mockImplementation(() => Promise.reject(new Error('x')));
        const { container } = render(<GuidedJsonCard projectId={5} />);
        await waitFor(() => expect(apiMock.get).toHaveBeenCalled());
        expect(container).toBeEmptyDOMElement();
    });
});
