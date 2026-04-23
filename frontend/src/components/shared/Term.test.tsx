import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import Term from './Term';
import { useGlossaryStore } from '../../stores/glossaryStore';

describe('Term component', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        useGlossaryStore.getState().reset();
    });

    afterEach(() => {
        useGlossaryStore.getState().reset();
    });

    it('renders the beginner label by default', () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        render(<Term id="domain_pack" />);
        expect(screen.getByRole('button')).toHaveTextContent('Domain Kit');
    });

    it('renders the advanced label when advanced prop is set', () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        render(<Term id="recipe" advanced />);
        expect(screen.getByRole('button')).toHaveTextContent('Recipe');
    });

    it('pluralizes when plural prop is set', () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        render(<Term id="gate" plural />);
        expect(screen.getByRole('button')).toHaveTextContent('Pass/Fail Checks');
    });

    it('opens the popover on click and shows plain-language copy', async () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        const user = userEvent.setup();
        render(<Term id="adapter" />);

        await user.click(screen.getByRole('button'));
        const tooltip = await screen.findByRole('tooltip');
        expect(tooltip).toHaveTextContent(/mapping layer/i);
        expect(tooltip).toHaveTextContent(/Also known as: Adapter/i);
    });

    it('fetches the glossary from the backend on mount', async () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        render(<Term id="domain_pack" />);
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/domain-blueprints/glossary/help');
        });
    });

    it('uses the backend plain_language when the glossary is loaded', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                entries: [
                    { term: 'recipe', plain_language: 'A fresh backend definition for recipe.', category: 'training' },
                ],
            },
        });

        const user = userEvent.setup();
        render(<Term id="recipe" />);
        await waitFor(() => {
            expect(useGlossaryStore.getState().loaded).toBe(true);
        });
        await user.click(screen.getByRole('button'));
        const tooltip = await screen.findByRole('tooltip');
        expect(tooltip).toHaveTextContent(/fresh backend definition/i);
    });

    it('falls back to id when the id is unknown', () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        render(<Term id="unknown_concept" />);
        expect(screen.getByText('unknown_concept')).toBeInTheDocument();
    });

    it('renders silent mode without a button/popover', () => {
        apiMock.get.mockResolvedValue({ data: { entries: [] } });
        render(<Term id="runtime" silent />);
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
        expect(screen.getByText('Training Backend')).toBeInTheDocument();
    });

    it('falls back to the definition.fallback when glossary fetch fails', async () => {
        apiMock.get.mockRejectedValue(new Error('network down'));
        const user = userEvent.setup();
        render(<Term id="runtime" />);
        await user.click(screen.getByRole('button'));
        const tooltip = await screen.findByRole('tooltip');
        expect(tooltip).toHaveTextContent(/backend that actually runs training/i);
    });
});
