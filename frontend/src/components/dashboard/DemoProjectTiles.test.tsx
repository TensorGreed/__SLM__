import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { DemoCatalogResponse, DemoSeedResponse } from '../../types/demoProjects';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DemoProjectTiles from './DemoProjectTiles';

const CATALOG: DemoCatalogResponse = {
    archetypes: [
        {
            slug: 'support-faq',
            name: 'Demo · Support FAQ',
            headline: 'Build a support assistant from real ticket Q&A',
            description: '20 cleaned tickets + 6 gold rows.',
            task_profile: 'instruction_sft',
            target_profile: 'vllm_server',
            suggested_brief: 'Build a support FAQ assistant.',
        },
        {
            slug: 'sentiment-classifier',
            name: 'Demo · Sentiment classifier',
            headline: 'Tiny product-review classifier on mobile-cpu',
            description: '30 reviews labelled positive/neutral/negative.',
            task_profile: 'classification',
            target_profile: 'mobile_cpu',
            suggested_brief: 'Train a 3-way sentiment classifier.',
        },
    ],
};

function renderWithRouter() {
    return render(
        <MemoryRouter initialEntries={['/']}>
            <Routes>
                <Route path="/" element={<DemoProjectTiles />} />
                <Route
                    path="/project/:id"
                    element={<div data-testid="project-workspace" />}
                />
            </Routes>
        </MemoryRouter>,
    );
}

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
});

describe('DemoProjectTiles', () => {
    it('renders both archetypes from the catalog', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        renderWithRouter();
        expect(
            await screen.findByText('Demo · Support FAQ'),
        ).toBeInTheDocument();
        expect(screen.getByText('Demo · Sentiment classifier')).toBeInTheDocument();
        expect(
            screen.getByText(/Build a support assistant from real ticket/i),
        ).toBeInTheDocument();
    });

    it('clicking a tile POSTs to seeder and navigates to the new project', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        const seedResponse: DemoSeedResponse = {
            summary: {
                slug: 'support-faq',
                created: true,
                project_id: 42,
                project_name: 'Demo · Support FAQ',
                gold_row_count: 6,
                source_row_count: 20,
            },
            project: {
                id: 42,
                name: 'Demo · Support FAQ',
                description: '20 cleaned tickets.',
                status: 'active',
                beginner_mode: true,
                target_profile_id: 'vllm_server',
                training_preferred_plan_profile: 'balanced',
                evaluation_preferred_pack_id: 'evalpack.general.default',
            },
        };
        apiMock.post.mockResolvedValueOnce({ data: seedResponse });

        renderWithRouter();
        await screen.findByText('Demo · Support FAQ');

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', {
                name: /Open the Demo · Support FAQ demo project/i,
            }),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/demo-projects/support-faq',
                {},
            );
        });
        // Router navigated.
        expect(await screen.findByTestId('project-workspace')).toBeInTheDocument();
    });

    it('renders nothing when the catalog is empty', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { archetypes: [] } });
        const { container } = renderWithRouter();
        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith('/demo-projects');
        });
        // The component returns null when no archetypes — no heading.
        expect(container.querySelector('.demo-project-tiles')).toBeNull();
    });

    it('surfaces an error when the seed POST fails', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        apiMock.post.mockRejectedValueOnce({
            response: { status: 500, data: { detail: 'demo_seed_failed' } },
        });

        renderWithRouter();
        await screen.findByText('Demo · Support FAQ');

        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', {
                name: /Open the Demo · Support FAQ demo project/i,
            }),
        );

        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent(/demo_seed_failed/i);
    });

    it('disables the other tile while one is being seeded', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        // Never resolve so we can observe the in-flight state.
        const pending = new Promise(() => {});
        apiMock.post.mockReturnValueOnce(pending);

        renderWithRouter();
        await screen.findByText('Demo · Support FAQ');

        const user = userEvent.setup();
        const supportButton = screen.getByRole('button', {
            name: /Open the Demo · Support FAQ demo project/i,
        });
        await user.click(supportButton);

        const sentimentButton = screen.getByRole('button', {
            name: /Open the Demo · Sentiment classifier demo project/i,
        });
        expect(sentimentButton).toBeDisabled();
        expect(supportButton).toHaveTextContent(/Seeding…/);
    });

    it('Reset confirms, POSTs to the reset endpoint, and navigates', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        const resetResponse: DemoSeedResponse = {
            summary: {
                slug: 'support-faq', created: true, reset: true, project_id: 7,
                project_name: 'Demo · Support FAQ', gold_row_count: 6, source_row_count: 20,
            },
            project: {
                id: 7, name: 'Demo · Support FAQ', description: '', status: 'active',
                beginner_mode: true, target_profile_id: 'vllm_server',
                training_preferred_plan_profile: 'balanced',
                evaluation_preferred_pack_id: 'evalpack.general.default',
            },
        };
        apiMock.post.mockResolvedValueOnce({ data: resetResponse });
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

        renderWithRouter();
        await screen.findByText('Demo · Support FAQ');
        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Reset the Demo · Support FAQ demo project/i }),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/demo-projects/support-faq/reset', {});
        });
        expect(await screen.findByTestId('project-workspace')).toBeInTheDocument();
        confirmSpy.mockRestore();
    });

    it('Reset does nothing when the confirm is cancelled', async () => {
        apiMock.get.mockResolvedValueOnce({ data: CATALOG });
        const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);

        renderWithRouter();
        await screen.findByText('Demo · Support FAQ');
        const user = userEvent.setup();
        await user.click(
            screen.getByRole('button', { name: /Reset the Demo · Support FAQ demo project/i }),
        );

        expect(apiMock.post).not.toHaveBeenCalled();
        confirmSpy.mockRestore();
    });
});
