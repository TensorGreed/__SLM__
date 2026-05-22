import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
    navigateMock: vi.fn(),
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useNavigate: () => navigateMock,
    };
});

import ProjectTemplateGallery from './ProjectTemplateGallery';

const SAMPLE_TEMPLATES = {
    templates: [
        {
            slug: 'ticket-router',
            name: 'Ticket Router SLM',
            headline: 'Auto-route inbound support tickets.',
            description: 'Multi-class classifier for support inboxes.',
            icon: '📨',
            recipe_id: 'classification',
            task_profile: 'classification',
            target_profile: 'mobile_cpu',
            training_preferred_plan_profile: 'fast-iteration',
            evaluation_preferred_pack_id: 'evalpack.classification.default',
            minimum_dataset_size: 150,
            recommended_base_models: ['HuggingFaceTB/SmolLM2-135M-Instruct'],
            labels: ['billing', 'technical', 'account', 'sales', 'legal'],
            suggested_brief: '',
            template_version: 'v1',
            dataset_input_field: 'ticket',
            dataset_output_field: 'label',
        },
        {
            slug: 'security-alert-summarizer',
            name: 'Security Alert Summarizer',
            headline: 'Turn vendor advisories into exec summaries.',
            description: 'Summarization for security advisories.',
            icon: '🚨',
            recipe_id: 'summarization',
            task_profile: 'summarization',
            target_profile: 'vllm_server',
            training_preferred_plan_profile: 'balanced',
            evaluation_preferred_pack_id: 'evalpack.general.default',
            minimum_dataset_size: 80,
            recommended_base_models: ['HuggingFaceTB/SmolLM2-135M-Instruct'],
            labels: [],
            suggested_brief: '',
            template_version: 'v1',
            dataset_input_field: 'advisory',
            dataset_output_field: 'summary',
        },
    ],
    count: 2,
};

function renderWithRouter(node: React.ReactNode) {
    return render(<MemoryRouter>{node}</MemoryRouter>);
}

describe('ProjectTemplateGallery', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        navigateMock.mockReset();
    });

    it('renders a loading state while the catalog fetch is pending', () => {
        apiMock.get.mockReturnValue(new Promise(() => undefined));
        renderWithRouter(<ProjectTemplateGallery />);
        expect(
            screen.getByTestId('project-template-gallery-loading'),
        ).toBeInTheDocument();
    });

    it('renders one card per template with headline + badges', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_TEMPLATES });
        renderWithRouter(<ProjectTemplateGallery />);

        await waitFor(() => {
            expect(screen.getByTestId('project-template-gallery')).toBeInTheDocument();
        });
        expect(screen.getByTestId('project-template-card-ticket-router')).toBeInTheDocument();
        expect(
            screen.getByTestId('project-template-card-security-alert-summarizer'),
        ).toBeInTheDocument();
        expect(
            screen.getByTestId('project-template-card-ticket-router-headline'),
        ).toHaveTextContent('Auto-route inbound support tickets.');
    });

    it('opens the inline name prompt when "Use this template" is clicked', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_TEMPLATES });
        renderWithRouter(<ProjectTemplateGallery />);

        const user = userEvent.setup();
        await user.click(
            await screen.findByTestId('project-template-pick-ticket-router'),
        );

        expect(
            screen.getByTestId('project-template-name-form-ticket-router'),
        ).toBeInTheDocument();
        const input = screen.getByTestId(
            'project-template-name-input-ticket-router',
        ) as HTMLInputElement;
        // Pre-fills with the template's display name.
        expect(input.value).toBe('Ticket Router SLM');
    });

    it('submits with the chosen name, navigates into the new project, fires a toast', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_TEMPLATES });
        apiMock.post.mockResolvedValueOnce({
            data: { id: 42, name: 'Acme Ticket Router' },
        });

        renderWithRouter(<ProjectTemplateGallery />);
        const user = userEvent.setup();
        await user.click(
            await screen.findByTestId('project-template-pick-ticket-router'),
        );

        const input = screen.getByTestId(
            'project-template-name-input-ticket-router',
        ) as HTMLInputElement;
        await user.clear(input);
        await user.type(input, 'Acme Ticket Router');
        await user.click(
            screen.getByTestId('project-template-submit-ticket-router'),
        );

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/project-templates/ticket-router/instantiate',
                { project_name: 'Acme Ticket Router' },
            );
        });
        expect(navigateMock).toHaveBeenCalledWith('/project/42/guide');
    });

    it('shows an inline error when the instantiate call fails', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_TEMPLATES });
        apiMock.post.mockRejectedValueOnce({
            response: { data: { detail: 'template_manifest_invalid:foo' } },
        });

        renderWithRouter(<ProjectTemplateGallery />);
        const user = userEvent.setup();
        await user.click(
            await screen.findByTestId('project-template-pick-ticket-router'),
        );
        await user.click(
            screen.getByTestId('project-template-submit-ticket-router'),
        );

        const errorBlock = await screen.findByTestId(
            'project-template-error-ticket-router',
        );
        expect(errorBlock).toHaveTextContent('template_manifest_invalid:foo');
        expect(navigateMock).not.toHaveBeenCalled();
    });

    it('cancels the name prompt without firing a POST', async () => {
        apiMock.get.mockResolvedValueOnce({ data: SAMPLE_TEMPLATES });
        renderWithRouter(<ProjectTemplateGallery />);
        const user = userEvent.setup();
        await user.click(
            await screen.findByTestId('project-template-pick-ticket-router'),
        );
        await user.click(
            screen.getByTestId('project-template-cancel-ticket-router'),
        );

        expect(
            screen.queryByTestId('project-template-name-form-ticket-router'),
        ).not.toBeInTheDocument();
        expect(apiMock.post).not.toHaveBeenCalled();
    });

    it('renders the empty-state when no templates exist + hideWhenEmpty is false', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { templates: [], count: 0 } });
        renderWithRouter(<ProjectTemplateGallery />);

        await waitFor(() => {
            expect(
                screen.getByTestId('project-template-gallery-empty'),
            ).toBeInTheDocument();
        });
    });

    it('renders nothing when hideWhenEmpty is true and templates list is empty', async () => {
        apiMock.get.mockResolvedValueOnce({ data: { templates: [], count: 0 } });
        const { container } = renderWithRouter(
            <ProjectTemplateGallery hideWhenEmpty />,
        );

        await waitFor(() => {
            expect(
                screen.queryByTestId('project-template-gallery-loading'),
            ).not.toBeInTheDocument();
        });
        expect(container.firstChild).toBeNull();
    });
});
