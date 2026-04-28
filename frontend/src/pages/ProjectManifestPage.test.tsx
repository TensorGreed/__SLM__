import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, navigateMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
    navigateMock: vi.fn(),
}));

vi.mock('../api/client', () => ({ default: apiMock }));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => ({
            projectId: 42,
            project: { id: 42, name: 'Demo Project' },
            pipelineStatus: null,
            refreshPipelineStatus: vi.fn(),
        }),
        useParams: () => ({ id: '42' }),
        useNavigate: () => navigateMock,
    };
});

import ProjectManifestPage from './ProjectManifestPage';

const SAMPLE_MANIFEST_YAML = `api_version: brewslm/v1
kind: Project
metadata:
  name: Demo Project
spec:
  workflow:
    target_profile_id: edge_gpu
`;

const SUMMARY_MANIFEST = {
    api_version: 'brewslm/v1',
    kind: 'Project',
    metadata: { name: 'Demo Project', description: '', labels: {} },
    spec: {
        workflow: { target_profile_id: 'edge_gpu' },
        blueprint: { domain_name: 'support_chat' },
        domain: {},
        model: {},
        data_sources: [
            { name: 'tickets_raw', type: 'raw' },
            { name: 'gold_dev', type: 'gold_dev' },
        ],
        adapters: [{ name: 'qa_adapter', version: 1 }],
        training_plan: {},
        eval_pack: {},
        export: {},
        deployment: {},
    },
};

const VALIDATION_OK = { ok: true, errors: [], warnings: [] };

const VALIDATION_FAIL = {
    ok: false,
    errors: [
        {
            code: 'UNKNOWN_TARGET_PROFILE',
            severity: 'error',
            field: 'spec.workflow.target_profile_id',
            message: 'Target profile mystery_target is not registered.',
            actionable_fix: 'Pick one of edge_gpu, vllm_server, mobile_cpu.',
        },
    ],
    warnings: [],
};

const DIFF_PLAN = {
    project_id: 42,
    project_name: 'Demo Project',
    api_version: 'brewslm/v1',
    actions: [
        {
            target: 'project',
            operation: 'noop',
            name: 'Demo Project',
            fields_changed: [],
            reason: '',
        },
        {
            target: 'data_source',
            operation: 'create',
            name: 'tickets_raw',
            fields_changed: [],
            reason: 'data_source_in_manifest_only',
        },
        {
            target: 'data_source',
            operation: 'update',
            name: 'gold_dev',
            fields_changed: ['record_count', 'description'],
            reason: 'data_source_drift',
        },
    ],
    warnings: ['adapter_not_in_manifest:legacy_adapter'],
    summary: { create: 1, update: 1, noop: 1 },
};

const APPLY_RESULT = {
    project_id: 42,
    project_name: 'Demo Project',
    plan_only: false,
    plan: DIFF_PLAN,
    validation: VALIDATION_OK,
    applied_actions: DIFF_PLAN.actions,
};

const APPLY_RESULT_PLAN_ONLY = {
    ...APPLY_RESULT,
    plan_only: true,
    applied_actions: [],
};

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();
    navigateMock.mockReset();
});

describe('ProjectManifestPage', () => {
    function mockSummaryEndpoint() {
        apiMock.get.mockImplementation(async (url: string, config?: { params?: { format?: string } }) => {
            if (url.includes('/manifest/export')) {
                if (config?.params?.format === 'json') {
                    return { data: SUMMARY_MANIFEST };
                }
                return { data: 'api_version: brewslm/v1\nkind: Project\n' };
            }
            throw new Error(`Unexpected GET ${url}`);
        });
    }

    it('drop-zone accepts a file and populates the textarea', async () => {
        mockSummaryEndpoint();
        render(<ProjectManifestPage />);
        const textarea = await screen.findByLabelText(/manifest text/i);
        const fileInput = screen.getByLabelText(/manifest file input/i);

        const file = new File([SAMPLE_MANIFEST_YAML], 'demo.brewslm.yaml', {
            type: 'application/x-yaml',
        });
        const user = userEvent.setup();
        await user.upload(fileInput, file);

        await waitFor(() => {
            expect((textarea as HTMLTextAreaElement).value).toContain('api_version: brewslm/v1');
        });
        expect(screen.getByText(/Loaded:/i)).toBeInTheDocument();
    });

    it('Validate posts to /manifest/validate and surfaces errors with code + field + actionable_fix', async () => {
        mockSummaryEndpoint();
        apiMock.post.mockResolvedValueOnce({ data: VALIDATION_FAIL });

        const user = userEvent.setup();
        render(<ProjectManifestPage />);

        const textarea = await screen.findByLabelText(/manifest text/i);
        await user.click(textarea);
        await user.paste(SAMPLE_MANIFEST_YAML);

        await user.click(screen.getByRole('button', { name: /^Validate$/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith('/manifest/validate', {
                manifest_yaml: expect.stringContaining('api_version'),
            });
        });

        expect(await screen.findByText(/UNKNOWN_TARGET_PROFILE/)).toBeInTheDocument();
        expect(screen.getByText(/spec.workflow.target_profile_id/)).toBeInTheDocument();
        expect(screen.getByText(/Pick one of edge_gpu/)).toBeInTheDocument();
    });

    it('Validate shows a green ok state for a clean manifest', async () => {
        mockSummaryEndpoint();
        apiMock.post.mockResolvedValueOnce({ data: VALIDATION_OK });

        const user = userEvent.setup();
        render(<ProjectManifestPage />);
        const textarea = await screen.findByLabelText(/manifest text/i);
        await user.click(textarea);
        await user.paste(SAMPLE_MANIFEST_YAML);

        await user.click(screen.getByRole('button', { name: /^Validate$/i }));

        const okStatus = await screen.findByText(/Manifest is clean/i);
        expect(okStatus).toBeInTheDocument();
    });

    it('Diff renders actions grouped by target with operation badges and fields_changed', async () => {
        mockSummaryEndpoint();
        apiMock.post.mockResolvedValueOnce({ data: DIFF_PLAN });

        const user = userEvent.setup();
        render(<ProjectManifestPage />);

        const textarea = await screen.findByLabelText(/manifest text/i);
        await user.click(textarea);
        await user.paste(SAMPLE_MANIFEST_YAML);

        await user.click(screen.getByRole('button', { name: /Diff against project/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/manifest/diff',
                { manifest_yaml: expect.stringContaining('api_version') },
            );
        });

        const diffSection = (await screen.findByText(/Diff plan/i)).closest('section');
        expect(diffSection).not.toBeNull();
        const scoped = within(diffSection as HTMLElement);
        // Group headers
        expect(scoped.getByText(/^project \(1\)$/i)).toBeInTheDocument();
        expect(scoped.getByText(/^data_source \(2\)$/i)).toBeInTheDocument();
        // Operation badges
        expect(scoped.getAllByText('create').length).toBeGreaterThan(0);
        expect(scoped.getAllByText('update').length).toBeGreaterThan(0);
        expect(scoped.getAllByText('noop').length).toBeGreaterThan(0);
        // fields_changed surfaced for the update row
        expect(scoped.getByText(/fields_changed: record_count, description/)).toBeInTheDocument();
        // Plan warning surfaced
        expect(scoped.getByText(/adapter_not_in_manifest:legacy_adapter/)).toBeInTheDocument();
    });

    it('Apply posts to the project apply endpoint and navigates to pipeline/data on success', async () => {
        mockSummaryEndpoint();
        apiMock.post.mockResolvedValueOnce({ data: APPLY_RESULT });

        const user = userEvent.setup();
        render(<ProjectManifestPage />);

        const textarea = await screen.findByLabelText(/manifest text/i);
        await user.click(textarea);
        await user.paste(SAMPLE_MANIFEST_YAML);

        await user.click(screen.getByRole('button', { name: /^Apply$/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/manifest/apply',
                {
                    manifest_yaml: expect.stringContaining('api_version'),
                    plan_only: false,
                },
            );
        });
        await waitFor(() => {
            expect(navigateMock).toHaveBeenCalledWith('/project/42/pipeline/data');
        });
    });

    it('Apply with plan_only=true does not navigate', async () => {
        mockSummaryEndpoint();
        apiMock.post.mockResolvedValueOnce({ data: APPLY_RESULT_PLAN_ONLY });

        const user = userEvent.setup();
        render(<ProjectManifestPage />);

        const textarea = await screen.findByLabelText(/manifest text/i);
        await user.click(textarea);
        await user.paste(SAMPLE_MANIFEST_YAML);

        await user.click(screen.getByLabelText(/Plan-only preview/i));
        await user.click(screen.getByRole('button', { name: /^Apply$/i }));

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/42/manifest/apply',
                {
                    manifest_yaml: expect.stringContaining('api_version'),
                    plan_only: true,
                },
            );
        });
        // No navigation on plan-only
        expect(navigateMock).not.toHaveBeenCalled();
        // Plan-only confirmation surfaces
        expect(await screen.findByText(/Plan-only run completed/i)).toBeInTheDocument();
    });

    it('Apply is disabled when validation has errors', async () => {
        mockSummaryEndpoint();
        apiMock.post.mockResolvedValueOnce({ data: VALIDATION_FAIL });

        const user = userEvent.setup();
        render(<ProjectManifestPage />);

        const textarea = await screen.findByLabelText(/manifest text/i);
        await user.click(textarea);
        await user.paste(SAMPLE_MANIFEST_YAML);

        await user.click(screen.getByRole('button', { name: /^Validate$/i }));
        await screen.findByText(/UNKNOWN_TARGET_PROFILE/);

        const applyBtn = screen.getByRole('button', { name: /^Apply$/i });
        expect(applyBtn).toBeDisabled();
    });

    it('renders the manifest summary card from /export?format=json', async () => {
        mockSummaryEndpoint();
        render(<ProjectManifestPage />);

        await waitFor(() => {
            expect(apiMock.get).toHaveBeenCalledWith(
                '/projects/42/manifest/export',
                expect.objectContaining({ params: { format: 'json' } }),
            );
        });
        expect(await screen.findByText('support_chat')).toBeInTheDocument();
        // 2 data sources + 1 adapter
        expect(screen.getByText('2')).toBeInTheDocument();
        expect(screen.getByText('1')).toBeInTheDocument();
        expect(screen.getByText('edge_gpu')).toBeInTheDocument();
    });
});
