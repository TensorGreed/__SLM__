import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioDomainDetectionPanel from './DataStudioDomainDetectionPanel';

const domainPayload = {
    project_id: 1,
    verdict: 'attention',
    detected_domain: {
        id: 'support_faq',
        label: 'Support FAQ',
        confidence: 0.86,
        confidence_label: 'high',
        source: 'sampled_data',
        summary: 'Detected from source rows.',
        matched_keywords: ['refund', 'password reset', 'ticket'],
        matched_fields: ['question', 'answer'],
        recommended_recipes: ['qa-sft', 'classification'],
    },
    applied: {
        profile_id: 'generic-domain-v1',
        profile_source: 'project',
        profile_display_name: 'Generic Domain',
        profile_version: '1.0.0',
        pack_id: 'general-pack-v1',
        pack_source: 'project',
        pack_display_name: 'General Domain Pack',
        pack_version: '1.0.0',
        pack_default_profile_id: 'generic-domain-v1',
    },
    recipe: {
        id: 'qa-sft',
        name: 'Question & Answer Assistant',
        task_profile: 'qa',
    },
    source: {
        dataset_type: 'raw',
        dataset_id: 10,
        dataset_name: 'Support FAQ',
        document_id: 20,
        document_name: 'support_faq.jsonl',
        document_count: 1,
        row_count: 2,
        sampled_records: 2,
    },
    evidence: [
        {
            id: 'field_signals',
            title: 'Column signals',
            message: 'Fields match this domain: question, answer.',
            score: 0.7,
        },
        {
            id: 'term_signals',
            title: 'Content signals',
            message: 'Sampled rows mention: refund, password reset, ticket.',
            score: 0.86,
        },
    ],
    suggested_actions: [
        {
            id: 'domain_action_1',
            label: 'Add customer phrasing variants for the top support topics.',
            target_tab: 'synthetic',
        },
    ],
    risks: [
        {
            id: 'domain_risk_1',
            severity: 'warning',
            title: 'Domain risk',
            message: 'Support data often contains personal account details.',
        },
    ],
    issues: [
        {
            id: 'domain_candidate_not_applied',
            severity: 'warning',
            title: 'Specific domain not applied',
            message: 'Sampled rows look like Support FAQ, but the project is still using generic domain defaults.',
            action_label: 'Review domain settings',
            target_tab: 'data',
        },
    ],
    domain_setup: {
        available: true,
        recommended: true,
        reason: 'Sampled rows look like Support FAQ, but the project is still using generic domain defaults.',
        read_only: true,
        requires_confirmation: true,
        create_mode: 'create_missing_drafts',
        detected_domain_id: 'support_faq',
        detected_domain_label: 'Support FAQ',
        profile_id: 'support-faq-profile-v1',
        pack_id: 'support-faq-pack-v1',
        profile_exists: false,
        pack_exists: false,
        profile_status: null,
        pack_status: null,
        can_create_profile: true,
        can_create_pack: true,
        guidance: [
            {
                id: 'task_shape',
                title: 'Task shape',
                recommendation: 'Use a Support FAQ profile aligned to the detected recipe and fields.',
                why: 'Domain profiles make required fields, splits, metrics, and review gates explicit.',
            },
            {
                id: 'coverage',
                title: 'Coverage',
                recommendation: 'Review required-field coverage and representative edge cases.',
                why: 'A domain setup is most useful when it encodes the examples the model must handle reliably.',
            },
        ],
        choices: [
            {
                id: 'use_existing',
                label: 'Use existing setup',
                target: 'domain',
                detail: 'Open the Domain controls to assign or inspect an existing profile and pack.',
            },
            {
                id: 'create_drafts',
                label: 'Create draft setup',
                target: 'create_drafts',
                detail: 'Create only the missing draft profile/pack records.',
            },
        ],
        profile_contract: {
            '$schema': 'slm.domain-profile/v1',
            profile_id: 'support-faq-profile-v1',
            status: 'draft',
            display_name: 'Support FAQ Profile',
        },
        pack_contract: {
            '$schema': 'slm.domain-pack/v1',
            pack_id: 'support-faq-pack-v1',
            status: 'draft',
            default_profile_id: 'support-faq-profile-v1',
            display_name: 'Support FAQ Pack',
        },
    },
    power_details: {
        signals: ['columns:question,answer', 'terms:refund,password reset,ticket'],
        candidate_domains: [],
        runtime: {},
    },
};

describe('DataStudioDomainDetectionPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders detected domain, applied runtime, evidence, and guidance', async () => {
        apiMock.get.mockResolvedValueOnce({ data: domainPayload });

        render(<DataStudioDomainDetectionPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-domain')).toBeInTheDocument();
        });

        expect(screen.getByText('Needs attention')).toBeInTheDocument();
        expect(screen.getAllByText('Support FAQ').length).toBeGreaterThan(0);
        expect(screen.getByText('86%')).toBeInTheDocument();
        expect(screen.getByText('Generic Domain')).toBeInTheDocument();
        expect(screen.getByText('General Domain Pack')).toBeInTheDocument();
        expect(screen.getAllByText('support_faq.jsonl', { exact: false }).length).toBeGreaterThan(0);
        expect(screen.getByText('Column signals')).toBeInTheDocument();
        expect(screen.getByText(/Add customer phrasing variants/i)).toBeInTheDocument();
        expect(screen.getByText('Specific domain not applied')).toBeInTheDocument();
        expect(screen.getByTestId('data-studio-domain-setup')).toBeInTheDocument();
        expect(screen.getByText('Create Support FAQ setup from detection')).toBeInTheDocument();
        expect(screen.getByText('support-faq-profile-v1')).toBeInTheDocument();
        expect(screen.getByText('support-faq-pack-v1')).toBeInTheDocument();
        expect(screen.getByText('Task shape')).toBeInTheDocument();
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/domain-detection');
    });

    it('creates missing setup drafts only after confirmation', async () => {
        vi.spyOn(window, 'confirm').mockReturnValueOnce(true);
        apiMock.get.mockResolvedValue({ data: domainPayload });
        apiMock.post.mockResolvedValueOnce({
            data: {
                status: 'created',
                project_id: 1,
                detected_domain_id: 'support_faq',
                detected_domain_label: 'Support FAQ',
                created_profile: true,
                created_pack: true,
                assigned_to_project: false,
                profile: {
                    profile_id: 'support-faq-profile-v1',
                    display_name: 'Support FAQ Profile',
                    status: 'draft',
                    version: '1.0.0',
                },
                pack: {
                    pack_id: 'support-faq-pack-v1',
                    display_name: 'Support FAQ Pack',
                    status: 'draft',
                    version: '1.0.0',
                    default_profile_id: 'support-faq-profile-v1',
                },
                next_targets: ['domain', 'domain-packs', 'domain-profiles'],
            },
        });

        render(<DataStudioDomainDetectionPanel projectId={1} />);

        const createButton = await screen.findByRole('button', { name: /Create draft setup/i });
        fireEvent.click(createButton);

        await waitFor(() => {
            expect(apiMock.post).toHaveBeenCalledWith(
                '/projects/1/data-studio/domain-detection/domain-setup',
                { confirm: true },
            );
        });
        expect(await screen.findByText(/Created profile draft support-faq-profile-v1/i)).toBeInTheDocument();
    });

    it('routes users to existing domain managers', async () => {
        const onOpenTarget = vi.fn();
        apiMock.get.mockResolvedValueOnce({ data: domainPayload });

        render(<DataStudioDomainDetectionPanel projectId={1} onOpenTarget={onOpenTarget} />);

        fireEvent.click(await screen.findByRole('button', { name: /Use existing setup/i }));
        fireEvent.click(screen.getByRole('button', { name: /Pack manager/i }));
        fireEvent.click(screen.getByRole('button', { name: /Profile manager/i }));

        expect(onOpenTarget).toHaveBeenNthCalledWith(1, 'domain');
        expect(onOpenTarget).toHaveBeenNthCalledWith(2, 'domain-packs');
        expect(onOpenTarget).toHaveBeenNthCalledWith(3, 'domain-profiles');
    });

    it('renders generic empty-state domain guidance', async () => {
        apiMock.get.mockResolvedValueOnce({
            data: {
                ...domainPayload,
                verdict: 'attention',
                detected_domain: {
                    ...domainPayload.detected_domain,
                    id: 'generic_domain',
                    label: 'Generic Domain',
                    confidence: 0.25,
                    confidence_label: 'low',
                    source: 'runtime_default',
                    matched_keywords: [],
                    matched_fields: [],
                },
                source: null,
                evidence: [
                    {
                        id: 'generic_runtime',
                        title: 'Generic runtime',
                        message: 'The project is using the generic domain defaults.',
                        score: 0.25,
                    },
                ],
                suggested_actions: [
                    {
                        id: 'domain_action_1',
                        label: 'Add representative source rows so BrewSLM can infer the domain.',
                        target_tab: 'data',
                    },
                ],
                risks: [],
                issues: [
                    {
                        id: 'domain_needs_source_evidence',
                        severity: 'warning',
                        title: 'Domain evidence is limited',
                        message: 'Add source rows so BrewSLM can confirm the training domain from real examples.',
                        action_label: 'Add sources',
                        target_tab: 'data',
                    },
                ],
                domain_setup: null,
            },
        });

        render(<DataStudioDomainDetectionPanel projectId={1} />);

        await waitFor(() => {
            expect(screen.getByText('Domain evidence is limited')).toBeInTheDocument();
        });

        expect(screen.getByText('25%')).toBeInTheDocument();
        expect(screen.getByText('No source sample yet')).toBeInTheDocument();
        expect(screen.getByText(/Add representative source rows/i)).toBeInTheDocument();
        expect(screen.queryByTestId('data-studio-domain-setup')).not.toBeInTheDocument();
    });
});
