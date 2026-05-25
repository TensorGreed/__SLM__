import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import DataStudioQualitySafetyPanel from './DataStudioQualitySafetyPanel';

const qualityPayload = {
    project_id: 1,
    verdict: 'blocked',
    read_only: true,
    auto_apply: false,
    source_of_truth: 'deterministic_data_studio_checks',
    summary: {
        scanned_rows: 24,
        sampled_rows: 24,
        blocker_count: 2,
        warning_count: 3,
        info_count: 4,
        pii_pci_signal_count: 3,
        duplicate_signal_count: 2,
        leakage_overlap_count: 1,
        low_quality_signal_count: 1,
        pending_review_count: 5,
        domain_signal_count: 1,
        domain_authored_check_count: 2,
        domain_authored_warning_count: 1,
        domain_authored_blocker_count: 0,
    },
    domain: {
        id: 'policy_qa',
        label: 'Policy Q&A',
        confidence: 0.86,
        source: 'sampled_data',
    },
    checks: [
        {
            id: 'pii_pci_sensitive_values',
            label: 'PII/PCI patterns detected',
            category: 'safety',
            status: 'blocked',
            severity: 'blocker',
            message: 'Found 3 deterministic sensitive-data signal(s). Values are not shown in Data Studio.',
            count: 3,
            target_tab: 'data',
            workflow_owner: 'Source Ingestion',
            source: 'policy.csv',
            domain_id: 'policy_qa',
            domain_label: 'Policy Q&A',
            evidence: ['Regex checks cover common PII/PCI patterns.'],
            action_label: 'Inspect sources',
        },
        {
            id: 'train_validation_test_leakage',
            label: 'Train/validation/test leakage risk',
            category: 'leakage',
            status: 'blocked',
            severity: 'blocker',
            message: 'Found 1 overlapping row fingerprint(s) across prepared splits.',
            count: 1,
            target_tab: 'dataprep',
            workflow_owner: 'Data Prep',
            source: 'Prepared splits',
            domain_id: 'policy_qa',
            domain_label: 'Policy Q&A',
            evidence: ['train/validation: 1 overlapping row(s)'],
            action_label: 'Refresh splits',
        },
        {
            id: 'domain_authored_required_coverage',
            label: 'Domain-required field coverage',
            category: 'domain-authored',
            status: 'attention',
            severity: 'warning',
            message: 'Applied domain contract requires stronger field coverage before training.',
            count: 1,
            target_tab: 'dataprep',
            workflow_owner: 'Domain Managers',
            source: 'Applied domain contract',
            domain_id: 'policy_qa',
            domain_label: 'Policy Q&A',
            evidence: ['context: 0% < 70%'],
            action_label: 'Review mapping',
            domain_authored: true,
            read_only_preview: true,
        },
        {
            id: 'domain_authored_review_gate',
            label: 'Domain review gate',
            category: 'domain-authored',
            status: 'ready',
            severity: 'info',
            message: 'Applied domain contract requires review gates.',
            count: 0,
            target_tab: 'domain',
            workflow_owner: 'Domain Managers',
            source: 'Applied domain contract',
            domain_id: 'policy_qa',
            domain_label: 'Policy Q&A',
            evidence: [],
            action_label: 'Open Review',
            domain_authored: true,
            read_only_preview: true,
        },
    ],
    findings_by_source: [
        {
            key: 'policy-csv',
            label: 'policy.csv',
            blocker_count: 1,
            warning_count: 1,
            info_count: 0,
            total: 2,
            target_tab: 'data',
        },
        {
            key: 'prepared-splits',
            label: 'Prepared splits',
            blocker_count: 1,
            warning_count: 0,
            info_count: 0,
            total: 1,
            target_tab: 'dataprep',
        },
    ],
    findings_by_status: [
        { status: 'blocked', label: 'Blockers', count: 2, target_tab: 'data' },
        { status: 'attention', label: 'Warnings', count: 3, target_tab: 'dataprep' },
        { status: 'ready', label: 'Ready checks', count: 4, target_tab: 'data' },
    ],
    findings_by_domain: [
        {
            key: 'policy-qa',
            label: 'Policy Q&A',
            blocker_count: 2,
            warning_count: 3,
            info_count: 4,
            total: 9,
            target_tab: 'domain',
        },
    ],
    findings_by_owner: [
        {
            key: 'source-ingestion',
            label: 'Source Ingestion',
            blocker_count: 1,
            warning_count: 1,
            info_count: 1,
            total: 3,
            target_tab: 'data',
        },
        {
            key: 'data-prep',
            label: 'Data Prep',
            blocker_count: 1,
            warning_count: 1,
            info_count: 1,
            total: 3,
            target_tab: 'dataprep',
        },
    ],
    issues: [],
    entry_points: [
        {
            label: 'Open Source Ingestion',
            target_tab: 'data',
            reason: 'Inspect source rows.',
            requires_confirmation: true,
        },
        {
            label: 'Open Data Prep',
            target_tab: 'dataprep',
            reason: 'Review leakage checks.',
            requires_confirmation: true,
        },
        {
            label: 'Open Domain Managers',
            target_tab: 'domain',
            reason: 'Tune policy checks.',
            requires_confirmation: true,
        },
    ],
    assist: {
        available: true,
        default_provider: 'ollama',
        openai_compatible_supported: true,
        purpose: 'explanations_only',
        auto_apply: false,
        target_tab: 'assist',
    },
    domain_authored: {
        available: true,
        preview_only: true,
        applied_profile_id: 'policy-qa-profile-v1',
        applied_profile_source: 'project',
        applied_pack_id: 'policy-qa-pack-v1',
        applied_pack_source: 'project',
        check_count: 2,
        failing_count: 1,
        blocker_count: 0,
        warning_count: 1,
        ready_count: 1,
        supported_sources: ['profile:data_quality', 'profile:audit'],
    },
    power_details: {},
};

describe('DataStudioQualitySafetyPanel', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
    });

    it('renders deterministic quality and safety scan findings without mutating', async () => {
        apiMock.get.mockResolvedValueOnce({ data: qualityPayload });
        const onOpenTarget = vi.fn();

        render(<DataStudioQualitySafetyPanel projectId={1} onOpenTarget={onOpenTarget} />);

        await waitFor(() => {
            expect(screen.getByTestId('data-studio-quality-safety')).toBeInTheDocument();
        });

        expect(screen.getByText('Blocked')).toBeInTheDocument();
        expect(screen.getByText('Policy Q&A')).toBeInTheDocument();
        expect(screen.getByText('86% domain confidence')).toBeInTheDocument();
        expect(screen.getByText('PII/PCI patterns detected')).toBeInTheDocument();
        expect(screen.getByText('Train/validation/test leakage risk')).toBeInTheDocument();
        expect(screen.getByText('By workflow owner')).toBeInTheDocument();
        expect(screen.getByText('Source Ingestion')).toBeInTheDocument();
        expect(screen.getByText('By source')).toBeInTheDocument();
        expect(screen.getByText('policy.csv')).toBeInTheDocument();
        expect(screen.getByText('Domain-authored previews')).toBeInTheDocument();
        expect(screen.getByText(/2 checks from policy-qa-profile-v1/i)).toBeInTheDocument();
        expect(screen.getByText('Domain-required field coverage')).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: /Inspect sources/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('data');

        fireEvent.click(screen.getByRole('button', { name: /Explain with Ollama/i }));
        expect(onOpenTarget).toHaveBeenCalledWith('assist');
        expect(apiMock.get).toHaveBeenCalledWith('/projects/1/data-studio/quality-safety');
    });
});
