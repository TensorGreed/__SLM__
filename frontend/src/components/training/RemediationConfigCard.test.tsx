import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));

import RemediationConfigCard from './RemediationConfigCard';

const CONFIG = {
    config: { detector: null, checks: null, judge: null, retrieve_passages: null, top_k: null },
    effective: { judge: true, retrieve_passages: true, top_k: 3 },
    detector_label: null,
    available_detectors: [
        { project_id: 7, project_name: 'Findings tagger', experiment_id: 31, run_name: 'r', completed_at: null },
    ],
};

const CHECKS = {
    checks: [
        { id: 'python_ast', languages: ['python'], kind: 'builtin', binary: null, available: true, allowed: true },
        { id: 'java_syntax', languages: ['java'], kind: 'external', binary: 'javac', available: true, allowed: false },
        { id: 'shellcheck', languages: ['shell'], kind: 'external', binary: 'shellcheck', available: false, allowed: true },
    ],
    network_isolation: { mode: 'auto', available: false },
};

function mockGets(config: unknown = CONFIG, checks: unknown = CHECKS) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('remediation-config')) return { data: config };
        if (url.includes('code-checks')) return { data: checks };
        return { data: {} };
    });
}

describe('RemediationConfigCard', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.put.mockReset();
    });

    it('lists checks with availability and the detector with its run id', async () => {
        mockGets();
        render(<RemediationConfigCard projectId={4} />);
        expect(await screen.findByTestId('rem-check-python_ast')).toHaveTextContent('ready');
        expect(screen.getByTestId('rem-check-java_syntax')).toHaveTextContent('not allowed');
        expect(screen.getByTestId('rem-check-shellcheck')).toHaveTextContent('not installed');
        expect(screen.getByTestId('rem-isolation')).toHaveTextContent('not available on this machine');
        expect(screen.getByRole('option', { name: 'Findings tagger · run 31' })).toBeInTheDocument();
        expect(screen.getByTestId('rem-top-k')).toHaveValue(3);
    });

    it('maps a stored detector without a run id to the project\'s latest available run', async () => {
        mockGets({
            ...CONFIG,
            config: { ...CONFIG.config, detector: { kind: 'project', project_id: 7, experiment_id: null } },
            detector_label: 'Findings tagger · run 31',
        });
        render(<RemediationConfigCard projectId={4} />);
        const select = await screen.findByTestId('rem-detector-select');
        expect(select).toHaveValue('project:7:31');
        expect(screen.queryByTestId('rem-detector-stored-option')).not.toBeInTheDocument();
    });

    it('shows a stored detector that no option matches, labelled from the backend', async () => {
        mockGets({
            ...CONFIG,
            config: { ...CONFIG.config, detector: { kind: 'project', project_id: 9, experiment_id: 12 } },
            detector_label: 'Old tagger · run 12',
        });
        render(<RemediationConfigCard projectId={4} />);
        const select = await screen.findByTestId('rem-detector-select');
        expect(select).toHaveValue('project:9:12');
        expect(screen.getByTestId('rem-detector-stored-option')).toHaveTextContent('Old tagger · run 12');
    });

    it('falls back to a project / run label when the backend has none', async () => {
        mockGets({
            ...CONFIG,
            config: { ...CONFIG.config, detector: { kind: 'project', project_id: 9, experiment_id: 12 } },
            detector_label: null,
        });
        render(<RemediationConfigCard projectId={4} />);
        await screen.findByTestId('rem-detector-select');
        expect(screen.getByTestId('rem-detector-stored-option')).toHaveTextContent('project 9 · run 12 (not available)');
    });

    it('flags ready external checks when the network is not isolated', async () => {
        mockGets(CONFIG, {
            checks: [
                { id: 'python_ast', languages: ['python'], kind: 'builtin', binary: null, available: true, allowed: true },
                { id: 'go_syntax', languages: ['go'], kind: 'external', binary: 'gofmt', available: true, allowed: true },
            ],
            network_isolation: { mode: 'auto', available: false },
        });
        render(<RemediationConfigCard projectId={4} />);
        expect(await screen.findByTestId('rem-check-go_syntax-network')).toHaveTextContent('network not isolated');
        expect(screen.queryByTestId('rem-check-python_ast-network')).not.toBeInTheDocument();
    });

    it('saves the picked detector, checks, toggles and top_k', async () => {
        mockGets();
        apiMock.put.mockResolvedValue({
            data: { ...CONFIG, detector_label: 'Findings tagger · run 31' },
        });
        const user = userEvent.setup();
        render(<RemediationConfigCard projectId={4} />);
        await screen.findByTestId('rem-detector-select');
        await user.selectOptions(screen.getByTestId('rem-detector-select'), 'project:7:31');
        await user.click(screen.getByTestId('rem-all-checks'));
        await user.click(screen.getByLabelText('Use check python_ast'));
        await user.click(screen.getByTestId('rem-judge-toggle'));
        fireEvent.change(screen.getByTestId('rem-top-k'), { target: { value: '5' } });
        await user.click(screen.getByTestId('rem-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        expect(apiMock.put).toHaveBeenCalledWith('/projects/4/remediation-config', {
            detector: { kind: 'project', project_id: 7, experiment_id: 31 },
            checks: ['python_ast'],
            judge: false,
            retrieve_passages: true,
            top_k: 5,
        });
    });

    it('score task->code rows is off by default and saved only when turned on', async () => {
        mockGets();
        apiMock.put.mockResolvedValue({ data: CONFIG });
        const user = userEvent.setup();
        render(<RemediationConfigCard projectId={4} />);
        const toggle = await screen.findByTestId('rem-score-code-examples');
        expect(toggle).not.toBeChecked();
        await user.click(toggle);
        await user.click(screen.getByTestId('rem-save'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalled());
        expect(apiMock.put.mock.calls[0][1]).toMatchObject({ score_code_examples: true });
    });

    it('shows the stored score task->code setting', async () => {
        mockGets({ ...CONFIG, config: { ...CONFIG.config, score_code_examples: true }, effective: { ...CONFIG.effective, score_code_examples: true } });
        render(<RemediationConfigCard projectId={4} />);
        expect(await screen.findByTestId('rem-score-code-examples')).toBeChecked();
    });

    it('shows a validation error from the server', async () => {
        mockGets();
        apiMock.put.mockRejectedValue({ response: { data: { message: 'detector project has no completed run' } } });
        const user = userEvent.setup();
        render(<RemediationConfigCard projectId={4} />);
        await user.click(await screen.findByTestId('rem-save'));
        expect(await screen.findByTestId('rem-save-error')).toHaveTextContent('detector project has no completed run');
    });
});
