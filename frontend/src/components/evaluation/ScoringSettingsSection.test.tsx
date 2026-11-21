/**
 * Results › Verdict › How it's scored › "Change how it's scored" (UI simplification pass, phase 4):
 * the scoring settings that lived on Train › More options, with the same show rules, plus the
 * project's null-as-absent switch.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { state, apiMock } = vi.hoisted(() => ({
    state: { profile: null as string | null, project: null as unknown },
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../api/toolScoring', () => ({
    getToolScoringConfig: vi.fn(async () => ({ config: {}, effective: null, catalog: null, effective_task_profile: state.profile, row_context_reads: [] })),
}));
vi.mock('../../stores/projectStore', () => ({
    useProjectStore: (selector: (s: { activeProject: unknown }) => unknown) => selector({ activeProject: state.project }),
}));
vi.mock('../training/ToolScoringConfigCard', () => ({ default: ({ isPlan }: { isPlan?: boolean }) => <div data-testid="card-tool-scoring">plan={String(Boolean(isPlan))}</div> }));
vi.mock('../training/StructuredPlanConfigCard', () => ({ default: () => <div data-testid="card-structured-plan" /> }));
vi.mock('../training/RemediationConfigCard', () => ({ default: () => <div data-testid="card-remediation" /> }));
vi.mock('../training/SqlExecutionConfigCard', () => ({ default: () => <div data-testid="card-sql-execution" /> }));
vi.mock('../training/AutoLiftSettingCard', () => ({ default: () => <div data-testid="card-auto-lift" /> }));

import HowItsScoredCard from './HowItsScoredCard';
import ScoringSettingsSection, { isScoringSettingId } from './ScoringSettingsSection';

function project(recipeId: string) {
    return { id: 4, name: 'p', selected_recipe: { recipe_id: recipeId } };
}

describe('ScoringSettingsSection', () => {
    beforeEach(() => {
        state.profile = null;
        state.project = project('qa-sft');
        apiMock.get.mockReset();
        apiMock.put.mockReset();
        apiMock.get.mockImplementation(async (url: string) => (url.includes('null-as-absent') ? { data: { enabled: false } } : { data: {} }));
    });

    it('a plain project: null as absent, SQL execution, automatic check; no task-specific cards', async () => {
        render(<ScoringSettingsSection projectId={4} />);
        expect(screen.getByTestId('card-sql-execution')).toBeInTheDocument();
        expect(screen.getByTestId('card-auto-lift')).toBeInTheDocument();
        expect(await screen.findByTestId('null-as-absent-project-toggle')).not.toBeChecked();
        expect(screen.queryByTestId('card-tool-scoring')).toBeNull();
        expect(screen.queryByTestId('card-structured-plan')).toBeNull();
        expect(screen.queryByTestId('card-remediation')).toBeNull();
    });

    it('tool calling (backend-reported) shows tool-call scoring', async () => {
        state.profile = 'tool_calling';
        render(<ScoringSettingsSection projectId={4} />);
        expect(await screen.findByTestId('card-tool-scoring')).toHaveTextContent('plan=false');
    });

    it('structured plan shows tool-call scoring (plan) and the plan rubric; remediation its checks', () => {
        state.project = project('structured-plan');
        const view = render(<ScoringSettingsSection projectId={4} />);
        expect(screen.getByTestId('card-tool-scoring')).toHaveTextContent('plan=true');
        expect(screen.getByTestId('card-structured-plan')).toBeInTheDocument();
        view.unmount();
        state.project = project('code-remediation');
        render(<ScoringSettingsSection projectId={4} />);
        expect(screen.getByTestId('card-remediation')).toBeInTheDocument();
    });

    it('saves the project null-as-absent switch with the same body as before', async () => {
        apiMock.put.mockResolvedValue({ data: { enabled: true } });
        const user = userEvent.setup();
        render(<ScoringSettingsSection projectId={4} />);
        await user.click(await screen.findByTestId('null-as-absent-project-toggle'));
        await waitFor(() => expect(apiMock.put).toHaveBeenCalledWith('/projects/4/settings/null-as-absent', { enabled: true }));
    });

    it('?open=how-scored-settings opens the section on the card (no run selected still shows it)', () => {
        render(<HowItsScoredCard projectId={4} runLabel={null} results={[]} openSection="how-scored-settings" />);
        expect(screen.getByTestId('how-scored')).toHaveTextContent('No run selected.');
        expect(screen.getByTestId('how-scored-settings-body')).toBeInTheDocument();
        expect(screen.getByTestId('how-scored-setting-auto-lift')).toBeInTheDocument();
    });

    it('knows its setting ids', () => {
        expect(isScoringSettingId('tool-scoring')).toBe(true);
        expect(isScoringSettingId('auto-lift')).toBe(true);
        expect(isScoringSettingId('models')).toBe(false);
        expect(isScoringSettingId(null)).toBe(false);
    });
});
