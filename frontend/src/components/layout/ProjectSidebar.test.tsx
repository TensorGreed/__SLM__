import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import ProjectSidebar from './ProjectSidebar';
import type { PipelineStatusResponse, PipelineStage, Project } from '../../types';
import { useProjectStore } from '../../stores/projectStore';

const STAGES: PipelineStage[] = [
  'ingestion',
  'cleaning',
  'gold_set',
  'synthetic',
  'dataset_prep',
  'data_adapter_preview',
  'tokenization',
  'training',
  'evaluation',
  'compression',
  'export',
  'completed',
];

const PIPELINE_STATUS: PipelineStatusResponse = {
  project_id: 1,
  current_stage: 'training',
  progress_percent: 64,
  stages: STAGES.map((stage, index) => ({
    stage,
    display_name: stage,
    index,
    status: index < 7 ? 'completed' : stage === 'training' ? 'active' : 'pending',
  })),
};

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderSidebar(
    initialEntries: Array<string | { pathname: string; state?: unknown }>,
    opts: { beginnerMode?: boolean } = {},
) {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route
          path="/project/:id/*"
          element={
            <>
              <ProjectSidebar
                projectId={1}
                projectName="Demo"
                pipelineStatus={PIPELINE_STATUS}
                beginnerMode={opts.beginnerMode}
              />
              <LocationProbe />
            </>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ProjectSidebar nav matrix', () => {
  it.each([
    ['/project/1/guide', undefined, 'Runs and Stages'],
    ['/project/1/wizard', undefined, 'Runs and Stages'],
    ['/project/1/wizard', { sidebarRail: 'training' }, 'Model Configuration'],
    ['/project/1/pipeline/data', undefined, 'Runs and Stages'],
    ['/project/1/pipeline/training', undefined, 'Model Configuration'],
    ['/project/1/training-config', undefined, 'Model Configuration'],
    ['/project/1/playground', undefined, 'Model Configuration'],
    ['/project/1/workflow', undefined, 'Recipes and Flows'],
    ['/project/1/recipes', undefined, 'Recipes and Flows'],
    ['/project/1/domain', undefined, 'Packs and Profiles'],
    ['/project/1/domain/packs', undefined, 'Packs and Profiles'],
    ['/project/1/domain/profiles', undefined, 'Packs and Profiles'],
  ])('maps %s to expected rail heading', (pathname, state, heading) => {
    renderSidebar([{ pathname, state }]);
    expect(screen.getByText(String(heading))).toBeInTheDocument();
  });

  it('keeps training context when clicking Guided Setup from training-config', async () => {
    const user = userEvent.setup();
    renderSidebar(['/project/1/training-config']);

    await user.click(screen.getByRole('button', { name: 'Guided Setup' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/project/1/wizard');
    expect(screen.getByText('Model Configuration')).toBeInTheDocument();
  });

  it('surfaces Playground as a panel item under the Training rail', async () => {
    const user = userEvent.setup();
    renderSidebar(['/project/1/training-config']);

    await user.click(screen.getByRole('button', { name: 'Playground' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/project/1/playground');
    expect(screen.getByText('Model Configuration')).toBeInTheDocument();
  });
});

describe('ProjectSidebar beginner-mode hiding', () => {
  beforeEach(() => {
    apiMock.put.mockReset();
  });

  it('hides Automation + Domain rails when beginner mode is on', () => {
    renderSidebar(['/project/1/pipeline/data'], { beginnerMode: true });
    expect(screen.queryByRole('button', { name: 'Automation' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Domain' })).not.toBeInTheDocument();
    // Core rails still present.
    expect(screen.getByRole('button', { name: 'Pipeline' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Training' })).toBeInTheDocument();
  });

  it('shows Automation + Domain rails when beginner mode is off', () => {
    renderSidebar(['/project/1/pipeline/data'], { beginnerMode: false });
    expect(screen.getByRole('button', { name: 'Automation' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Domain' })).toBeInTheDocument();
  });

  it('hides Adapter Studio in training panel when beginner mode is on', () => {
    renderSidebar(['/project/1/training-config'], { beginnerMode: true });
    expect(screen.queryByRole('button', { name: 'Adapter Studio' })).not.toBeInTheDocument();
    // Autopilot Planner still available.
    expect(screen.getByRole('button', { name: 'Autopilot Planner' })).toBeInTheDocument();
  });

  it('hides workflow + recipes panel entries even on deep-link', () => {
    renderSidebar(['/project/1/workflow'], { beginnerMode: true });
    expect(screen.queryByRole('button', { name: 'Workflow Builder' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Recipes' })).not.toBeInTheDocument();
  });

  it('hides domain panel entries even on deep-link', () => {
    renderSidebar(['/project/1/domain/packs'], { beginnerMode: true });
    expect(screen.queryByRole('button', { name: 'Domain Packs' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Domain Profiles' })).not.toBeInTheDocument();
  });

  it('shows the beginner-mode badge and leave button', () => {
    renderSidebar(['/project/1/guide'], { beginnerMode: true });
    expect(screen.getByRole('note', { name: /Beginner mode active/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Leave beginner mode/i })).toBeInTheDocument();
  });

  it('leave-beginner button PUTs beginner_mode=false after confirm', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const updatedProject: Project = {
      id: 1,
      name: 'Demo',
      description: null,
      status: 'active',
      pipeline_stage: 'training',
      base_model_name: null,
      domain_pack_id: null,
      domain_profile_id: null,
      beginner_mode: false,
      created_at: '2026-04-23T00:00:00Z',
      updated_at: '2026-04-23T00:00:00Z',
    };
    apiMock.put.mockResolvedValue({ data: updatedProject });

    const user = userEvent.setup();
    renderSidebar(['/project/1/guide'], { beginnerMode: true });

    await user.click(screen.getByRole('button', { name: /Leave beginner mode/i }));
    await waitFor(() => {
      expect(apiMock.put).toHaveBeenCalledWith('/projects/1', { beginner_mode: false });
    });
    expect(useProjectStore.getState().activeProject).toEqual(updatedProject);
    confirmSpy.mockRestore();
  });

  it('leave-beginner is cancelled if confirm returns false', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const user = userEvent.setup();
    renderSidebar(['/project/1/guide'], { beginnerMode: true });

    await user.click(screen.getByRole('button', { name: /Leave beginner mode/i }));
    expect(apiMock.put).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it('surfaces errors from the leave-beginner call', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    apiMock.put.mockRejectedValue({ response: { data: { detail: 'db is down' } } });

    const user = userEvent.setup();
    renderSidebar(['/project/1/guide'], { beginnerMode: true });

    await user.click(screen.getByRole('button', { name: /Leave beginner mode/i }));
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/db is down/);
    confirmSpy.mockRestore();
  });

  it('shows Enter beginner mode button when beginner mode is off and hides Leave button', () => {
    renderSidebar(['/project/1/pipeline/data'], { beginnerMode: false });
    expect(screen.getByRole('button', { name: /Enter beginner mode/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Leave beginner mode/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('note', { name: /Beginner mode active/i })).not.toBeInTheDocument();
  });

  it('enter-beginner button PUTs beginner_mode=true after confirm', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const updatedProject: Project = {
      id: 1,
      name: 'Demo',
      description: null,
      status: 'active',
      pipeline_stage: 'training',
      base_model_name: null,
      domain_pack_id: null,
      domain_profile_id: null,
      beginner_mode: true,
      created_at: '2026-04-23T00:00:00Z',
      updated_at: '2026-04-23T00:00:00Z',
    };
    apiMock.put.mockResolvedValue({ data: updatedProject });

    const user = userEvent.setup();
    renderSidebar(['/project/1/pipeline/data'], { beginnerMode: false });

    await user.click(screen.getByRole('button', { name: /Enter beginner mode/i }));
    await waitFor(() => {
      expect(apiMock.put).toHaveBeenCalledWith('/projects/1', { beginner_mode: true });
    });
    confirmSpy.mockRestore();
  });
});

afterEach(() => {
  useProjectStore.setState({ activeProject: null, pipelineStatus: null });
});

