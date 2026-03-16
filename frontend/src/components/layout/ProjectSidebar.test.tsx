import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';

import ProjectSidebar from './ProjectSidebar';
import type { PipelineStatusResponse, PipelineStage } from '../../types';

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

function renderSidebar(initialEntries: Array<string | { pathname: string; state?: unknown }>) {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route
          path="/project/:id/*"
          element={
            <>
              <ProjectSidebar projectId={1} projectName="Demo" pipelineStatus={PIPELINE_STATUS} />
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
    ['/project/1/guide', undefined, 'Start and Discover'],
    ['/project/1/wizard', undefined, 'Start and Discover'],
    ['/project/1/wizard', { sidebarRail: 'training' }, 'Model Configuration'],
    ['/project/1/pipeline/data', undefined, 'Runs and Stages'],
    ['/project/1/pipeline/training', undefined, 'Model Configuration'],
    ['/project/1/training-config', undefined, 'Model Configuration'],
    ['/project/1/playground', undefined, 'Prompt Testing'],
    ['/project/1/workflow', undefined, 'Recipes and Flows'],
    ['/project/1/recipes', undefined, 'Recipes and Flows'],
    ['/project/1/domain', undefined, 'Packs and Profiles'],
    ['/project/1/domain/packs', undefined, 'Packs and Profiles'],
    ['/project/1/domain/profiles', undefined, 'Packs and Profiles'],
  ])('maps %s to expected rail heading', (pathname, state, heading) => {
    renderSidebar([{ pathname, state }]);
    expect(screen.getByText(String(heading))).toBeInTheDocument();
  });

  it('keeps training context when moving training-config -> guided setup -> training stage', async () => {
    const user = userEvent.setup();
    renderSidebar(['/project/1/training-config']);

    await user.click(screen.getByRole('button', { name: 'Guided Setup' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/project/1/wizard');
    expect(screen.getByText('Model Configuration')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Training Stage' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/project/1/pipeline/training');
    expect(screen.getByText('Model Configuration')).toBeInTheDocument();
  });

  it('keeps home context for home wizard and then lands in pipeline data correctly', async () => {
    const user = userEvent.setup();
    renderSidebar(['/project/1/guide']);

    await user.click(screen.getByRole('button', { name: 'Wizard Mode' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/project/1/wizard');
    expect(screen.getByText('Start and Discover')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Go to Data Pipeline' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/project/1/pipeline/data');
    expect(screen.getByText('Runs and Stages')).toBeInTheDocument();
  });
});

