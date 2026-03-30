import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
  },
}));

vi.mock('../api/client', () => ({
  default: apiMock,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({ projectId: 77, project: { id: 77, name: 'Test' }, pipelineStatus: null }),
  };
});

import ProjectAdapterStudioPage from './ProjectAdapterStudioPage';

describe('ProjectAdapterStudioPage', () => {
  beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.post.mockReset();

    apiMock.post.mockImplementation(async (url: string) => {
      if (url === '/projects/77/adapter-studio/infer') {
        return {
          data: {
            profile: {
              schema: {
                fields: [
                  { path: 'prompt', inferred_type: 'string', null_rate: 0.0, sensitive: false },
                  { path: 'response', inferred_type: 'string', null_rate: 0.0, sensitive: false },
                ],
              },
            },
            inference: {
              resolved_adapter_id: 'qa-pair',
              resolved_task_profile: 'qa',
              confidence: 0.88,
              adapter_contract: {
                task_profiles: ['qa', 'instruction_sft'],
                preferred_training_tasks: ['causal_lm'],
                output_contract: { required_fields: ['text', 'question', 'answer'] },
              },
              mapping_canvas: {
                question: 'prompt',
                answer: 'response',
              },
              drop_analysis: {
                sampled_records: 10,
                mapped_records: 9,
                dropped_records: 1,
                drop_rate: 0.1,
                unmapped_fields: ['ticket_id'],
              },
              auto_fix_suggestions: [
                {
                  message: "Map canonical field 'text' to 'prompt'.",
                  suggested_field_mapping: { text: 'prompt' },
                },
              ],
            },
            preview_rows: [
              { index: 0, raw: { prompt: 'Q1', response: 'A1' }, mapped: { question: 'Q1', answer: 'A1' } },
            ],
          },
        };
      }

      if (url === '/projects/77/adapter-studio/preview') {
        return {
          data: {
            profile: {
              schema: {
                fields: [
                  { path: 'prompt', inferred_type: 'string', null_rate: 0.0, sensitive: false },
                  { path: 'response', inferred_type: 'string', null_rate: 0.0, sensitive: false },
                ],
              },
            },
            drop_analysis: {
              sampled_records: 8,
              mapped_records: 7,
              dropped_records: 1,
              drop_rate: 0.125,
              unmapped_fields: ['ticket_id'],
            },
            preview: {
              resolved_adapter_id: 'qa-pair',
              sampled_records: 8,
              mapped_records: 7,
              dropped_records: 1,
              adapter_contract: {
                task_profiles: ['qa'],
                preferred_training_tasks: ['causal_lm'],
                output_contract: { required_fields: ['text', 'question', 'answer'] },
              },
              conformance_report: {
                required_fields_below_100: ['text'],
              },
              preview_rows: [
                { index: 0, raw: { prompt: 'Q1', response: 'A1' }, mapped: { text: 'Q1', answer: 'A1' } },
              ],
            },
          },
        };
      }

      if (url === '/projects/77/adapter-studio/validate') {
        return {
          data: {
            status: 'warning',
            reason_codes: ['CONTRACT_COVERAGE_GAP'],
            coverage: {
              conformance_report: { required_fields_below_100: ['text'] },
              type_conflicts: [{ message: "Canonical field 'text' expects ['string'] but source 'id' is inferred as 'integer'." }],
            },
          },
        };
      }

      if (url === '/projects/77/adapter-studio/adapters') {
        return { data: { version: 1 } };
      }

      if (url.includes('/export')) {
        return {
          data: {
            written_files: {
              template_json: '/tmp/adapter_template.json',
              plugin_python: '/tmp/adapter_plugin.py',
            },
          },
        };
      }

      return { data: {} };
    });

    apiMock.get.mockResolvedValue({ data: { count: 0, items: [] } });
  });

  it('renders mapping UI from inferred adapter and applies mapping suggestion', async () => {
    const user = userEvent.setup();
    render(<ProjectAdapterStudioPage />);

    await user.click(screen.getByRole('button', { name: /Infer Adapter/i }));

    await waitFor(() => {
      expect(screen.getByDisplayValue('qa-pair')).toBeInTheDocument();
    });
    expect(screen.getByText(/Schema Explorer/i)).toBeInTheDocument();
    expect(screen.getByText("Map canonical field 'text' to 'prompt'.")).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Apply Mapping/i }));
    expect((screen.getByLabelText('text') as HTMLSelectElement).value).toBe('prompt');
  });

  it('shows transformed row preview UI after preview run', async () => {
    const user = userEvent.setup();
    render(<ProjectAdapterStudioPage />);

    await user.click(screen.getByRole('button', { name: /Preview Transform/i }));

    expect(await screen.findByText(/Transformed Row Preview and Error Summary/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Q1/).length).toBeGreaterThan(0);
    expect(screen.getByText(/ticket_id/)).toBeInTheDocument();
  });

  it('shows validation status and reason codes in error summary panel', async () => {
    const user = userEvent.setup();
    render(<ProjectAdapterStudioPage />);

    await user.click(screen.getByRole('button', { name: /Validate Coverage/i }));

    expect(await screen.findByText(/Validation Status:/i)).toBeInTheDocument();
    expect(screen.getByText(/CONTRACT_COVERAGE_GAP/)).toBeInTheDocument();
    expect(screen.getByText(/expects \['string'\]/i)).toBeInTheDocument();
  });
});
