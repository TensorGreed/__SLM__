import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

vi.mock('../../api/client', () => ({
  default: apiMock,
}));

import EvalPanel from './EvalPanel';

describe('EvalPanel remediation planner', () => {
  beforeEach(() => {
    let generated = false;

    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/experiments')) {
        return { data: [{ id: 21, name: 'exp-21' }] };
      }
      if (url.includes('/evaluation/results/21')) {
        return {
          data: [
            {
              id: 501,
              dataset_name: 'gold_test',
              eval_type: 'llm_judge',
              pass_rate: 0.61,
              metrics: {
                scored_predictions: [],
              },
            },
          ],
        };
      }
      if (url.includes('/evaluation/safety-scorecard/21')) {
        return { data: { overall_risk: 'medium', red_flags: [] } };
      }
      if (url.includes('/evaluation/scorecard/21')) {
        return {
          data: {
            experiment_id: 21,
            is_ship: false,
            decision: 'NO-SHIP',
            reasons: [],
            failed_gates: [],
            missing_metrics: [],
            gate_report: {
              passed: true,
              checks: [],
              missing_required_metrics: [],
              failed_gate_ids: [],
            },
          },
        };
      }
      if (url.includes('/evaluation/packs')) {
        return { data: { packs: [] } };
      }
      if (url.includes('/evaluation/pack-preference')) {
        return {
          data: {
            preferred_pack_id: null,
            active_pack_id: null,
            active_pack_source: 'default',
            active_pack: { display_name: 'Auto' },
          },
        };
      }
      if (url.includes('/evaluation/gates/21')) {
        return {
          data: {
            captured_at: '2026-03-26T00:00:00Z',
            passed: true,
            failed_gate_ids: [],
            missing_required_metrics: [],
            checks: [],
          },
        };
      }
      if (url.includes('/evaluation/remediation-plans?') || url.endsWith('/evaluation/remediation-plans')) {
        return {
          data: {
            project_id: 4,
            count: 1,
            plans: [
              {
                plan_id: generated ? 'plan-2' : 'plan-1',
                created_at: '2026-03-26T00:00:00Z',
                experiment_id: 21,
                evaluation_result_id: 501,
                eval_type: 'llm_judge',
                dataset_name: 'gold_test',
                root_causes: generated ? ['coverage_gap'] : ['hallucination'],
                summary: {
                  total_failures_analyzed: generated ? 34 : 26,
                  cluster_count: 2,
                  recommendation_count: 2,
                  dominant_root_cause: generated ? 'coverage_gap' : 'hallucination',
                },
              },
            ],
          },
        };
      }
      if (url.includes('/evaluation/remediation-plans/plan-1')) {
        return {
          data: {
            plan_id: 'plan-1',
            created_at: '2026-03-26T00:00:00Z',
            summary: {
              total_failures_analyzed: 26,
              cluster_count: 2,
              recommendation_count: 2,
              dominant_root_cause: 'hallucination',
            },
            clusters: [
              {
                cluster_id: 'cluster-1',
                root_cause: 'hallucination',
                slice: 'billing',
                failure_count: 14,
                confidence: 0.74,
              },
            ],
            recommendations: [
              {
                recommendation_id: 'rec-1',
                root_cause: 'hallucination',
                title: 'Tighten answer grounding',
                confidence: 0.79,
                data_operations: ['Collect grounded references for billing flows'],
                training_config_changes: ['Increase citation penalty'],
                expected_impact: { metric: 'llm_judge', estimated_delta: 0.07 },
              },
            ],
          },
        };
      }
      if (url.includes('/evaluation/remediation-plans/plan-2')) {
        return {
          data: {
            plan_id: 'plan-2',
            created_at: '2026-03-26T00:00:00Z',
            summary: {
              total_failures_analyzed: 34,
              cluster_count: 3,
              recommendation_count: 2,
              dominant_root_cause: 'coverage_gap',
            },
            clusters: [
              {
                cluster_id: 'cluster-2',
                root_cause: 'coverage_gap',
                slice: 'refund-policy',
                failure_count: 20,
                confidence: 0.81,
              },
            ],
            recommendations: [
              {
                recommendation_id: 'rec-2',
                root_cause: 'coverage_gap',
                title: 'Expand refund scenario coverage',
                confidence: 0.84,
                data_operations: ['Collect refund policy edge cases'],
                training_config_changes: ['Raise max_steps by 15%'],
                expected_impact: { metric: 'llm_judge', estimated_delta: 0.09 },
              },
            ],
          },
        };
      }
      return { data: {} };
    });

    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/evaluation/remediation-plans/generate')) {
        generated = true;
        return {
          data: {
            plan_id: 'plan-2',
            created_at: '2026-03-26T00:00:00Z',
            summary: {
              total_failures_analyzed: 34,
              cluster_count: 3,
              recommendation_count: 2,
              dominant_root_cause: 'coverage_gap',
            },
            clusters: [
              {
                cluster_id: 'cluster-2',
                root_cause: 'coverage_gap',
                slice: 'refund-policy',
                failure_count: 20,
                confidence: 0.81,
              },
            ],
            recommendations: [
              {
                recommendation_id: 'rec-2',
                root_cause: 'coverage_gap',
                title: 'Expand refund scenario coverage',
                confidence: 0.84,
                data_operations: ['Collect refund policy edge cases'],
                training_config_changes: ['Raise max_steps by 15%'],
                expected_impact: { metric: 'llm_judge', estimated_delta: 0.09 },
              },
            ],
          },
        };
      }
      return { data: {} };
    });
  });

  it('generates and renders remediation recommendations with root-cause details', async () => {
    const user = userEvent.setup();
    render(<EvalPanel projectId={4} />);

    await user.click(await screen.findByRole('button', { name: 'exp-21' }));
    expect(await screen.findByText('Closed-Loop Remediation Planner')).toBeInTheDocument();

    expect(await screen.findByText(/Tighten answer grounding/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Generate Remediation Plan' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith(
        '/projects/4/evaluation/remediation-plans/generate',
        expect.objectContaining({
          experiment_id: 21,
          max_failures: 200,
        }),
      );
    });

    expect(await screen.findByText(/Expand refund scenario coverage/i)).toBeInTheDocument();
    expect((await screen.findAllByText(/coverage gap/i)).length).toBeGreaterThan(0);
  });
});
