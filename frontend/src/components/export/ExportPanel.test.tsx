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

import ExportPanel from './ExportPanel';

describe('ExportPanel serve run flow', () => {
  beforeEach(() => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/experiments')) {
        return { data: [{ id: 1, name: 'exp-1', status: 'completed' }] };
      }
      if (url.includes('/export/list')) {
        return { data: [] };
      }
      if (url.includes('/registry/models')) {
        return {
          data: {
            models: [
              {
                id: 11,
                experiment_id: 1,
                name: 'registry-model',
                version: 'v1',
                stage: 'candidate',
                deployment_status: 'not_deployed',
                readiness: { metrics: {} },
              },
            ],
          },
        };
      }
      if (url.includes('/export/deployment-targets')) {
        return {
          data: {
            default_target_ids: ['runner.vllm'],
            targets: [
              {
                target_id: 'runner.vllm',
                kind: 'runner',
                display_name: 'vLLM',
                description: 'OpenAI-compatible runtime',
                compatible: true,
                smoke_supported: true,
              },
            ],
          },
        };
      }
      if (url.includes('/export/serve-runs/run-1')) {
        return {
          data: {
            run_id: 'run-1',
            source: 'registry',
            model_id: 11,
            template_id: 'builtin.fastapi',
            status: 'running',
            can_stop: true,
            command: 'python serve.py',
            first_token_curl: 'curl first-token',
            logs_tail: ['running'],
            telemetry: {
              first_healthy_at: null,
              startup_latency_ms: null,
              smoke_passed: null,
              first_token_at: null,
              first_token_latency_ms: null,
              throughput_tokens_per_sec: null,
              first_token_passed: null,
              health_checks: [],
              smoke_checks: [],
              first_token_checks: [],
            },
          },
        };
      }
      return { data: {} };
    });

    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/registry/models/11/serve-plan')) {
        return {
          data: {
            source: 'registry',
            model_id: 11,
            model_name: 'registry-model',
            model_stage: 'candidate',
            host: '127.0.0.1',
            port: 8080,
            run_dir: '/tmp/serve-run',
            templates: [
              {
                template_id: 'builtin.fastapi',
                display_name: 'Built-in FastAPI Server',
                command: 'python serve.py',
                healthcheck: { curl: 'curl health' },
                smoke_test: { curl: 'curl smoke' },
                first_token_probe: { curl: 'curl first-token' },
                notes: [],
              },
            ],
          },
        };
      }
      if (url.includes('/registry/models/11/serve-runs/start')) {
        return {
          data: {
            run_id: 'run-1',
            source: 'registry',
            model_id: 11,
            template_id: 'builtin.fastapi',
            status: 'running',
            can_stop: true,
            command: 'python serve.py',
            first_token_curl: 'curl first-token',
            logs_tail: ['booting...'],
            telemetry: {
              first_healthy_at: null,
              startup_latency_ms: null,
              smoke_passed: null,
              first_token_at: null,
              first_token_latency_ms: null,
              throughput_tokens_per_sec: null,
              first_token_passed: null,
              health_checks: [],
              smoke_checks: [],
              first_token_checks: [],
            },
          },
        };
      }
      if (url.includes('/export/serve-runs/run-1/stop')) {
        return {
          data: {
            run_id: 'run-1',
            source: 'registry',
            model_id: 11,
            template_id: 'builtin.fastapi',
            status: 'stopping',
            can_stop: true,
            command: 'python serve.py',
            first_token_curl: 'curl first-token',
            logs_tail: ['stopping...'],
            telemetry: {
              first_healthy_at: null,
              startup_latency_ms: null,
              smoke_passed: null,
              first_token_at: null,
              first_token_latency_ms: null,
              throughput_tokens_per_sec: null,
              first_token_passed: null,
              health_checks: [],
              smoke_checks: [],
              first_token_checks: [],
            },
          },
        };
      }
      return { data: {} };
    });
  });

  it('starts and stops a local serve run from registry serve plan', async () => {
    const user = userEvent.setup();
    render(<ExportPanel projectId={9} />);

    const servePlanButton = await screen.findByRole('button', { name: 'Serve Plan' });
    await user.click(servePlanButton);

    expect(await screen.findByText('One-Click Serve Plan')).toBeInTheDocument();
    expect(await screen.findByText('First-Token Probe Curl')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Run Now' }));

    expect(await screen.findByText('Live Serve Status')).toBeInTheDocument();
    expect(await screen.findByText('Throughput')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Stop' }));

    await waitFor(() => {
      expect(apiMock.post).toHaveBeenCalledWith('/projects/9/export/serve-runs/run-1/stop');
    });
    expect(await screen.findByText('stopping')).toBeInTheDocument();
  });

  it('shows measured vs estimated provenance in optimization tradeoff cards', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/export/optimize')) {
        return {
          data: {
            project_id: 9,
            target_id: 'runner.vllm',
            candidates: [
              {
                id: 'c-measured',
                name: 'HF + 4bit',
                quantization: '4-bit',
                runtime_template: 'huggingface',
                metrics: {
                  latency_ms: 18.2,
                  memory_gb: 1.2,
                  quality_score: 0.91,
                },
                is_recommended: true,
                reasons: ['Measured probe run completed'],
                metric_source: 'measured',
                metric_sources: {
                  latency_ms: 'measured',
                  memory_gb: 'measured',
                  quality_score: 'measured',
                },
                measurement: {
                  mode: 'measured',
                },
              },
              {
                id: 'c-estimated',
                name: 'GGUF + 4bit',
                quantization: '4-bit',
                runtime_template: 'gguf',
                metrics: {
                  latency_ms: 45,
                  memory_gb: 0.88,
                  quality_score: 0.84,
                },
                is_recommended: false,
                reasons: ['Probe unavailable for GGUF on this host'],
                metric_source: 'estimated',
                metric_sources: {
                  latency_ms: 'estimated',
                  memory_gb: 'measured',
                  quality_score: 'estimated',
                },
                measurement: {
                  mode: 'estimated',
                  fallback_reason: 'Runtime probe unavailable for this format.',
                },
              },
            ],
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ExportPanel projectId={9} />);

    const selectors = await screen.findAllByRole('combobox');
    await user.selectOptions(selectors[0], '1');
    await user.click(screen.getByRole('button', { name: 'Optimize for Target' }));

    expect(await screen.findByText('Optimization Tradeoff Cards')).toBeInTheDocument();
    expect(screen.getByLabelText('Measured metrics')).toBeInTheDocument();
    expect(screen.getByLabelText('Estimated metrics')).toBeInTheDocument();
    expect(screen.getByText('~45.0ms')).toBeInTheDocument();
    expect(screen.getByText(/Runtime probe unavailable for this format\./i)).toBeInTheDocument();
  });
});
