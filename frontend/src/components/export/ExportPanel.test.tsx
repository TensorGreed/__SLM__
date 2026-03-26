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

  it('renders benchmark matrix recommendations with measured and estimated provenance', async () => {
    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/export/optimize/matrix/start')) {
        return {
          data: {
            run_id: 'matrix-1',
            project_id: 9,
            status: 'completed',
            target_ids: ['runner.vllm'],
            runtime_fingerprint: {
              gpu: 'RTX 4090',
              toolchain: 'torch-2.4',
            },
            summary: {
              target_count: 1,
              candidate_evaluation_count: 2,
              measured_candidate_count: 1,
              mixed_candidate_count: 0,
              estimated_candidate_count: 1,
            },
            targets: [
              {
                target_id: 'runner.vllm',
                target_device: 'server',
                candidate_count: 2,
                measured_candidate_count: 1,
                mixed_candidate_count: 0,
                estimated_candidate_count: 1,
                recommended_candidate_id: 'c-measured',
              },
            ],
          },
        };
      }
      return { data: {} };
    });

    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/experiments')) {
        return { data: [{ id: 1, name: 'exp-1', status: 'completed' }] };
      }
      if (url.includes('/export/list')) {
        return { data: [] };
      }
      if (url.includes('/registry/models')) {
        return { data: { models: [] } };
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
      if (url.includes('/export/optimize/matrix/matrix-1/recommendations')) {
        return {
          data: {
            project_id: 9,
            run_id: 'matrix-1',
            status: 'completed',
            recommendations_by_target: [
              {
                target_id: 'runner.vllm',
                target_device: 'server',
                recommended_candidate_id: 'c-measured',
                recommendations: [
                  {
                    rank: 1,
                    id: 'c-measured',
                    name: 'HF 4-bit',
                    runtime_template: 'huggingface',
                    quantization: '4-bit',
                    metrics: { latency_ms: 14.2, memory_gb: 1.1, quality_score: 0.93 },
                    metric_source: 'measured',
                    metric_sources: {
                      latency_ms: 'measured',
                      memory_gb: 'measured',
                      quality_score: 'measured',
                    },
                    confidence: { score: 0.97 },
                    measurement: { mode: 'measured' },
                    reasons: ['Probe completed'],
                  },
                  {
                    rank: 2,
                    id: 'c-est',
                    name: 'GGUF q4',
                    runtime_template: 'gguf',
                    quantization: '4-bit',
                    metrics: { latency_ms: 26.1, memory_gb: 0.9, quality_score: 0.87 },
                    metric_source: 'estimated',
                    metric_sources: {
                      latency_ms: 'estimated',
                      memory_gb: 'estimated',
                      quality_score: 'estimated',
                    },
                    confidence: { score: 0.63 },
                    measurement: {
                      mode: 'estimated',
                      fallback_reason: 'Probe runtime unavailable.',
                      remediation_hint: 'Install runtime probes and rerun matrix.',
                    },
                    reasons: ['Fallback estimate'],
                  },
                ],
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
    await user.click(screen.getByRole('button', { name: 'Run Benchmark Matrix' }));

    expect(await screen.findByText('Benchmark Matrix Run')).toBeInTheDocument();
    expect(await screen.findByText('TOP RANK')).toBeInTheDocument();
    expect(await screen.findByText(/Probe runtime unavailable\./i)).toBeInTheDocument();
    expect(await screen.findByText(/Remediation: Install runtime probes and rerun matrix\./i)).toBeInTheDocument();
  });

  it('shows mobile sdk smoke validation details in deploy plan', async () => {
    apiMock.get.mockImplementation(async (url: string) => {
      if (url.includes('/training/experiments')) {
        return { data: [{ id: 1, name: 'exp-1', status: 'completed' }] };
      }
      if (url.includes('/export/list')) {
        return {
          data: [
            {
              id: 1,
              export_format: 'gguf',
              quantization: '4-bit',
              status: 'completed',
              output_path: '/tmp/export-1',
              file_size_bytes: 1024,
              created_at: '2026-03-26T00:00:00Z',
            },
          ],
        };
      }
      if (url.includes('/registry/models')) {
        return { data: { models: [] } };
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
      return { data: {} };
    });

    apiMock.post.mockImplementation(async (url: string) => {
      if (url.includes('/export/1/deploy-as-api')) {
        return {
          data: {
            target_id: 'sdk.apple_coreml_stub',
            target_kind: 'sdk',
            display_name: 'Apple CoreML Reference App',
            summary: 'Mobile SDK runnable reference bundle generated.',
            steps: ['step 1', 'step 2'],
            sdk_artifact: {
              zip_path: '/tmp/mobile/reference_app.zip',
              readme_path: '/tmp/mobile/README.md',
              entrypoint_path: '/tmp/mobile/ios_app/main.swift',
              runtime_path: '/tmp/mobile/ios_app/runtime_config.json',
              model_placement_paths: ['ios_app/Models/model.mlmodelc'],
              run_commands: ['xcodebuild -scheme ReferenceApp'],
              smoke_commands: ['./scripts/smoke.sh'],
              smoke_validation: {
                smoke_passed: true,
                warnings: ['Use release build for perf.'],
                errors: [],
                checks: [
                  {
                    check_id: 'entrypoint_exists',
                    status: 'pass',
                    message: 'Entrypoint found',
                  },
                ],
              },
            },
          },
        };
      }
      return { data: {} };
    });

    const user = userEvent.setup();
    render(<ExportPanel projectId={9} />);

    const selectors = await screen.findAllByRole('combobox');
    await user.selectOptions(selectors[3], 'sdk.apple_coreml_stub');

    await user.click(await screen.findByText('GGUF'));
    await user.click(screen.getByRole('button', { name: 'Deploy / SDK Plan' }));

    expect(await screen.findByText('Smoke Validation')).toBeInTheDocument();
    expect(await screen.findByText('PASS')).toBeInTheDocument();
    expect(await screen.findByText(/ios_app\/Models\/model\.mlmodelc/i)).toBeInTheDocument();
    expect(await screen.findByText(/xcodebuild -scheme ReferenceApp/i)).toBeInTheDocument();
  });
});
