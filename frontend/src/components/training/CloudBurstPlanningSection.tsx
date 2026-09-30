/**
 * Cloud Burst planning (Power tools): quote a remote GPU lease, build a
 * launch plan, submit / poll / cancel a managed job and sync its artifacts.
 * Owns all of its state; mount with ``key={projectId}`` to reset per project.
 */

import { useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import { loadWorkflowStagePrefill } from '../../utils/workflowGraphPrefill';
import type {
  CloudBurstCatalogResponse,
  CloudBurstLaunchPlanResponse,
  CloudBurstMetricPoint,
  CloudBurstQuoteResponse,
  CloudBurstRunListResponse,
  CloudBurstRunStatusResponse,
} from './trainingPanelTypes';
import { asRecord } from './trainingPanelUtils';

interface CloudBurstPlanningSectionProps {
  projectId: number;
}

export default function CloudBurstPlanningSection({ projectId }: CloudBurstPlanningSectionProps) {
  const [cloudBurstCatalog, setCloudBurstCatalog] = useState<CloudBurstCatalogResponse | null>(null);
  const [cloudBurstProviderId, setCloudBurstProviderId] = useState('');
  const [cloudBurstGpuSku, setCloudBurstGpuSku] = useState('');
  const [cloudBurstDurationHours, setCloudBurstDurationHours] = useState('2');
  const [cloudBurstStorageGb, setCloudBurstStorageGb] = useState('50');
  const [cloudBurstEgressGb, setCloudBurstEgressGb] = useState('0');
  const [cloudBurstSpot, setCloudBurstSpot] = useState(true);
  const [cloudBurstRegion, setCloudBurstRegion] = useState('');
  const [cloudBurstImage, setCloudBurstImage] = useState('');
  const [cloudBurstStartupScript, setCloudBurstStartupScript] = useState('');
  const [cloudBurstExperimentId, setCloudBurstExperimentId] = useState('');
  const [cloudBurstExecutionMode, setCloudBurstExecutionMode] = useState('auto');
  const [cloudBurstAllowFallbackToSimulation, setCloudBurstAllowFallbackToSimulation] = useState(true);
  const [cloudBurstIdempotencyKey, setCloudBurstIdempotencyKey] = useState('');
  const [cloudBurstSyncCursor, setCloudBurstSyncCursor] = useState('');
  const [cloudBurstLoadingCatalog, setCloudBurstLoadingCatalog] = useState(false);
  const [cloudBurstLoadingQuote, setCloudBurstLoadingQuote] = useState(false);
  const [cloudBurstLoadingPlan, setCloudBurstLoadingPlan] = useState(false);
  const [cloudBurstError, setCloudBurstError] = useState('');
  const [cloudBurstInfo, setCloudBurstInfo] = useState('');
  const [cloudBurstQuote, setCloudBurstQuote] = useState<CloudBurstQuoteResponse | null>(null);
  const [cloudBurstPlan, setCloudBurstPlan] = useState<CloudBurstLaunchPlanResponse | null>(null);
  const [cloudBurstRuns, setCloudBurstRuns] = useState<CloudBurstRunStatusResponse[]>([]);
  const [cloudBurstActiveRunId, setCloudBurstActiveRunId] = useState('');
  const [cloudBurstActiveRun, setCloudBurstActiveRun] = useState<CloudBurstRunStatusResponse | null>(null);
  const [cloudBurstLoadingRuns, setCloudBurstLoadingRuns] = useState(false);
  const [cloudBurstSubmittingJob, setCloudBurstSubmittingJob] = useState(false);
  const [cloudBurstCancellingJob, setCloudBurstCancellingJob] = useState(false);
  const [cloudBurstSyncingArtifacts, setCloudBurstSyncingArtifacts] = useState(false);
  const [cloudBurstPrefillStage, setCloudBurstPrefillStage] = useState('');

  const cloudProviders = Array.isArray(cloudBurstCatalog?.providers) ? cloudBurstCatalog.providers : [];
  const cloudGpuSkus = Array.isArray(cloudBurstCatalog?.gpu_skus) ? cloudBurstCatalog.gpu_skus : [];
  const selectedCloudProvider = cloudProviders.find((item) => item.provider_id === cloudBurstProviderId) || null;

  const cloudBurstActiveStatus = String(cloudBurstActiveRun?.status || '').trim().toLowerCase();
  const cloudBurstActiveIsTerminal = ['completed', 'failed', 'cancelled'].includes(cloudBurstActiveStatus);
  const cloudBurstMetrics = useMemo(() => {
    const rows = Array.isArray(cloudBurstActiveRun?.metrics_tail)
      ? cloudBurstActiveRun.metrics_tail
      : [];
    const parsed = rows.map((row) => {
      const item = asRecord(row);
      const step = Number(item.step);
      const epoch = Number(item.epoch);
      const trainLoss = Number(item.train_loss);
      const evalLoss = Number(item.eval_loss);
      const learningRate = Number(item.learning_rate);
      const throughputTps = Number(item.throughput_tps);
      return {
        step: Number.isFinite(step) ? step : undefined,
        epoch: Number.isFinite(epoch) ? epoch : undefined,
        train_loss: Number.isFinite(trainLoss) ? trainLoss : null,
        eval_loss: Number.isFinite(evalLoss) ? evalLoss : null,
        learning_rate: Number.isFinite(learningRate) ? learningRate : null,
        throughput_tps: Number.isFinite(throughputTps) ? throughputTps : null,
        at: typeof item.at === 'string' ? item.at : null,
      } as CloudBurstMetricPoint;
    });
    return parsed;
  }, [cloudBurstActiveRun?.metrics_tail]);
  const cloudBurstLatestMetric = cloudBurstMetrics.length > 0
    ? cloudBurstMetrics[cloudBurstMetrics.length - 1]
    : null;
  const cloudBurstMetricSeries = useMemo(() => {
    const buildSeries = (field: 'train_loss' | 'eval_loss') => {
      const points = cloudBurstMetrics
        .map((item, idx) => ({ idx, value: Number(item[field]) }))
        .filter((item) => Number.isFinite(item.value));
      if (points.length < 2) {
        return '';
      }
      const values = points.map((item) => item.value);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min;
      return points
        .map((item, localIndex) => {
          const x = points.length <= 1 ? 0 : (localIndex / (points.length - 1)) * 100;
          const normalized = span <= 0 ? 0.5 : (item.value - min) / span;
          const y = 100 - (normalized * 100);
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(' ');
    };
    return {
      train: buildSeries('train_loss'),
      eval: buildSeries('eval_loss'),
    };
  }, [cloudBurstMetrics]);

  const loadCloudBurstCatalog = async () => {
    setCloudBurstLoadingCatalog(true);
    try {
      const res = await api.get<CloudBurstCatalogResponse>(`/projects/${projectId}/training/cloud-burst/catalog`);
      const payload = res.data || null;
      setCloudBurstCatalog(payload);
      const providers = Array.isArray(payload?.providers) ? payload.providers : [];
      const gpuSkus = Array.isArray(payload?.gpu_skus) ? payload.gpu_skus : [];
      const providerSelected = Boolean(
        cloudBurstProviderId && providers.some((item) => item.provider_id === cloudBurstProviderId),
      );
      const gpuSelected = Boolean(
        cloudBurstGpuSku && gpuSkus.some((item) => item.gpu_sku === cloudBurstGpuSku),
      );
      if (!providerSelected && providers.length > 0) {
        setCloudBurstProviderId(providers[0].provider_id);
      }
      if (!gpuSelected && gpuSkus.length > 0) {
        setCloudBurstGpuSku(gpuSkus[0].gpu_sku);
      }
      setCloudBurstError('');
    } catch (err: any) {
      setCloudBurstCatalog(null);
      setCloudBurstError(err?.response?.data?.detail || 'Failed to load cloud burst catalog');
    } finally {
      setCloudBurstLoadingCatalog(false);
    }
  };

  const requestCloudBurstQuote = async () => {
    if (!cloudBurstProviderId || !cloudBurstGpuSku) {
      setCloudBurstError('Select provider and GPU SKU before requesting quote.');
      return;
    }
    setCloudBurstLoadingQuote(true);
    setCloudBurstError('');
    try {
      const durationValue = Number.parseFloat(cloudBurstDurationHours);
      const storageValue = Number.parseInt(cloudBurstStorageGb, 10);
      const egressValue = Number.parseFloat(cloudBurstEgressGb);
      const res = await api.post<CloudBurstQuoteResponse>(
        `/projects/${projectId}/training/cloud-burst/quote`,
        {
          provider_id: cloudBurstProviderId,
          gpu_sku: cloudBurstGpuSku,
          duration_hours: Number.isFinite(durationValue) ? durationValue : 2.0,
          storage_gb: Number.isFinite(storageValue) ? storageValue : 50,
          egress_gb: Number.isFinite(egressValue) ? egressValue : 0.0,
          spot: cloudBurstSpot,
        },
      );
      setCloudBurstQuote(res.data || null);
    } catch (err: any) {
      setCloudBurstQuote(null);
      setCloudBurstError(err?.response?.data?.detail || 'Failed to estimate cloud burst quote');
    } finally {
      setCloudBurstLoadingQuote(false);
    }
  };

  const requestCloudBurstPlan = async () => {
    if (!cloudBurstProviderId || !cloudBurstGpuSku) {
      setCloudBurstError('Select provider and GPU SKU before building launch plan.');
      return;
    }
    setCloudBurstLoadingPlan(true);
    setCloudBurstError('');
    try {
      const durationValue = Number.parseFloat(cloudBurstDurationHours);
      const parsedExperimentId = Number.parseInt(cloudBurstExperimentId, 10);
      const res = await api.post<CloudBurstLaunchPlanResponse>(
        `/projects/${projectId}/training/cloud-burst/launch-plan`,
        {
          provider_id: cloudBurstProviderId,
          gpu_sku: cloudBurstGpuSku,
          duration_hours: Number.isFinite(durationValue) ? durationValue : 2.0,
          experiment_id: Number.isFinite(parsedExperimentId) && parsedExperimentId > 0
            ? parsedExperimentId
            : undefined,
          region: cloudBurstRegion.trim() || undefined,
          image: cloudBurstImage.trim(),
          startup_script: cloudBurstStartupScript.trim(),
          spot: cloudBurstSpot,
        },
      );
      setCloudBurstPlan(res.data || null);
    } catch (err: any) {
      setCloudBurstPlan(null);
      setCloudBurstError(err?.response?.data?.detail || 'Failed to build cloud burst launch plan');
    } finally {
      setCloudBurstLoadingPlan(false);
    }
  };

  const loadCloudBurstJobs = async (options?: { silent?: boolean }) => {
    if (!options?.silent) {
      setCloudBurstLoadingRuns(true);
    }
    try {
      const res = await api.get<CloudBurstRunListResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs?limit=12`,
      );
      const rows = Array.isArray(res.data?.runs) ? res.data.runs : [];
      setCloudBurstRuns(rows);
      if (!cloudBurstActiveRunId && rows.length > 0) {
        const firstRunId = String(rows[0]?.run_id || '').trim();
        setCloudBurstActiveRunId(firstRunId);
      }
      setCloudBurstError('');
    } catch (err: any) {
      if (!options?.silent) {
        setCloudBurstError(err?.response?.data?.detail || 'Failed to load cloud burst jobs');
      }
    } finally {
      if (!options?.silent) {
        setCloudBurstLoadingRuns(false);
      }
    }
  };

  const loadCloudBurstJobStatus = async (
    runId: string,
    options?: { silent?: boolean; logsTail?: number },
  ) => {
    const trimmedRunId = String(runId || '').trim();
    if (!trimmedRunId) {
      setCloudBurstActiveRun(null);
      setCloudBurstSyncCursor('');
      return;
    }
    if (!options?.silent) {
      setCloudBurstLoadingRuns(true);
    }
    try {
      const logsTail = Math.max(20, Math.min(1000, Number(options?.logsTail || 200)));
      const res = await api.get<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/${trimmedRunId}?logs_tail=${logsTail}`,
      );
      const run = res.data || null;
      setCloudBurstActiveRun(run);
      setCloudBurstActiveRunId(trimmedRunId);
      const nextCursor = String(run?.artifacts?.last_sync_summary?.next_cursor || '').trim();
      setCloudBurstSyncCursor(nextCursor);
      setCloudBurstError('');
    } catch (err: any) {
      if (!options?.silent) {
        setCloudBurstError(err?.response?.data?.detail || 'Failed to load cloud burst job status');
      }
    } finally {
      if (!options?.silent) {
        setCloudBurstLoadingRuns(false);
      }
    }
  };

  const submitCloudBurstManagedJob = async () => {
    if (!cloudBurstProviderId || !cloudBurstGpuSku) {
      setCloudBurstError('Select provider and GPU SKU before submitting a managed job.');
      return;
    }
    setCloudBurstSubmittingJob(true);
    setCloudBurstError('');
    setCloudBurstInfo('');
    try {
      const durationValue = Number.parseFloat(cloudBurstDurationHours);
      const parsedExperimentId = Number.parseInt(cloudBurstExperimentId, 10);
      const res = await api.post<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/submit`,
        {
          provider_id: cloudBurstProviderId,
          gpu_sku: cloudBurstGpuSku,
          duration_hours: Number.isFinite(durationValue) ? durationValue : 2.0,
          experiment_id: Number.isFinite(parsedExperimentId) && parsedExperimentId > 0
            ? parsedExperimentId
            : undefined,
          region: cloudBurstRegion.trim() || undefined,
          image: cloudBurstImage.trim(),
          startup_script: cloudBurstStartupScript.trim(),
          spot: cloudBurstSpot,
          auto_artifact_sync: true,
          artifact_sync_policy: 'smart',
          execution_mode: cloudBurstExecutionMode,
          allow_fallback_to_simulation: cloudBurstAllowFallbackToSimulation,
          idempotency_key: cloudBurstIdempotencyKey.trim() || undefined,
        },
      );
      const run = res.data || null;
      setCloudBurstActiveRun(run);
      setCloudBurstActiveRunId(String(run?.run_id || '').trim());
      setCloudBurstSyncCursor('');
      if (run?.idempotent_replay) {
        setCloudBurstInfo('Idempotency replay returned an existing managed run.');
      } else if (
        String(run?.execution_mode_requested || '') !== String(run?.execution_mode_effective || '')
        && String(run?.execution_mode_fallback_reason || '').trim()
      ) {
        setCloudBurstInfo(String(run?.execution_mode_fallback_reason || '').trim());
      }
      await loadCloudBurstJobs({ silent: true });
    } catch (err: any) {
      setCloudBurstError(err?.response?.data?.detail || 'Failed to submit managed cloud burst job');
    } finally {
      setCloudBurstSubmittingJob(false);
    }
  };

  const cancelCloudBurstManagedJob = async (runId: string) => {
    const trimmedRunId = String(runId || '').trim();
    if (!trimmedRunId) {
      return;
    }
    setCloudBurstCancellingJob(true);
    setCloudBurstError('');
    setCloudBurstInfo('');
    try {
      const res = await api.post<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/${trimmedRunId}/cancel`,
      );
      setCloudBurstActiveRun(res.data || null);
      await loadCloudBurstJobs({ silent: true });
    } catch (err: any) {
      setCloudBurstError(err?.response?.data?.detail || 'Failed to cancel cloud burst job');
    } finally {
      setCloudBurstCancellingJob(false);
    }
  };

  const syncCloudBurstManagedArtifacts = async (runId: string) => {
    const trimmedRunId = String(runId || '').trim();
    if (!trimmedRunId) {
      return;
    }
    setCloudBurstSyncingArtifacts(true);
    setCloudBurstError('');
    setCloudBurstInfo('');
    try {
      const res = await api.post<CloudBurstRunStatusResponse>(
        `/projects/${projectId}/training/cloud-burst/jobs/${trimmedRunId}/sync-artifacts`,
        {
          policy: 'smart',
          dry_run: false,
          max_files: 2000,
          cursor: cloudBurstSyncCursor.trim() || undefined,
        },
      );
      setCloudBurstActiveRun(res.data || null);
      const syncSummary = asRecord(asRecord(res.data).sync);
      const nextCursor = String(syncSummary.next_cursor || '').trim();
      setCloudBurstSyncCursor(nextCursor);
      if (nextCursor) {
        setCloudBurstInfo(
          `Partial sync complete. Continue syncing with next cursor (${nextCursor.slice(0, 24)}...).`,
        );
      }
      await loadCloudBurstJobs({ silent: true });
    } catch (err: any) {
      setCloudBurstError(err?.response?.data?.detail || 'Failed to sync cloud burst artifacts');
    } finally {
      setCloudBurstSyncingArtifacts(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const applyCloudBurstPrefill = async () => {
      const prefill = await loadWorkflowStagePrefill(projectId, ['cloud_burst']);
      if (cancelled || !prefill) {
        return;
      }
      const cfg = prefill.config || {};
      const providerToken = String(cfg.provider_id || '').trim();
      if (providerToken) setCloudBurstProviderId(providerToken);
      const skuToken = String(cfg.gpu_sku || '').trim();
      if (skuToken) setCloudBurstGpuSku(skuToken);

      const durationValue = Number(cfg.duration_hours);
      if (Number.isFinite(durationValue) && durationValue > 0) {
        setCloudBurstDurationHours(String(durationValue));
      }
      const storageValue = Number(cfg.storage_gb);
      if (Number.isFinite(storageValue) && storageValue > 0) {
        setCloudBurstStorageGb(String(Math.round(storageValue)));
      }
      const egressValue = Number(cfg.egress_gb);
      if (Number.isFinite(egressValue) && egressValue >= 0) {
        setCloudBurstEgressGb(String(egressValue));
      }
      if (typeof cfg.spot === 'boolean') {
        setCloudBurstSpot(cfg.spot);
      }
      const regionToken = String(cfg.region || '').trim();
      if (regionToken) setCloudBurstRegion(regionToken);
      const imageToken = String(cfg.image || '').trim();
      if (imageToken) setCloudBurstImage(imageToken);
      const startupToken = String(cfg.startup_script || '').trim();
      if (startupToken) setCloudBurstStartupScript(startupToken);
      const experimentValue = Number(cfg.experiment_id);
      if (Number.isFinite(experimentValue) && experimentValue > 0) {
        setCloudBurstExperimentId(String(Math.round(experimentValue)));
      }
      const executionModeToken = String(cfg.execution_mode || '').trim().toLowerCase();
      if (executionModeToken === 'auto' || executionModeToken === 'live' || executionModeToken === 'simulate') {
        setCloudBurstExecutionMode(executionModeToken);
      }
      if (typeof cfg.allow_fallback_to_simulation === 'boolean') {
        setCloudBurstAllowFallbackToSimulation(cfg.allow_fallback_to_simulation);
      }
      const idempotencyToken = String(cfg.idempotency_key || '').trim();
      if (idempotencyToken) {
        setCloudBurstIdempotencyKey(idempotencyToken);
      }
      setCloudBurstPrefillStage(prefill.stage);
    };
    void applyCloudBurstPrefill();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    const runId = String(cloudBurstActiveRunId || '').trim();
    if (!runId) {
      return;
    }
    void loadCloudBurstJobStatus(runId, { silent: true, logsTail: 240 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudBurstActiveRunId, projectId]);

  useEffect(() => {
    const runId = String(cloudBurstActiveRunId || '').trim();
    if (!runId || cloudBurstActiveIsTerminal || !cloudBurstActiveStatus) {
      return;
    }
    const interval = window.setInterval(() => {
      void loadCloudBurstJobStatus(runId, { silent: true, logsTail: 240 });
      void loadCloudBurstJobs({ silent: true });
    }, 4000);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloudBurstActiveRunId, cloudBurstActiveStatus, cloudBurstActiveIsTerminal, projectId]);

  useEffect(() => {
    void loadCloudBurstCatalog();
    void loadCloudBurstJobs({ silent: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  return (
    <details className="training-collapsible">
      <summary>
        <span>Cloud Burst Planning</span>
        <small>Estimate remote GPU lease cost and generate one-click launch plans</small>
      </summary>
      <div className="training-collapsible__content">
        {cloudBurstPrefillStage && (
          <div className="form-hint">
            Prefilled from workflow template stage: {cloudBurstPrefillStage}
          </div>
        )}
        <div className="form-group form-group--spaced">
          <div className="form-inline-actions">
            <select
              className="input training-recipe-select"
              value={cloudBurstProviderId}
              onChange={(e) => setCloudBurstProviderId(e.target.value)}
            >
              <option value="">Select provider</option>
              {cloudProviders.map((provider) => (
                <option key={provider.provider_id} value={provider.provider_id}>
                  {provider.display_name || provider.provider_id}
                </option>
              ))}
            </select>
            <select
              className="input training-recipe-select"
              value={cloudBurstGpuSku}
              onChange={(e) => setCloudBurstGpuSku(e.target.value)}
            >
              <option value="">Select GPU SKU</option>
              {cloudGpuSkus.map((sku) => (
                <option key={sku.gpu_sku} value={sku.gpu_sku}>
                  {sku.display_name || sku.gpu_sku}
                </option>
              ))}
            </select>
            <input
              className="input"
              type="number"
              min={0.25}
              max={72}
              step={0.25}
              value={cloudBurstDurationHours}
              onChange={(e) => setCloudBurstDurationHours(e.target.value)}
              placeholder="hours"
            />
            <label className="form-label form-label-inline">
              <input
                type="checkbox"
                checked={cloudBurstSpot}
                onChange={(e) => setCloudBurstSpot(e.target.checked)}
              />
              Spot
            </label>
            <button
              className="btn btn-secondary"
              onClick={() => void loadCloudBurstCatalog()}
              disabled={cloudBurstLoadingCatalog}
            >
              {cloudBurstLoadingCatalog ? 'Refreshing...' : 'Refresh Catalog'}
            </button>
          </div>
        </div>
        {selectedCloudProvider && (
          <div className="form-hint">
            Provider capabilities:
            {' '}
            live execution {selectedCloudProvider.supports_live_execution ? 'yes' : 'no'}
            {' • '}
            managed cancel {selectedCloudProvider.supports_managed_cancel ? 'yes' : 'no'}
            {' • '}
            live logs {selectedCloudProvider.supports_live_logs ? 'yes' : 'no'}
          </div>
        )}
        <div className="training-grid-2">
          <div className="form-group">
            <label className="form-label">Execution Mode</label>
            <select
              className="input"
              value={cloudBurstExecutionMode}
              onChange={(e) => setCloudBurstExecutionMode(e.target.value)}
            >
              <option value="auto">Auto (prefer live when available)</option>
              <option value="live">Live provider job</option>
              <option value="simulate">Simulated managed run</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Idempotency Key (optional)</label>
            <input
              className="input"
              value={cloudBurstIdempotencyKey}
              onChange={(e) => setCloudBurstIdempotencyKey(e.target.value)}
              placeholder="same key => same run response"
            />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label form-label-inline">
            <input
              type="checkbox"
              checked={cloudBurstAllowFallbackToSimulation}
              onChange={(e) => setCloudBurstAllowFallbackToSimulation(e.target.checked)}
            />
            Allow fallback to simulation when live submit is unavailable
          </label>
        </div>
        <div className="training-grid-2">
          <div className="form-group">
            <label className="form-label">Storage (GB)</label>
            <input
              className="input"
              type="number"
              min={10}
              max={2000}
              value={cloudBurstStorageGb}
              onChange={(e) => setCloudBurstStorageGb(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Egress (GB)</label>
            <input
              className="input"
              type="number"
              min={0}
              max={5000}
              step={0.5}
              value={cloudBurstEgressGb}
              onChange={(e) => setCloudBurstEgressGb(e.target.value)}
            />
          </div>
        </div>
        <div className="training-grid-2">
          <div className="form-group">
            <label className="form-label">Region (optional)</label>
            <input
              className="input"
              value={cloudBurstRegion}
              onChange={(e) => setCloudBurstRegion(e.target.value)}
              placeholder={
                Array.isArray(selectedCloudProvider?.regions) && selectedCloudProvider.regions.length > 0
                  ? selectedCloudProvider.regions.join(', ')
                  : 'provider default'
              }
            />
          </div>
          <div className="form-group">
            <label className="form-label">Experiment ID (optional)</label>
            <input
              className="input"
              type="number"
              min={1}
              value={cloudBurstExperimentId}
              onChange={(e) => setCloudBurstExperimentId(e.target.value)}
              placeholder="latest or manual id"
            />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Image (optional override)</label>
          <input
            className="input"
            value={cloudBurstImage}
            onChange={(e) => setCloudBurstImage(e.target.value)}
            placeholder="ghcr.io/slm/platform-trainer:latest"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Startup Script (optional override)</label>
          <textarea
            className="input"
            value={cloudBurstStartupScript}
            onChange={(e) => setCloudBurstStartupScript(e.target.value)}
            rows={3}
            placeholder="bash /workspace/entrypoint.sh"
          />
        </div>
        <div className="form-inline-actions">
          <button
            className="btn btn-secondary"
            onClick={() => void requestCloudBurstQuote()}
            disabled={cloudBurstLoadingQuote}
          >
            {cloudBurstLoadingQuote ? 'Estimating...' : 'Estimate Quote'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => void requestCloudBurstPlan()}
            disabled={cloudBurstLoadingPlan}
          >
            {cloudBurstLoadingPlan ? 'Planning...' : 'Build Launch Plan'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => void submitCloudBurstManagedJob()}
            disabled={cloudBurstSubmittingJob}
          >
            {cloudBurstSubmittingJob ? 'Submitting...' : 'Submit Managed Job'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => void loadCloudBurstJobs()}
            disabled={cloudBurstLoadingRuns}
          >
            {cloudBurstLoadingRuns ? 'Refreshing...' : 'Refresh Jobs'}
          </button>
        </div>
        {cloudBurstError && (
          <div className="training-alert training-alert--error">
            {cloudBurstError}
          </div>
        )}
        {cloudBurstInfo && (
          <div className="training-alert training-alert--warning">
            {cloudBurstInfo}
          </div>
        )}
        {cloudBurstQuote && (
          <div className="resolved-defaults-panel">
            <div className="resolved-defaults-panel__title">Cloud Burst Quote</div>
            <div className="resolved-defaults-panel__kv">
              <span>Provider</span>
              <strong>{cloudBurstQuote.provider_name || cloudBurstQuote.provider_id || '-'}</strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>GPU</span>
              <strong>{cloudBurstQuote.gpu_sku || '-'}</strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Total Cost (USD)</span>
              <strong>
                {Number.isFinite(Number(cloudBurstQuote.cost_breakdown_usd?.total))
                  ? `$${Number(cloudBurstQuote.cost_breakdown_usd?.total).toFixed(2)}`
                  : 'n/a'}
              </strong>
            </div>
            {Array.isArray(cloudBurstQuote.warnings) && cloudBurstQuote.warnings.length > 0 && (
              <div className="training-alert training-alert--warning training-alert--tight">
                {cloudBurstQuote.warnings.join(' | ')}
              </div>
            )}
          </div>
        )}
        {cloudBurstPlan && (
          <div className="resolved-defaults-panel">
            <div className="resolved-defaults-panel__title">Cloud Burst Launch Plan</div>
            <div className="resolved-defaults-panel__kv">
              <span>Launch ID</span>
              <strong>{cloudBurstPlan.launch_id || '-'}</strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Credentials Ready</span>
              <strong>{cloudBurstPlan.credentials?.ready ? 'yes' : 'no'}</strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Missing Credentials</span>
              <strong>
                {Array.isArray(cloudBurstPlan.credentials?.missing_keys) && cloudBurstPlan.credentials?.missing_keys.length > 0
                  ? cloudBurstPlan.credentials?.missing_keys.join(', ')
                  : 'none'}
              </strong>
            </div>
            <div className="resolved-defaults-panel__grid">
              <div>
                <div className="resolved-defaults-panel__subtitle">Request Template</div>
                <pre className="resolved-defaults-panel__json">
                  {JSON.stringify(cloudBurstPlan.request_template || {}, null, 2)}
                </pre>
              </div>
              <div>
                <div className="resolved-defaults-panel__subtitle">Full Plan</div>
                <pre className="resolved-defaults-panel__json">
                  {JSON.stringify(cloudBurstPlan || {}, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
        {cloudBurstRuns.length > 0 && (
          <div className="resolved-defaults-panel">
            <div className="resolved-defaults-panel__title">Managed Cloud Burst Jobs</div>
            {cloudBurstRuns.slice(0, 8).map((run, idx) => {
              const runId = String(run.run_id || '').trim();
              const runStatus = String(run.status || 'unknown').trim().toLowerCase();
              const selected = runId && runId === cloudBurstActiveRunId;
              return (
                <div key={`cloud-burst-run-${runId || idx}`} className="resolved-defaults-panel__kv">
                  <span>
                    {runId || 'unknown'} • {String(run.provider_id || '-')} • {String(run.gpu_sku || '-')}
                    {run.execution_mode_effective ? ` • ${run.execution_mode_effective}` : ''}
                    {run.experiment_id ? ` • exp ${run.experiment_id}` : ''}
                  </span>
                  <strong>
                    {runStatus}
                    {' '}
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => void loadCloudBurstJobStatus(runId)}
                      disabled={!runId}
                    >
                      {selected ? 'Viewing' : 'Inspect'}
                    </button>
                  </strong>
                </div>
              );
            })}
          </div>
        )}
        {cloudBurstActiveRun && (
          <div className="resolved-defaults-panel">
            <div className="resolved-defaults-panel__title">Managed Job Status</div>
            <div className="resolved-defaults-panel__kv">
              <span>Run ID</span>
              <strong>{cloudBurstActiveRun.run_id || '-'}</strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Execution</span>
              <strong>
                {String(cloudBurstActiveRun.execution_mode_effective || 'unknown')}
                {cloudBurstActiveRun.execution_mode_requested
                  ? ` (requested ${cloudBurstActiveRun.execution_mode_requested})`
                  : ''}
              </strong>
            </div>
            {cloudBurstActiveRun.execution_mode_fallback_reason && (
              <div className="training-alert training-alert--warning training-alert--tight">
                {cloudBurstActiveRun.execution_mode_fallback_reason}
              </div>
            )}
            <div className="resolved-defaults-panel__kv">
              <span>Status</span>
              <strong>
                {String(cloudBurstActiveRun.status || 'unknown')}
                {cloudBurstActiveRun.cancel_requested ? ' (cancel requested)' : ''}
              </strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Status Reason</span>
              <strong>{String(cloudBurstActiveRun.status_reason || '-')}</strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Provider Job</span>
              <strong>
                {String(cloudBurstActiveRun.provider_id || '-')}
                {cloudBurstActiveRun.provider_job_id ? ` • ${cloudBurstActiveRun.provider_job_id}` : ''}
              </strong>
            </div>
            <div className="resolved-defaults-panel__kv">
              <span>Provider Status</span>
              <strong>{String(cloudBurstActiveRun.provider_status_raw || '-')}</strong>
            </div>
            {cloudBurstActiveRun.current_run_cost !== undefined && (
              <div className="resolved-defaults-panel__kv">
                <span>Current Run Cost</span>
                <strong className="text-success">
                  ${Number(cloudBurstActiveRun.current_run_cost).toFixed(4)}
                </strong>
              </div>
            )}

            {cloudBurstActiveRun.status_timeline && cloudBurstActiveRun.status_timeline.length > 0 && (
              <div className="cloud-burst-timeline">
                <div className="cloud-burst-timeline__title">Run Timeline</div>
                <div className="cloud-burst-timeline__track">
                  {cloudBurstActiveRun.status_timeline.map((event, idx) => (
                    <div key={`timeline-${idx}`} className="cloud-burst-timeline__event">
                      <div className="cloud-burst-timeline__dot" data-status={event.status}></div>
                      <div className="cloud-burst-timeline__content">
                        <div className="cloud-burst-timeline__status">{event.status}</div>
                        <div className="cloud-burst-timeline__time">
                          {new Date(event.at).toLocaleTimeString()}
                        </div>
                        <div className="cloud-burst-timeline__reason">{event.reason}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!!cloudBurstActiveRun.idempotent_replay && (
              <div className="training-alert training-alert--warning training-alert--tight">
                This run was returned via idempotency replay.
              </div>
            )}
            <div className="form-inline-actions">
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => void loadCloudBurstJobStatus(String(cloudBurstActiveRun.run_id || ''))}
                disabled={cloudBurstLoadingRuns}
              >
                {cloudBurstLoadingRuns ? 'Refreshing...' : 'Refresh Status'}
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => void syncCloudBurstManagedArtifacts(String(cloudBurstActiveRun.run_id || ''))}
                disabled={cloudBurstSyncingArtifacts}
              >
                {cloudBurstSyncingArtifacts
                  ? 'Syncing...'
                  : cloudBurstSyncCursor
                    ? 'Sync Next Batch'
                    : 'Sync Artifacts'}
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => void cancelCloudBurstManagedJob(String(cloudBurstActiveRun.run_id || ''))}
                disabled={cloudBurstCancellingJob || cloudBurstActiveRun.can_cancel === false}
              >
                {cloudBurstCancellingJob ? 'Cancelling...' : 'Cancel Job'}
              </button>
            </div>
            {cloudBurstActiveRun.artifacts?.last_sync_summary && (
              <div className="training-alert training-alert--warning training-alert--tight">
                Last sync: {String(cloudBurstActiveRun.artifacts?.last_sync_summary?.status || 'unknown')}
                {' • '}
                {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.copied_count || 0)}
                {' / '}
                {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.file_count || 0)}
                {' files'}
                {' • unchanged '}
                {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.unchanged_count || 0)}
                {' • remaining '}
                {Number(cloudBurstActiveRun.artifacts?.last_sync_summary?.remaining_count || 0)}
                {Array.isArray(cloudBurstActiveRun.artifacts?.last_sync_summary?.errors)
                  && cloudBurstActiveRun.artifacts?.last_sync_summary?.errors?.length
                  ? ` • errors: ${cloudBurstActiveRun.artifacts?.last_sync_summary?.errors?.slice(0, 2).join(' | ')}`
                  : ''}
              </div>
            )}
            {String(cloudBurstActiveRun.artifacts?.last_sync_summary?.next_cursor || '').trim() && (
              <div className="form-hint">
                Next sync cursor available. Continue sync to fetch remaining artifacts.
              </div>
            )}
            {cloudBurstMetrics.length > 0 && (
              <div className="cloud-burst-metrics">
                <div className="resolved-defaults-panel__subtitle">Live Metrics</div>
                <div className="cloud-burst-metrics__kv">
                  <span>
                    step {Number.isFinite(Number(cloudBurstLatestMetric?.step))
                      ? Number(cloudBurstLatestMetric?.step)
                      : '-'}
                  </span>
                  <span>
                    train {Number.isFinite(Number(cloudBurstLatestMetric?.train_loss))
                      ? Number(cloudBurstLatestMetric?.train_loss).toFixed(4)
                      : '-'}
                  </span>
                  <span>
                    eval {Number.isFinite(Number(cloudBurstLatestMetric?.eval_loss))
                      ? Number(cloudBurstLatestMetric?.eval_loss).toFixed(4)
                      : '-'}
                  </span>
                </div>
                <svg
                  className="cloud-burst-metrics__chart"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  role="img"
                  aria-label="Cloud burst loss metrics trend"
                >
                  <rect x="0" y="0" width="100" height="100" className="cloud-burst-metrics__bg" />
                  {cloudBurstMetricSeries.train && (
                    <polyline
                      fill="none"
                      points={cloudBurstMetricSeries.train}
                      className="cloud-burst-metrics__line cloud-burst-metrics__line--train"
                    />
                  )}
                  {cloudBurstMetricSeries.eval && (
                    <polyline
                      fill="none"
                      points={cloudBurstMetricSeries.eval}
                      className="cloud-burst-metrics__line cloud-burst-metrics__line--eval"
                    />
                  )}
                </svg>
              </div>
            )}
            {Array.isArray(cloudBurstActiveRun.logs_tail) && cloudBurstActiveRun.logs_tail.length > 0 && (
              <div>
                <div className="resolved-defaults-panel__subtitle">Job Logs (tail)</div>
                <pre className="resolved-defaults-panel__json">
                  {cloudBurstActiveRun.logs_tail.join('\n')}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </details>
  );
}
