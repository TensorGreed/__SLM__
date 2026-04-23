import { useEffect, useMemo, useState } from 'react';
import { useOutletContext } from 'react-router-dom';

import api from '../api/client';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectModelsPage.css';

function errorDetail(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (err instanceof Error && err.message) return err.message;
  return fallback;
}

interface BaseModelRecord {
  id: number;
  model_key: string;
  source_type: string;
  source_ref: string;
  display_name: string;
  model_family: string;
  architecture: string;
  tokenizer?: string | null;
  chat_template?: string | null;
  context_length?: number | null;
  params_estimate_b?: number | null;
  license?: string | null;
  peft_support: boolean;
  full_finetune_support: boolean;
  supported_task_families: string[];
  training_mode_support: string[];
  estimated_hardware_needs?: {
    estimated_min_vram_gb?: number | null;
    estimated_ideal_vram_gb?: number | null;
  } | null;
}

interface CompatibilityReason {
  code: string;
  severity: string;
  message: string;
  unblock_actions?: string[];
}

interface CompatibilityRow {
  model_id: number;
  model_key: string;
  compatibility_score: number;
  compatible: boolean;
  reason_codes: string[];
  why_recommended: CompatibilityReason[];
  why_risky: CompatibilityReason[];
  recommended_next_actions: string[];
  model?: BaseModelRecord;
}

interface ModelListResponse {
  count: number;
  models: BaseModelRecord[];
}

interface CompatibleResponse {
  project_id: number;
  count: number;
  compatible_count: number;
  models: CompatibilityRow[];
}

interface ValidateResponse {
  model_id: number;
  model_key: string;
  compatibility_score: number;
  compatible: boolean;
  reason_codes: string[];
  why_recommended: CompatibilityReason[];
  why_risky: CompatibilityReason[];
  recommended_next_actions: string[];
}

export default function ProjectModelsPage() {
  const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
  const [rows, setRows] = useState<BaseModelRecord[]>([]);
  const [recommendations, setRecommendations] = useState<CompatibilityRow[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [selectedValidation, setSelectedValidation] = useState<ValidateResponse | null>(null);
  const [selectedValidationModel, setSelectedValidationModel] = useState<BaseModelRecord | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState('');

  const [family, setFamily] = useState('');
  const [licenseToken, setLicenseToken] = useState('');
  const [hardwareFit, setHardwareFit] = useState('');
  const [search, setSearch] = useState('');
  const [trainingMode, setTrainingMode] = useState('');
  const [minContextLength, setMinContextLength] = useState('');
  const [maxParamsB, setMaxParamsB] = useState('');

  const loadData = async () => {
    setError('');
    setIsLoading(true);
    try {
      const modelParams: Record<string, string | number> = {};
      if (family.trim()) modelParams.family = family.trim();
      if (licenseToken.trim()) modelParams.license = licenseToken.trim();
      if (hardwareFit.trim()) modelParams.hardware_fit = hardwareFit.trim();
      if (search.trim()) modelParams.search = search.trim();
      if (trainingMode.trim()) modelParams.training_mode = trainingMode.trim();
      if (minContextLength.trim()) modelParams.min_context_length = Number(minContextLength);
      if (maxParamsB.trim()) modelParams.max_params_b = Number(maxParamsB);

      const [modelResp, compatibleResp] = await Promise.all([
        api.get<ModelListResponse>('/models', { params: modelParams }),
        api.get<CompatibleResponse>(`/projects/${projectId}/models/compatible`, {
          params: {
            ...modelParams,
            limit: 20,
            include_incompatible: true,
            allow_network: false,
          },
        }),
      ]);

      setRows(Array.isArray(modelResp.data?.models) ? modelResp.data.models : []);
      setRecommendations(Array.isArray(compatibleResp.data?.models) ? compatibleResp.data.models : []);
    } catch (err) {
      setError(errorDetail(err, 'Failed to load model registry data.'));
      setRows([]);
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const recommendationByModelKey = useMemo(() => {
    const index = new Map<string, CompatibilityRow>();
    for (const item of recommendations) {
      if (item?.model_key) index.set(item.model_key, item);
    }
    return index;
  }, [recommendations]);

  const validateModel = async (model: BaseModelRecord) => {
    setError('');
    setStatusMessage('');
    setSelectedValidation(null);
    setSelectedValidationModel(model);
    setSelectedModelKey(model.model_key);
    try {
      const resp = await api.post<ValidateResponse>(`/projects/${projectId}/models/validate`, {
        model_id: model.id,
        allow_network: false,
      });
      setSelectedValidation(resp.data);
    } catch (err) {
      setError(errorDetail(err, 'Validation failed.'));
    }
  };

  const chooseModelForProject = async (model: BaseModelRecord) => {
    setError('');
    setStatusMessage('');
    try {
      await api.put(`/projects/${projectId}`, {
        base_model_name: model.source_ref || model.display_name,
      });
      setStatusMessage(`Selected '${model.display_name}' as project base model.`);
    } catch (err) {
      setError(errorDetail(err, 'Failed to set base model.'));
    }
  };

  const tokenizerWarning = selectedValidation?.reason_codes?.includes('TOKENIZER_METADATA_MISSING');
  const chatTemplateWarning = selectedValidation?.reason_codes?.includes('CHAT_TEMPLATE_MISSING');

  return (
    <div className="workspace-page model-registry-page">
      <section className="workspace-page-header">
        <div>
          <h2 className="workspace-page-title">Universal Base Model Registry</h2>
          <p className="workspace-page-subtitle">
            Import and inspect base models, then validate compatibility with your Domain Blueprint, dataset adapter, runtime, and target.
          </p>
        </div>
      </section>

      <section className="card model-registry-filters">
        <div className="model-filter-grid">
          <label>
            Family
            <input value={family} onChange={(e) => setFamily(e.target.value)} placeholder="llama, qwen, t5, bert..." />
          </label>
          <label>
            License
            <input value={licenseToken} onChange={(e) => setLicenseToken(e.target.value)} placeholder="apache, mit, llama..." />
          </label>
          <label>
            Hardware Fit
            <select value={hardwareFit} onChange={(e) => setHardwareFit(e.target.value)}>
              <option value="">Any</option>
              <option value="mobile">Mobile</option>
              <option value="laptop">Laptop</option>
              <option value="server">Server</option>
            </select>
          </label>
          <label>
            Training Mode
            <input value={trainingMode} onChange={(e) => setTrainingMode(e.target.value)} placeholder="sft, domain_pretrain, dpo..." />
          </label>
          <label>
            Search
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="model name, architecture..." />
          </label>
          <label>
            Min Context
            <input value={minContextLength} onChange={(e) => setMinContextLength(e.target.value)} placeholder="1024" />
          </label>
          <label>
            Max Params (B)
            <input value={maxParamsB} onChange={(e) => setMaxParamsB(e.target.value)} placeholder="7" />
          </label>
        </div>
        <div className="model-filter-actions">
          <button className="btn btn-secondary" onClick={() => void loadData()} disabled={isLoading}>
            {isLoading ? 'Loading...' : 'Apply Filters'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => {
              setFamily('');
              setLicenseToken('');
              setHardwareFit('');
              setSearch('');
              setTrainingMode('');
              setMinContextLength('');
              setMaxParamsB('');
              void loadData();
            }}
            disabled={isLoading}
          >
            Reset
          </button>
        </div>
      </section>

      {error && <div className="alert alert-error">{error}</div>}
      {statusMessage && <div className="alert alert-info">{statusMessage}</div>}

      {selectedValidation && selectedValidationModel && (
        <section className="card model-validation-panel">
          <div className="model-validation-header">
            <h3>
              Validation: {selectedValidationModel.display_name} ({selectedValidation.compatibility_score.toFixed(3)})
            </h3>
            <span className={`badge ${selectedValidation.compatible ? 'badge-success' : 'badge-danger'}`}>
              {selectedValidation.compatible ? 'Compatible' : 'Not Compatible'}
            </span>
          </div>
          {(tokenizerWarning || chatTemplateWarning) && (
            <div className="model-warning-strip" role="alert">
              <strong>Tokenizer/Template Warning:</strong>{' '}
              {tokenizerWarning && 'Tokenizer metadata is missing. '}
              {chatTemplateWarning && 'Chat template is missing for this model.'}
            </div>
          )}
          <div className="model-reason-grid">
            <div>
              <h4>Why Recommended</h4>
              <ul>
                {(selectedValidation.why_recommended || []).slice(0, 5).map((item, idx) => (
                  <li key={`rec-${idx}`}>{item.message}</li>
                ))}
              </ul>
            </div>
            <div>
              <h4>Why Risky</h4>
              <ul>
                {(selectedValidation.why_risky || []).slice(0, 6).map((item, idx) => (
                  <li key={`risk-${idx}`}>{item.message}</li>
                ))}
              </ul>
            </div>
          </div>
          {(selectedValidation.recommended_next_actions || []).length > 0 && (
            <div className="model-next-actions">
              <h4>Unblock Actions</h4>
              <ul>
                {(selectedValidation.recommended_next_actions || []).slice(0, 8).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      <section className="model-grid">
        {rows.map((model) => {
          const compat = recommendationByModelKey.get(model.model_key);
          const risky = Array.isArray(compat?.why_risky) ? compat?.why_risky : [];
          const recommended = Array.isArray(compat?.why_recommended) ? compat?.why_recommended : [];
          return (
            <article
              key={model.model_key}
              className={`card model-card ${selectedModelKey === model.model_key ? 'model-card-selected' : ''}`}
            >
              <div className="model-card-head">
                <h3>{model.display_name}</h3>
                <span className={`badge ${compat?.compatible ? 'badge-success' : 'badge-warning'}`}>
                  {compat?.compatible ? 'Recommended' : 'Risky'}
                </span>
              </div>
              <div className="model-meta">
                <span>Family: {model.model_family}</span>
                <span>Architecture: {model.architecture}</span>
                <span>Context: {model.context_length || 'n/a'}</span>
                <span>Params: {model.params_estimate_b ? `${model.params_estimate_b.toFixed(2)}B` : 'n/a'}</span>
                <span>License: {model.license || 'unknown'}</span>
                <span>Training: {(model.training_mode_support || []).join(', ') || 'n/a'}</span>
              </div>
              {compat && (
                <div className="model-compat-summary">
                  <div className="model-score">Compatibility Score: {(compat.compatibility_score || 0).toFixed(3)}</div>
                  <div className="model-why-grid">
                    <div>
                      <h4>Why recommended</h4>
                      <ul>
                        {recommended.slice(0, 3).map((item, idx) => (
                          <li key={`${model.model_key}-r-${idx}`}>{item.message}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4>Why risky</h4>
                      <ul>
                        {risky.slice(0, 3).map((item, idx) => (
                          <li key={`${model.model_key}-k-${idx}`}>{item.message}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
              <div className="model-card-actions">
                <button className="btn btn-secondary" onClick={() => void validateModel(model)}>
                  Validate For Project
                </button>
                <button className="btn btn-primary" onClick={() => void chooseModelForProject(model)}>
                  Choose Model
                </button>
              </div>
            </article>
          );
        })}
      </section>
    </div>
  );
}
