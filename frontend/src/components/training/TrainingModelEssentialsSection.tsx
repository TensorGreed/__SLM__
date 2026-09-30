/**
 * Model & essentials column: base model + recommender / benchmark (Power), core
 * hyperparameters and the hyperparameter sweep.
 * Reads/writes the shared config form; TrainingPanel owns loading + payload.
 */

import HyperparameterSweepPanel from './HyperparameterSweepPanel';
import ParetoComparisonPanel from './ParetoComparisonPanel';
import type {
  TrainingRuntimeSpec,
  TrainingRuntimeCatalogResponse,
  ModelWizardRecommendation,
  ModelWizardResponse,
  ModelBenchmarkSweepResponse,
  ModelBenchmarkHistoryRun,
  ModelIntrospectionSummary,
} from './trainingPanelTypes';
import type { ModelSelectionApplySource } from './trainingPanelUtils';
import { MODEL_WIZARD_TASK_PROFILES } from './trainingPanelUtils';
import type { TrainingConfigForm } from './useTrainingConfigForm';

interface TrainingModelEssentialsSectionProps {
  form: TrainingConfigForm;
  applyBenchmarkWinner: () => void;
  applyModelSelectionChoice: (args: {
    modelId: string;
    rankIndex: number;
    selectedScore?: number;
    defaults?: ModelWizardRecommendation['suggested_defaults'];
    applySource?: ModelSelectionApplySource;
  }) => void;
  applyModelWizardRecommendation: (item: ModelWizardRecommendation, rankIndex: number) => void;
  baseModelIntrospection: ModelIntrospectionSummary | null;
  baseModelIntrospectionError: string;
  baseModelIntrospectionLoading: boolean;
  benchmarkError: string;
  benchmarkHistory: ModelBenchmarkHistoryRun[];
  benchmarkLoading: boolean;
  benchmarkResult: ModelBenchmarkSweepResponse | null;
  buildTrainingConfigPayload: () => Record<string, unknown>;
  introspectBaseModel: (options?: { modelId?: string; silent?: boolean }) => Promise<void>;
  isAlignmentMode: boolean;
  projectId: number;
  runModelBenchmarkSweep: () => Promise<void>;
  runModelWizard: (options?: { silent?: boolean }) => Promise<void>;
  runtimeCatalog: TrainingRuntimeCatalogResponse | null;
  runtimeCatalogError: string;
  selectedRuntimeModalities: string[];
  selectedRuntimeModalitiesDeclared: boolean | null;
  selectedRuntimeSpec: TrainingRuntimeSpec | null;
  setWizardPrimaryLanguage: (value: string) => void;
  setWizardTargetDevice: (value: string) => void;
  setWizardTaskProfile: (value: string) => void;
  setWizardVramGb: (value: string) => void;
  showSetupConfig: boolean;
  showSetupPower: boolean;
  wizardError: string;
  wizardLoading: boolean;
  wizardPrimaryLanguage: string;
  wizardResult: ModelWizardResponse | null;
  wizardTargetDevice: string;
  wizardTaskProfile: string;
  wizardVramGb: string;
}

export default function TrainingModelEssentialsSection({
  form,
  applyBenchmarkWinner,
  applyModelSelectionChoice,
  applyModelWizardRecommendation,
  baseModelIntrospection,
  baseModelIntrospectionError,
  baseModelIntrospectionLoading,
  benchmarkError,
  benchmarkHistory,
  benchmarkLoading,
  benchmarkResult,
  buildTrainingConfigPayload,
  introspectBaseModel,
  isAlignmentMode,
  projectId,
  runModelBenchmarkSweep,
  runModelWizard,
  runtimeCatalog,
  runtimeCatalogError,
  selectedRuntimeModalities,
  selectedRuntimeModalitiesDeclared,
  selectedRuntimeSpec,
  setWizardPrimaryLanguage,
  setWizardTargetDevice,
  setWizardTaskProfile,
  setWizardVramGb,
  showSetupConfig,
  showSetupPower,
  wizardError,
  wizardLoading,
  wizardPrimaryLanguage,
  wizardResult,
  wizardTargetDevice,
  wizardTaskProfile,
  wizardVramGb,
}: TrainingModelEssentialsSectionProps) {
  const {
    baseModel,
    setBaseModel,
    trainingMode,
    setTrainingMode,
    trainingRuntimeId,
    setTrainingRuntimeId,
    taskType,
    setTaskType,
    trainerBackend,
    setTrainerBackend,
    chatTemplate,
    setChatTemplate,
    lr,
    setLr,
    epochs,
    setEpochs,
    batchSize,
    setBatchSize,
    gradientAccumulationSteps,
    setGradientAccumulationSteps,
    maxSeqLength,
    setMaxSeqLength,
    optimizer,
    setOptimizer,
    saveSteps,
    setSaveSteps,
    evalSteps,
    setEvalSteps,
    useProfileDefaults,
    touchedConfig,
    setTouchedConfig,
  } = form;

  return (
        <div>
          <h4 className="training-config-section-title">
            {showSetupPower ? 'Model & Recommender' : 'Essentials'}
          </h4>
          {showSetupPower && (
          <div className="training-model-wizard">
            <div className="training-model-wizard__head">
              <strong>Model Selection Wizard</strong>
              <span>Pick hardware + goal, then apply a recommended base model.</span>
            </div>
            <div className="training-model-wizard__controls">
              <div className="form-group">
                <label className="form-label">Target Device</label>
                <select
                  className="input"
                  value={wizardTargetDevice}
                  onChange={(e) => setWizardTargetDevice(e.target.value)}
                >
                  <option value="mobile">Mobile</option>
                  <option value="laptop">Laptop</option>
                  <option value="server">Server</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Primary Goal</label>
                <select
                  className="input"
                  value={wizardPrimaryLanguage}
                  onChange={(e) => setWizardPrimaryLanguage(e.target.value)}
                >
                  <option value="english">English</option>
                  <option value="multilingual">Multilingual</option>
                  <option value="coding">Coding</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Available VRAM (GB)</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  step={1}
                  value={wizardVramGb}
                  onChange={(e) => setWizardVramGb(e.target.value)}
                  placeholder="Optional"
                />
              </div>
              <div className="form-group">
                <label className="form-label">Task Profile</label>
                <select
                  className="input"
                  value={wizardTaskProfile}
                  onChange={(e) => setWizardTaskProfile(e.target.value)}
                >
                  {MODEL_WIZARD_TASK_PROFILES.map((profile) => (
                    <option key={profile} value={profile}>
                      {profile}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="training-model-wizard__actions">
              <button
                className="btn btn-secondary"
                onClick={() => void runModelWizard()}
                disabled={wizardLoading}
              >
                {wizardLoading ? 'Finding Models...' : 'Recommend Models'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => void runModelBenchmarkSweep()}
                disabled={wizardLoading || benchmarkLoading}
              >
                {benchmarkLoading ? 'Running Benchmark...' : 'Run Benchmark Sweep'}
              </button>
              <span className="training-model-wizard__hint">
                Uses lightweight heuristics from model size, VRAM fit, and task goal.
              </span>
            </div>
            {wizardError && (
              <div className="training-alert training-alert--warning training-alert--tight">
                {wizardError}
              </div>
            )}
            {wizardResult?.blocked_by_data_shape && (
              <div
                className="training-alert training-alert--error"
                data-testid="recommender-data-shape-banner"
                style={{
                  padding: 'var(--space-md)',
                  border: '1px solid var(--color-error, #b91c1c)',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(185, 28, 28, 0.05)',
                }}
              >
                <div
                  style={{
                    fontWeight: 600,
                    marginBottom: 6,
                    color: 'var(--color-error, #b91c1c)',
                  }}
                >
                  Data shape blocks model choice
                </div>
                <div style={{ fontSize: '0.875rem' }}>
                  {wizardResult.data_shape_message
                    || 'Your prepared training data has no target field. Fix the data shape before picking a model.'}
                </div>
              </div>
            )}
            {Array.isArray(wizardResult?.warnings) && wizardResult.warnings.length > 0 && (
              <div className="training-model-wizard__warnings">
                {wizardResult.warnings.join(' | ')}
              </div>
            )}
            {wizardResult?.adaptive_ranking?.enabled && (
              <div className="training-model-wizard__warnings">
                Adaptive ranking active ({wizardResult.adaptive_ranking.context_label || 'global'}): boosted{' '}
                {wizardResult.adaptive_ranking.boosted_model_count || 0} model(s) from prior applies.
              </div>
            )}
            {benchmarkError && (
              <div className="training-alert training-alert--warning training-alert--tight">
                {benchmarkError}
              </div>
            )}
            {Array.isArray(benchmarkResult?.warnings) && benchmarkResult.warnings.length > 0 && (
              <div className="training-model-wizard__warnings">
                {benchmarkResult.warnings.join(' | ')}
              </div>
            )}
            {Array.isArray(benchmarkResult?.matrix) && benchmarkResult.matrix.length > 0 && (
              <div className="training-model-benchmark">
                <div className="training-model-benchmark__head">
                  <strong>Benchmark Sweep ({benchmarkResult.benchmark_mode || 'real_sampled'})</strong>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={applyBenchmarkWinner}
                  >
                    Apply Benchmark Winner
                  </button>
                </div>
                <div className="training-model-benchmark__meta">
                  Run {benchmarkResult.run_id || 'n/a'} • Sampled {benchmarkResult.sampled_row_count || 0} rows • Avg{' '}
                  {Number.isFinite(Number(benchmarkResult.sampled_avg_tokens))
                    ? `${Number(benchmarkResult.sampled_avg_tokens).toFixed(1)} tokens`
                    : 'n/a'}
                </div>
                <div className="training-model-benchmark__summary">
                  <span>Best quality: {benchmarkResult.tradeoff_summary?.best_quality_model_id || 'n/a'}</span>
                  <span>Best speed: {benchmarkResult.tradeoff_summary?.best_speed_model_id || 'n/a'}</span>
                  <span>Best balance: {benchmarkResult.tradeoff_summary?.best_balance_model_id || 'n/a'}</span>
                </div>
                <div className="training-model-benchmark__rows">
                  {benchmarkResult.matrix.map((row, index) => (
                    <div className="training-model-benchmark__row" key={`${benchmarkResult.run_id || 'run'}-${row.model_id || index}`}>
                      <div className="training-model-benchmark__row-head">
                        <strong>#{row.rank || index + 1} {row.model_id || 'unknown'}</strong>
                        <span>{row.benchmark_mode || 'sampled_heuristic'}</span>
                      </div>
                      <div className="training-model-benchmark__row-metrics">
                        <span>Quality {Number.isFinite(Number(row.estimated_accuracy_percent)) ? `${Number(row.estimated_accuracy_percent).toFixed(1)}%` : 'n/a'}</span>
                        <span>Latency {Number.isFinite(Number(row.estimated_latency_ms)) ? `${Number(row.estimated_latency_ms).toFixed(1)} ms` : 'n/a'}</span>
                        <span>Throughput {Number.isFinite(Number(row.estimated_throughput_tps)) ? `${Number(row.estimated_throughput_tps).toFixed(1)} t/s` : 'n/a'}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <ParetoComparisonPanel
                  matrix={benchmarkResult.matrix}
                  currentBaseModel={baseModel}
                  bestBalanceModelId={benchmarkResult.tradeoff_summary?.best_balance_model_id}
                  onPromote={(row) => {
                    const winnerId = String(row.model_id || '').trim();
                    if (!winnerId) return;
                    const rankIndex = (benchmarkResult.matrix || []).findIndex(
                      (r) => String(r.model_id || '').trim() === winnerId,
                    );
                    applyModelSelectionChoice({
                      modelId: winnerId,
                      rankIndex: rankIndex >= 0 ? rankIndex : 0,
                      selectedScore: Number(row.estimated_quality_score),
                      defaults: row.suggested_defaults,
                      applySource: 'benchmark',
                    });
                  }}
                />
              </div>
            )}
            {benchmarkHistory.length > 0 && (
              <div className="training-model-benchmark__history">
                <strong>Recent Benchmark Runs</strong>
                {benchmarkHistory.slice(0, 3).map((run, idx) => (
                  <div className="training-model-benchmark__history-row" key={`${run.run_id || 'history'}-${idx}`}>
                    <span>{run.run_id || 'run'}</span>
                    <span>{run.benchmark_mode || 'real_sampled'}</span>
                    <span>
                      Winner:{' '}
                      {run.tradeoff_summary?.best_balance_model_id
                        || run.matrix?.[0]?.model_id
                        || 'n/a'}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {Array.isArray(wizardResult?.recommendations) && wizardResult.recommendations.length > 0 && (
              <div className="training-model-wizard__results">
                {wizardResult.recommendations.map((item, idx) => (
                  <div className="training-model-wizard__card" key={item.model_id}>
                    <div className="training-model-wizard__card-head">
                      <strong>{item.model_id}</strong>
                      {Number.isFinite(Number(item.match_score)) && (
                        <span className="badge badge-info">score {Number(item.match_score).toFixed(2)}</span>
                      )}
                    </div>
                    <div className="training-model-wizard__card-meta">
                      {item.params_b ? `${item.params_b}B params` : 'unknown size'} • min VRAM{' '}
                      {Number.isFinite(Number(item.estimated_min_vram_gb))
                        ? `${Number(item.estimated_min_vram_gb)} GB`
                        : 'n/a'}
                    </div>
                    <div className="training-model-wizard__card-meta">
                      {item.architecture || 'unknown'} • ctx{' '}
                      {Number.isFinite(Number(item.context_length))
                        ? Number(item.context_length)
                        : 'n/a'}
                      {item.license ? ` • ${item.license}` : ''}
                    </div>
                    {(Number.isFinite(Number(item.introspection_estimated_min_vram_gb))
                      || Number.isFinite(Number(item.introspection_estimated_ideal_vram_gb))) && (
                        <div className="training-model-wizard__card-meta">
                          Introspection VRAM:{' '}
                          {Number.isFinite(Number(item.introspection_estimated_min_vram_gb))
                            ? `${Number(item.introspection_estimated_min_vram_gb)} GB min`
                            : 'n/a'}
                          {' / '}
                          {Number.isFinite(Number(item.introspection_estimated_ideal_vram_gb))
                            ? `${Number(item.introspection_estimated_ideal_vram_gb)} GB ideal`
                            : 'n/a'}
                        </div>
                      )}
                    {item.metadata_source && (
                      <div className="training-model-wizard__meta-source">
                        Metadata: {item.metadata_source}
                      </div>
                    )}
                    {Array.isArray(item.match_reasons) && item.match_reasons.length > 0 && (
                      <div className="training-model-wizard__reasons">
                        {item.match_reasons.slice(0, 3).map((reason, idx) => (
                          <div key={`${item.model_id}-reason-${idx}`}>{reason}</div>
                        ))}
                      </div>
                    )}
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => applyModelWizardRecommendation(item, idx)}
                    >
                      Apply Model + Defaults
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
          )}
          {showSetupPower && (
            <HyperparameterSweepPanel
              projectId={projectId}
              baseModel={baseModel}
              baseConfig={buildTrainingConfigPayload()}
            />
          )}
          {showSetupConfig && (
            <>
          <div className="form-group">
            <label className="form-label">Base Model</label>
            <input className="input" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />
            <div className="form-inline-actions">
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => void introspectBaseModel()}
                disabled={baseModelIntrospectionLoading}
              >
                {baseModelIntrospectionLoading ? 'Inspecting...' : 'Introspect Model'}
              </button>
              <span className="form-hint">
                Reads local/HF config metadata: architecture, context, license, memory hints.
              </span>
            </div>
            {baseModelIntrospectionError && (
              <div className="form-hint form-hint-warning">
                {baseModelIntrospectionError}
              </div>
            )}
            {baseModelIntrospection && (
              <div className="training-model-introspection">
                <div className="training-model-introspection__head">
                  <strong>Model Introspection</strong>
                  <span>{baseModelIntrospection.source || 'none'}</span>
                </div>
                <div className="training-model-introspection__grid">
                  <div className="training-model-introspection__row">
                    <span>Model ID</span>
                    <strong>{baseModelIntrospection.model_id || baseModel}</strong>
                  </div>
                  <div className="training-model-introspection__row">
                    <span>Architecture</span>
                    <strong>{baseModelIntrospection.architecture || 'unknown'}</strong>
                  </div>
                  <div className="training-model-introspection__row">
                    <span>Model Type</span>
                    <strong>{baseModelIntrospection.model_type || 'unknown'}</strong>
                  </div>
                  <div className="training-model-introspection__row">
                    <span>Context Length</span>
                    <strong>
                      {Number.isFinite(Number(baseModelIntrospection.context_length))
                        ? Number(baseModelIntrospection.context_length)
                        : 'n/a'}
                    </strong>
                  </div>
                  <div className="training-model-introspection__row">
                    <span>License</span>
                    <strong>{baseModelIntrospection.license || 'unknown'}</strong>
                  </div>
                  <div className="training-model-introspection__row">
                    <span>Params (est)</span>
                    <strong>
                      {Number.isFinite(Number(baseModelIntrospection.params_estimate_b))
                        ? `${Number(baseModelIntrospection.params_estimate_b).toFixed(2)}B`
                        : 'n/a'}
                    </strong>
                  </div>
                  <div className="training-model-introspection__row">
                    <span>VRAM (min/ideal)</span>
                    <strong>
                      {Number.isFinite(Number(baseModelIntrospection.memory_profile?.estimated_min_vram_gb))
                        ? `${Number(baseModelIntrospection.memory_profile?.estimated_min_vram_gb)} GB`
                        : 'n/a'}
                      {' / '}
                      {Number.isFinite(Number(baseModelIntrospection.memory_profile?.estimated_ideal_vram_gb))
                        ? `${Number(baseModelIntrospection.memory_profile?.estimated_ideal_vram_gb)} GB`
                        : 'n/a'}
                    </strong>
                  </div>
                </div>
                {Array.isArray(baseModelIntrospection.warnings) && baseModelIntrospection.warnings.length > 0 && (
                  <div className="training-model-introspection__warnings">
                    {baseModelIntrospection.warnings.join(' | ')}
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="training-grid-2">
            <div className="form-group">
              <label className="form-label">Training Mode</label>
              <select
                className="input"
                value={trainingMode}
                onChange={(e) => {
                  const nextMode = e.target.value;
                  setTrainingMode(nextMode);
                  setTouchedConfig((prev) => ({ ...prev, training_mode: true }));
                  if (nextMode === 'dpo' || nextMode === 'orpo') {
                    setTaskType('causal_lm');
                    setTouchedConfig((prev) => ({ ...prev, task_type: true }));
                  }
                }}
              >
                <option value="sft">SFT</option>
                <option value="domain_pretrain">Continued pretraining (documents)</option>
                <option value="dpo">DPO</option>
                <option value="orpo">ORPO</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Runtime</label>
              <select
                className="input"
                value={trainingRuntimeId}
                onChange={(e) => {
                  setTrainingRuntimeId(e.target.value);
                  setTouchedConfig((prev) => ({ ...prev, training_runtime_id: true }));
                }}
              >
                <option value="auto">
                  Auto
                  {runtimeCatalog?.default_runtime_id ? ` (${runtimeCatalog.default_runtime_id})` : ''}
                </option>
                {(runtimeCatalog?.runtimes || []).map((runtime) => (
                  <option key={runtime.runtime_id} value={runtime.runtime_id}>
                    {runtime.label}
                    {runtime.execution_backend ? ` [${runtime.execution_backend}]` : ''}
                  </option>
                ))}
              </select>
              {runtimeCatalogError && (
                <div className="form-hint form-hint-warning">
                  {runtimeCatalogError}
                </div>
              )}
              {selectedRuntimeSpec && (
                <div className="form-hint">
                  Modalities: {selectedRuntimeModalities.length > 0 ? selectedRuntimeModalities.join(', ') : 'text'}
                  {selectedRuntimeModalitiesDeclared === false ? ' (assumed default)' : ''}
                </div>
              )}
            </div>
          </div>
          <div className="training-grid-2">
            <div className="form-group">
              <label className="form-label">Epochs</label>
              <input
                className="input"
                type="number"
                value={epochs}
                onChange={(e) => {
                  setEpochs(Number(e.target.value) || 1);
                  setTouchedConfig((prev) => ({ ...prev, num_epochs: true }));
                }}
              />
              {useProfileDefaults && !touchedConfig.num_epochs && (
                <p className="form-hint">Auto: scaled to your dataset size unless you change it.</p>
              )}
            </div>
            <div className="form-group">
              <label className="form-label">Batch Size</label>
              <input
                className="input"
                type="number"
                value={batchSize}
                onChange={(e) => {
                  setBatchSize(Number(e.target.value) || 1);
                  setTouchedConfig((prev) => ({ ...prev, batch_size: true }));
                }}
              />
            </div>
          </div>
          <div className="training-grid-2">
            <div className="form-group">
              <label className="form-label">Learning Rate</label>
              <input
                className="input"
                value={lr}
                onChange={(e) => {
                  setLr(e.target.value);
                  setTouchedConfig((prev) => ({ ...prev, learning_rate: true }));
                }}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Optimizer</label>
              <select
                className="input"
                value={optimizer}
                onChange={(e) => {
                  setOptimizer(e.target.value);
                  setTouchedConfig((prev) => ({ ...prev, optimizer: true }));
                }}
              >
                <option value="paged_adamw_8bit">Paged AdamW (8-bit)</option>
                <option value="adamw_torch">AdamW</option>
              </select>
            </div>
          </div>
            </>
          )}
          {showSetupPower && (
            <>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Task Type</label>
                  <select
                    className="input"
                    value={taskType}
                    onChange={(e) => {
                      setTaskType(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, task_type: true }));
                    }}
                    disabled={isAlignmentMode}
                  >
                    <option value="causal_lm">Causal LM</option>
                    <option value="seq2seq">Seq2Seq</option>
                    <option value="classification">Classification</option>
                  </select>
                  {isAlignmentMode && (
                    <div className="form-hint">
                      DPO/ORPO currently run on causal LM preference pairs.
                    </div>
                  )}
                </div>
                <div className="form-group">
                  <label className="form-label">Trainer Backend</label>
                  <select
                    className="input"
                    value={trainerBackend}
                    onChange={(e) => {
                      setTrainerBackend(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, trainer_backend: true }));
                    }}
                  >
                    <option value="auto">Auto (HF Trainer)</option>
                    <option value="hf_trainer">HF Trainer</option>
                    <option value="trl_sft">TRL SFTTrainer</option>
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Chat Template</label>
                <select
                  className="input"
                  value={chatTemplate}
                  onChange={(e) => {
                    setChatTemplate(e.target.value);
                    setTouchedConfig((prev) => ({ ...prev, chat_template: true }));
                  }}
                >
                  <option value="llama3">Llama-3</option>
                  <option value="chatml">ChatML</option>
                  <option value="zephyr">Zephyr</option>
                  <option value="phi3">Phi-3</option>
                </select>
              </div>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Grad Accum Steps</label>
                  <input
                    className="input"
                    type="number"
                    min={1}
                    value={gradientAccumulationSteps}
                    onChange={(e) => {
                      setGradientAccumulationSteps(Math.max(1, Number(e.target.value) || 1));
                      setTouchedConfig((prev) => ({ ...prev, gradient_accumulation_steps: true }));
                    }}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Max Seq Length</label>
                  <input
                    className="input"
                    type="number"
                    min={128}
                    value={maxSeqLength}
                    onChange={(e) => {
                      setMaxSeqLength(Math.max(128, Number(e.target.value) || 128));
                      setTouchedConfig((prev) => ({ ...prev, max_seq_length: true }));
                    }}
                  />
                </div>
              </div>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Save Steps</label>
                  <input
                    className="input"
                    type="number"
                    min={1}
                    value={saveSteps}
                    onChange={(e) => {
                      setSaveSteps(Math.max(1, Number(e.target.value) || 1));
                      setTouchedConfig((prev) => ({ ...prev, save_steps: true }));
                    }}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Eval Steps</label>
                  <input
                    className="input"
                    type="number"
                    min={1}
                    value={evalSteps}
                    onChange={(e) => {
                      setEvalSteps(Math.max(1, Number(e.target.value) || 1));
                      setTouchedConfig((prev) => ({ ...prev, eval_steps: true }));
                    }}
                  />
                </div>
              </div>
            </>
          )}
        </div>
  );
}
