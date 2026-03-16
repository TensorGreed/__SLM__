import { useEffect, useState } from 'react';

import api from '../../api/client';
import './AlignmentScaffoldPanel.css';

interface AlignmentRecipe {
  recipe_id: string;
  display_name: string;
  training_mode: string;
  description?: string;
}

interface AlignmentRecipeCatalogResponse {
  recipes?: AlignmentRecipe[];
}

interface AlignmentRecipeResolveResponse {
  resolved_config?: Record<string, unknown>;
}

interface AlignmentContractResponse {
  ok?: boolean;
  total_rows?: number;
  valid_rows?: number;
  invalid_rows?: number;
  coverage?: number;
  errors?: string[];
  warnings?: string[];
}

interface AlignmentJudgeResponse {
  scored_count?: number;
  keep_count?: number;
  drop_count?: number;
  average_quality_score?: number;
  scored_rows?: Array<{ row_index?: number; quality_score?: number; keep?: boolean; prompt_preview?: string }>;
}

interface AlignmentDatasetSummaryResponse {
  source_path?: string;
  exists?: boolean;
  sample_size?: number;
  contract?: AlignmentContractResponse;
  quality?: {
    quality_threshold?: number;
    scored_count?: number;
    keep_count?: number;
    drop_count?: number;
    keep_ratio?: number;
    average_quality_score?: number;
  };
  rows_preview?: Array<{ prompt?: string; chosen?: string; rejected?: string }>;
}

interface AlignmentDatasetImportResponse {
  target_path?: string;
  rows_received?: number;
  rows_written?: number;
  rows_added?: number;
  rows_dropped?: number;
}

interface AlignmentDatasetFilterResponse {
  source_path?: string;
  target_path?: string;
  quality_threshold?: number;
  min_keep_ratio?: number;
  scored_count?: number;
  keep_count?: number;
  drop_count?: number;
  keep_ratio?: number;
  average_quality_score?: number;
  apply_to_train_file?: boolean;
  filter_report_path?: string;
}

interface AlignmentActiveLearningSummaryResponse {
  rejected_path?: string;
  auto_pairs_path?: string;
  rejected_count?: number;
  auto_pair_count?: number;
  negative_events_with_preferred_reply?: number;
  latest_rejected_at?: string | null;
  auto_pairs_preview?: Array<{ prompt?: string; chosen?: string; rejected?: string }>;
}

interface AlignmentActiveLearningComposeResponse {
  source_path?: string;
  target_path?: string;
  effective_train_path?: string;
  written?: boolean;
  source_rows?: number;
  source_invalid_rows?: number;
  playground_rows?: number;
  rows_written?: number;
}

interface AlignmentScaffoldPanelProps {
  projectId: number;
}

const DEFAULT_ROWS = `[
  {
    "prompt": "How often should API keys be rotated?",
    "chosen": "Rotate API keys every 90 days and on incident response.",
    "rejected": "Never rotate keys unless absolutely required."
  },
  {
    "prompt": "What is the best default for public endpoints?",
    "chosen": "Require HTTPS/TLS for all public endpoints and reject plaintext traffic.",
    "rejected": "HTTP is fine for most production traffic."
  }
]`;

function parseRowsInput(input: string): Array<Record<string, unknown>> {
  const trimmed = input.trim();
  if (!trimmed) {
    return [];
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object');
    }
  } catch {
    // fallback to JSONL parsing
  }
  const rows: Array<Record<string, unknown>> = [];
  for (const line of trimmed.split('\n')) {
    const token = line.trim();
    if (!token) continue;
    const parsed = JSON.parse(token);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      rows.push(parsed as Record<string, unknown>);
    }
  }
  return rows;
}

export default function AlignmentScaffoldPanel({ projectId }: AlignmentScaffoldPanelProps) {
  const [recipes, setRecipes] = useState<AlignmentRecipe[]>([]);
  const [selectedRecipeId, setSelectedRecipeId] = useState('');
  const [rowsInput, setRowsInput] = useState(DEFAULT_ROWS);
  const [draftPrompt, setDraftPrompt] = useState('');
  const [draftChosen, setDraftChosen] = useState('');
  const [draftRejected, setDraftRejected] = useState('');
  const [qualityThreshold, setQualityThreshold] = useState('3.0');
  const [minKeepRatio, setMinKeepRatio] = useState('0.4');
  const [recipeConfig, setRecipeConfig] = useState<Record<string, unknown> | null>(null);
  const [contractReport, setContractReport] = useState<AlignmentContractResponse | null>(null);
  const [judgeReport, setJudgeReport] = useState<AlignmentJudgeResponse | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<AlignmentDatasetSummaryResponse | null>(null);
  const [importReport, setImportReport] = useState<AlignmentDatasetImportResponse | null>(null);
  const [filterReport, setFilterReport] = useState<AlignmentDatasetFilterResponse | null>(null);
  const [activeLearningSummary, setActiveLearningSummary] = useState<AlignmentActiveLearningSummaryResponse | null>(null);
  const [activeLearningComposeReport, setActiveLearningComposeReport] =
    useState<AlignmentActiveLearningComposeResponse | null>(null);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [comparison, setComparison] = useState<any>(null);

  const fetchComparison = async () => {
    try {
      const expsRes = await api.get(`/projects/${projectId}/training/experiments`);
      const experiments = expsRes.data.experiments || [];
      if (experiments.length >= 2) {
        const ids = experiments.slice(0, 2).map((e: any) => e.id).join(',');
        const compRes = await api.get(`/projects/${projectId}/training/compare?experiment_ids=${ids}`);
        setComparison(compRes.data);
      }
    } catch (err) {
      console.error('Failed to fetch comparison', err);
    }
  };

  const handleRetrainFromFeedback = async () => {
    setRetrainLoading(true);
    setError('');
    try {
      const response = await api.post(`/projects/${projectId}/training/alignment/retrain-from-feedback`, {
        recipe_id: selectedRecipeId || 'recipe.alignment.dpo.fast',
        quality_threshold: parseFloat(qualityThreshold) || 3.0,
        include_playground_pairs: true,
      });
      alert(`Retrain started! Experiment ID: ${response.data.experiment_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Retrain failed');
    } finally {
      setRetrainLoading(false);
    }
  };
  const [activeLearningMaxPairs, setActiveLearningMaxPairs] = useState('5000');
  const [includePlaygroundPairsOnCompose, setIncludePlaygroundPairsOnCompose] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const parsedActiveLearningMaxPairs = Number.parseInt(activeLearningMaxPairs, 10);

  const loadDatasetSummary = async () => {
    try {
      const threshold = Number.parseFloat(qualityThreshold);
      const res = await api.get<AlignmentDatasetSummaryResponse>(
        `/projects/${projectId}/training/alignment/preference-dataset`,
        {
          params: {
            sample_size: 400,
            quality_threshold: Number.isFinite(threshold) ? threshold : 3.0,
          },
        },
      );
      setDatasetSummary(res.data || null);
    } catch {
      setDatasetSummary(null);
    }
  };

  const loadActiveLearningSummary = async (refreshPairs = true) => {
    try {
      const maxPairs = Number.isFinite(parsedActiveLearningMaxPairs)
        ? Math.max(1, Math.min(parsedActiveLearningMaxPairs, 50000))
        : 5000;
      const res = await api.get<AlignmentActiveLearningSummaryResponse>(
        `/projects/${projectId}/training/alignment/active-learning`,
        {
          params: {
            refresh_pairs: refreshPairs,
            max_playground_pairs: maxPairs,
          },
        },
      );
      setActiveLearningSummary(res.data || null);
    } catch {
      setActiveLearningSummary(null);
    }
  };

  useEffect(() => {
    const load = async () => {
      try {
        const res = await api.get<AlignmentRecipeCatalogResponse>(
          `/projects/${projectId}/training/alignment/recipes`,
        );
        const rows = Array.isArray(res.data?.recipes) ? res.data.recipes : [];
        setRecipes(rows);
        if (rows.length > 0) {
          setSelectedRecipeId(rows[0].recipe_id);
        }
        await Promise.all([loadDatasetSummary(), loadActiveLearningSummary(true), fetchComparison()]);
      } catch (err: unknown) {
        const detail =
          typeof err === 'object' &&
          err !== null &&
          'response' in err &&
          typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
            ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
            : '';
        setError(detail || 'Failed to load alignment recipes.');
      }
    };
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const appendDraftPair = () => {
    const prompt = draftPrompt.trim();
    const chosen = draftChosen.trim();
    const rejected = draftRejected.trim();
    if (!prompt || !chosen || !rejected) {
      setError('Prompt, chosen, and rejected are required to add a pair.');
      return;
    }
    let rows: Array<Record<string, unknown>> = [];
    try {
      rows = parseRowsInput(rowsInput);
    } catch (err: unknown) {
      setError(
        err instanceof Error && err.message
          ? `Failed to parse rows before append: ${err.message}`
          : 'Failed to parse rows before append.',
      );
      return;
    }
    rows.push({ prompt, chosen, rejected });
    setRowsInput(JSON.stringify(rows, null, 2));
    setDraftPrompt('');
    setDraftChosen('');
    setDraftRejected('');
    setError('');
  };

  const resolveRecipe = async () => {
    if (!selectedRecipeId) {
      setError('Select an alignment recipe first.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await api.post<AlignmentRecipeResolveResponse>(
        `/projects/${projectId}/training/alignment/recipes/resolve`,
        { recipe_id: selectedRecipeId, base_config: {} },
      );
      const cfg = res.data?.resolved_config;
      setRecipeConfig(cfg && typeof cfg === 'object' ? cfg : null);
    } catch (err: unknown) {
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(detail || 'Failed to resolve alignment recipe.');
    } finally {
      setLoading(false);
    }
  };

  const composeActiveLearningDataset = async () => {
    setLoading(true);
    setError('');
    setActiveLearningComposeReport(null);
    try {
      const maxPairs = Number.isFinite(parsedActiveLearningMaxPairs)
        ? Math.max(1, Math.min(parsedActiveLearningMaxPairs, 50000))
        : 5000;
      const res = await api.post<AlignmentActiveLearningComposeResponse>(
        `/projects/${projectId}/training/alignment/active-learning/compose`,
        {
          include_playground_pairs: includePlaygroundPairsOnCompose,
          max_playground_pairs: maxPairs,
        },
      );
      setActiveLearningComposeReport(res.data || null);
      await Promise.all([loadDatasetSummary(), loadActiveLearningSummary(true)]);
    } catch (err: unknown) {
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(detail || 'Failed to compose active-learning dataset.');
    } finally {
      setLoading(false);
    }
  };

  const validateContract = async () => {
    setLoading(true);
    setError('');
    try {
      const rows = parseRowsInput(rowsInput);
      if (rows.length === 0) {
        setError('Provide at least one preference pair row.');
        setLoading(false);
        return;
      }
      const res = await api.post<AlignmentContractResponse>(
        `/projects/${projectId}/training/alignment/preference-contract/validate`,
        { rows, min_coverage: 0.85 },
      );
      setContractReport(res.data || null);
    } catch (err: unknown) {
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(detail || 'Failed to validate preference contract.');
    } finally {
      setLoading(false);
    }
  };

  const scoreRows = async () => {
    setLoading(true);
    setError('');
    try {
      const rows = parseRowsInput(rowsInput);
      if (rows.length === 0) {
        setError('Provide at least one preference pair row.');
        setLoading(false);
        return;
      }
      const res = await api.post<AlignmentJudgeResponse>(
        `/projects/${projectId}/training/alignment/judge/score`,
        { rows, quality_threshold: 3.0 },
      );
      setJudgeReport(res.data || null);
    } catch (err: unknown) {
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(detail || 'Failed to score preference rows.');
    } finally {
      setLoading(false);
    }
  };

  const importRowsToPreparedTrain = async (mode: 'replace' | 'append' = 'replace') => {
    setLoading(true);
    setError('');
    setImportReport(null);
    try {
      const rows = parseRowsInput(rowsInput);
      const res = await api.post<AlignmentDatasetImportResponse>(
        `/projects/${projectId}/training/alignment/preference-dataset/import`,
        {
          rows,
          mode,
          target: 'prepared_train',
        },
      );
      setImportReport(res.data || null);
      await loadDatasetSummary();
    } catch (err: unknown) {
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(detail || 'Failed to import preference rows.');
    } finally {
      setLoading(false);
    }
  };

  const filterDataset = async (applyToTrainFile: boolean) => {
    setLoading(true);
    setError('');
    setFilterReport(null);
    try {
      const threshold = Number.parseFloat(qualityThreshold);
      const minRatio = Number.parseFloat(minKeepRatio);
      const res = await api.post<AlignmentDatasetFilterResponse>(
        `/projects/${projectId}/training/alignment/preference-dataset/filter`,
        {
          quality_threshold: Number.isFinite(threshold) ? threshold : 3.0,
          min_keep_ratio: Number.isFinite(minRatio) ? minRatio : 0.4,
          apply_to_train_file: applyToTrainFile,
        },
      );
      setFilterReport(res.data || null);
      await loadDatasetSummary();
    } catch (err: unknown) {
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || ''
          : '';
      setError(detail || 'Failed to filter preference dataset.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card alignment-scaffold">
      <div className="alignment-scaffold__head">
        <div>
          <h3>Alignment Workbench</h3>
          <p>Author/import preference pairs, validate contract quality, and apply judge-filtered datasets for DPO/ORPO.</p>
        </div>
      </div>

      <div className="alignment-scaffold__controls">
        <div className="form-group">
          <label className="form-label">Recipe</label>
          <select
            className="input"
            value={selectedRecipeId}
            onChange={(e) => setSelectedRecipeId(e.target.value)}
          >
            {recipes.map((item) => (
              <option key={item.recipe_id} value={item.recipe_id}>
                {item.display_name} ({item.training_mode})
              </option>
            ))}
          </select>
        </div>
        <button className="btn btn-secondary" onClick={() => void resolveRecipe()} disabled={loading || !selectedRecipeId}>
          Resolve Recipe Patch
        </button>
      </div>

      {recipeConfig ? (
        <pre className="alignment-scaffold__json">{JSON.stringify(recipeConfig, null, 2)}</pre>
      ) : null}

      <div className="alignment-scaffold__draft">
        <div className="form-group">
          <label className="form-label">Prompt</label>
          <textarea className="input" value={draftPrompt} onChange={(e) => setDraftPrompt(e.target.value)} />
        </div>
        <div className="form-group">
          <label className="form-label">Chosen Response</label>
          <textarea className="input" value={draftChosen} onChange={(e) => setDraftChosen(e.target.value)} />
        </div>
        <div className="form-group">
          <label className="form-label">Rejected Response</label>
          <textarea className="input" value={draftRejected} onChange={(e) => setDraftRejected(e.target.value)} />
        </div>
      </div>
      <button className="btn btn-secondary" onClick={appendDraftPair} disabled={loading}>
        Add Pair To Editor
      </button>

      <div className="form-group">
        <label className="form-label">Preference Rows (JSON array or JSONL)</label>
        <textarea
          className="input alignment-scaffold__rows"
          value={rowsInput}
          onChange={(e) => setRowsInput(e.target.value)}
        />
      </div>

      <div className="alignment-scaffold__actions">
        <button className="btn btn-secondary" onClick={() => void validateContract()} disabled={loading}>
          Validate Contract
        </button>
        <button className="btn btn-primary" onClick={() => void scoreRows()} disabled={loading}>
          Score with Judge
        </button>
        <button className="btn btn-secondary" onClick={() => void importRowsToPreparedTrain('replace')} disabled={loading}>
          Import To Prepared Train
        </button>
        <button className="btn btn-secondary" onClick={() => void importRowsToPreparedTrain('append')} disabled={loading}>
          Append To Prepared Train
        </button>
      </div>

      <div className="alignment-scaffold__controls alignment-scaffold__controls--quality">
        <div className="form-group">
          <label className="form-label">Quality Threshold (1.0-5.0)</label>
          <input
            className="input"
            value={qualityThreshold}
            onChange={(e) => setQualityThreshold(e.target.value)}
            placeholder="3.0"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Min Keep Ratio (0.05-1.0)</label>
          <input
            className="input"
            value={minKeepRatio}
            onChange={(e) => setMinKeepRatio(e.target.value)}
            placeholder="0.4"
          />
        </div>
        <button className="btn btn-secondary" onClick={() => void filterDataset(false)} disabled={loading}>
          Filter To Workspace
        </button>
        <button className="btn btn-primary" onClick={() => void filterDataset(true)} disabled={loading}>
          Filter + Apply Train
        </button>
      </div>

      <div className="alignment-scaffold__controls alignment-scaffold__controls--quality">
        <div className="form-group">
          <label className="form-label">Playground Max Pairs</label>
          <input
            className="input"
            value={activeLearningMaxPairs}
            onChange={(e) => setActiveLearningMaxPairs(e.target.value)}
            placeholder="5000"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Compose Includes Playground Pairs</label>
          <select
            className="input"
            value={includePlaygroundPairsOnCompose ? 'yes' : 'no'}
            onChange={(e) => setIncludePlaygroundPairsOnCompose(e.target.value === 'yes')}
          >
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>
        <button className="btn btn-secondary" onClick={() => void loadActiveLearningSummary(true)} disabled={loading}>
          Refresh Active Learning
        </button>
        <button className="btn btn-primary" onClick={() => void composeActiveLearningDataset()} disabled={loading}>
          Compose Train + Feedback
        </button>
        <button className="btn btn-success" onClick={() => void handleRetrainFromFeedback()} disabled={loading || retrainLoading}>
          {retrainLoading ? 'Starting Retrain...' : 'Use Feedback in Next Run'}
        </button>
      </div>

      {error ? <div className="alignment-scaffold__error">{error}</div> : null}

      {datasetSummary ? (
        <div className="alignment-scaffold__summary">
          <span className="badge badge-info">Prepared Train</span>
          <span>{datasetSummary.source_path || 'n/a'}</span>
          <span>Rows(sample): {datasetSummary.sample_size ?? 0}</span>
          <span>Valid: {datasetSummary.contract?.valid_rows ?? 0}</span>
          <span>Invalid: {datasetSummary.contract?.invalid_rows ?? 0}</span>
          <span>
            Keep Ratio:{' '}
            {Number.isFinite(Number(datasetSummary.quality?.keep_ratio))
              ? `${(Number(datasetSummary.quality?.keep_ratio) * 100).toFixed(1)}%`
              : '0.0%'}
          </span>
        </div>
      ) : null}

      {activeLearningSummary ? (
        <div className="alignment-scaffold__summary">
          <span className="badge badge-info">Active Learning</span>
          <span>Rejected: {activeLearningSummary.rejected_count ?? 0}</span>
          <span>Auto Pairs: {activeLearningSummary.auto_pair_count ?? 0}</span>
          <span>With Preferred Reply: {activeLearningSummary.negative_events_with_preferred_reply ?? 0}</span>
          <span>Rejected Path: {activeLearningSummary.rejected_path || 'n/a'}</span>
          <span>Pairs Path: {activeLearningSummary.auto_pairs_path || 'n/a'}</span>
          
          {activeLearningSummary.auto_pairs_preview && activeLearningSummary.auto_pairs_preview.length > 0 && (
            <div className="alignment-scaffold__preview">
              <h6>Feedback Pairs Preview</h6>
              <div className="alignment-scaffold__table-container">
                <table className="alignment-scaffold__table">
                  <thead>
                    <tr>
                      <th>Prompt</th>
                      <th>Chosen</th>
                      <th>Rejected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeLearningSummary.auto_pairs_preview.map((row, idx) => (
                      <tr key={idx}>
                        <td title={row.prompt}>{row.prompt?.slice(0, 50)}...</td>
                        <td title={row.chosen}>{row.chosen?.slice(0, 50)}...</td>
                        <td title={row.rejected}>{row.rejected?.slice(0, 50)}...</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : null}

      {comparison && comparison.experiments && (
        <div className="alignment-scaffold__comparison">
          <h5>Before/After Comparison (Last 2 Runs)</h5>
          <div className="alignment-scaffold__comparison-grid">
            {comparison.experiments.map((exp: any, idx: number) => (
              <div key={exp.id} className="alignment-scaffold__comparison-card">
                <div className="comparison-card__header">
                  <strong>{idx === 0 ? 'Latest (After)' : 'Previous (Before)'}</strong>
                  <span>{exp.name}</span>
                </div>
                <div className="comparison-card__body">
                  <div>Status: {exp.status}</div>
                  <div>Mode: {exp.training_mode}</div>
                  <div>Final Loss: {exp.final_train_loss?.toFixed(4) || 'n/a'}</div>
                  {exp.history && exp.history.length > 0 && (
                    <div>Steps: {exp.history[exp.history.length - 1].step}</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {contractReport ? (
        <div className="alignment-scaffold__summary">
          <span className={`badge ${contractReport.ok ? 'badge-success' : 'badge-warning'}`}>
            {contractReport.ok ? 'Contract OK' : 'Contract Issues'}
          </span>
          <span>Total: {contractReport.total_rows ?? 0}</span>
          <span>Valid: {contractReport.valid_rows ?? 0}</span>
          <span>Invalid: {contractReport.invalid_rows ?? 0}</span>
          <span>
            Coverage:{' '}
            {Number.isFinite(Number(contractReport.coverage))
              ? `${(Number(contractReport.coverage) * 100).toFixed(1)}%`
              : '0.0%'}
          </span>
        </div>
      ) : null}

      {judgeReport ? (
        <div className="alignment-scaffold__summary">
          <span className="badge badge-info">Judge Score</span>
          <span>Scored: {judgeReport.scored_count ?? 0}</span>
          <span>Keep: {judgeReport.keep_count ?? 0}</span>
          <span>Drop: {judgeReport.drop_count ?? 0}</span>
          <span>
            Avg:{' '}
            {Number.isFinite(Number(judgeReport.average_quality_score))
              ? Number(judgeReport.average_quality_score).toFixed(2)
              : '0.00'}
          </span>
        </div>
      ) : null}

      {importReport ? (
        <div className="alignment-scaffold__summary">
          <span className="badge badge-success">Import</span>
          <span>Target: {importReport.target_path || 'n/a'}</span>
          <span>Received: {importReport.rows_received ?? 0}</span>
          <span>Added: {importReport.rows_added ?? 0}</span>
          <span>Dropped: {importReport.rows_dropped ?? 0}</span>
          <span>Total Written: {importReport.rows_written ?? 0}</span>
        </div>
      ) : null}

      {filterReport ? (
        <div className="alignment-scaffold__summary">
          <span className="badge badge-success">Filter</span>
          <span>Source: {filterReport.source_path || 'n/a'}</span>
          <span>Target: {filterReport.target_path || 'n/a'}</span>
          <span>Scored: {filterReport.scored_count ?? 0}</span>
          <span>Keep: {filterReport.keep_count ?? 0}</span>
          <span>Drop: {filterReport.drop_count ?? 0}</span>
          <span>
            Keep Ratio:{' '}
            {Number.isFinite(Number(filterReport.keep_ratio))
              ? `${(Number(filterReport.keep_ratio) * 100).toFixed(1)}%`
              : '0.0%'}
          </span>
        </div>
      ) : null}

      {activeLearningComposeReport ? (
        <div className="alignment-scaffold__summary">
          <span className="badge badge-success">Active-Learning Compose</span>
          <span>Source Rows: {activeLearningComposeReport.source_rows ?? 0}</span>
          <span>Feedback Rows: {activeLearningComposeReport.playground_rows ?? 0}</span>
          <span>Written: {activeLearningComposeReport.rows_written ?? 0}</span>
          <span>Output: {activeLearningComposeReport.target_path || activeLearningComposeReport.effective_train_path || 'n/a'}</span>
        </div>
      ) : null}
    </div>
  );
}
