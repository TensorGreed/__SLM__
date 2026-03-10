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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
        await loadDatasetSummary();
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
    </div>
  );
}
