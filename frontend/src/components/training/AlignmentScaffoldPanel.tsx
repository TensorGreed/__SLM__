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
  const [recipeConfig, setRecipeConfig] = useState<Record<string, unknown> | null>(null);
  const [contractReport, setContractReport] = useState<AlignmentContractResponse | null>(null);
  const [judgeReport, setJudgeReport] = useState<AlignmentJudgeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
  }, [projectId]);

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

  return (
    <div className="card alignment-scaffold">
      <div className="alignment-scaffold__head">
        <div>
          <h3>Alignment Workbench (Phase 3 Scaffold)</h3>
          <p>Validate DPO/ORPO pair contracts and run judge-quality scoring before alignment training.</p>
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
      </div>

      {error ? <div className="alignment-scaffold__error">{error}</div> : null}

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
    </div>
  );
}
