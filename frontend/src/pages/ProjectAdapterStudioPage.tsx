import { useMemo, useState } from 'react';
import { useOutletContext } from 'react-router-dom';

import api from '../api/client';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectAdapterStudioPage.css';

type AnyObj = Record<string, any>;

const CANONICAL_FIELDS = [
  'text',
  'question',
  'answer',
  'source_text',
  'target_text',
  'label',
  'context',
  'messages',
  'prompt',
  'chosen',
  'rejected',
  'tool_name',
  'structured_output',
];

function parseJsonObject(text: string): AnyObj {
  const token = String(text || '').trim();
  if (!token) return {};
  const parsed = JSON.parse(token);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('JSON must be an object');
  }
  return parsed as AnyObj;
}

export default function ProjectAdapterStudioPage() {
  const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();

  const [sourceType, setSourceType] = useState('project_dataset');
  const [sourceRef, setSourceRef] = useState('');
  const [datasetType, setDatasetType] = useState('raw');
  const [sampleSize, setSampleSize] = useState('400');
  const [split, setSplit] = useState('');

  const [adapterId, setAdapterId] = useState('auto');
  const [taskProfile, setTaskProfile] = useState('');
  const [adapterConfigText, setAdapterConfigText] = useState('');

  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [adapterName, setAdapterName] = useState('studio-adapter');

  const [profileResult, setProfileResult] = useState<AnyObj | null>(null);
  const [inferResult, setInferResult] = useState<AnyObj | null>(null);
  const [previewResult, setPreviewResult] = useState<AnyObj | null>(null);
  const [validateResult, setValidateResult] = useState<AnyObj | null>(null);
  const [versionList, setVersionList] = useState<AnyObj[]>([]);
  const [selectedVersion, setSelectedVersion] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');

  const sourcePayload = useMemo(() => {
    const payload: AnyObj = { source_type: sourceType };
    if (sourceRef.trim()) payload.source_ref = sourceRef.trim();
    if (split.trim()) payload.split = split.trim();
    if (sourceType === 'project_dataset') payload.dataset_type = datasetType;
    return payload;
  }, [datasetType, sourceRef, sourceType, split]);

  const schemaFields = useMemo(() => {
    const fields = ((profileResult?.schema || inferResult?.profile?.schema || previewResult?.profile?.schema || {}) as AnyObj)
      .fields;
    return Array.isArray(fields) ? fields : [];
  }, [profileResult, inferResult, previewResult]);

  const sourceFieldOptions = useMemo(
    () => schemaFields.map((item: AnyObj) => String(item?.path || '')).filter(Boolean),
    [schemaFields],
  );

  const dropAnalysis = (previewResult?.drop_analysis || inferResult?.inference?.drop_analysis || {}) as AnyObj;
  const suggestions = (inferResult?.inference?.auto_fix_suggestions || previewResult?.preview?.auto_fix_suggestions || []) as AnyObj[];
  const conformance = (previewResult?.preview?.conformance_report || validateResult?.coverage?.conformance_report || {}) as AnyObj;
  const contract = (inferResult?.inference?.adapter_contract || previewResult?.preview?.adapter_contract || {}) as AnyObj;
  const previewRows = ((previewResult?.preview || inferResult)?.preview_rows || []) as AnyObj[];

  const refreshVersions = async () => {
    const resp = await api.get(`/projects/${projectId}/adapter-studio/adapters`, { params: { adapter_name: adapterName } });
    const items = Array.isArray(resp.data?.items) ? resp.data.items : [];
    setVersionList(items);
    if (items.length > 0) {
      setSelectedVersion(String(items[0].version));
    }
  };

  const handleProfile = async () => {
    setError('');
    setStatus('');
    setIsLoading(true);
    try {
      const resp = await api.post(`/projects/${projectId}/adapter-studio/profile`, {
        source: sourcePayload,
        sample_size: Number(sampleSize) || 400,
      });
      setProfileResult(resp.data);
      setStatus('Schema profile generated.');
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || 'Failed to profile dataset.'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleInfer = async () => {
    setError('');
    setStatus('');
    setIsLoading(true);
    try {
      const resp = await api.post(`/projects/${projectId}/adapter-studio/infer`, {
        source: sourcePayload,
        sample_size: Number(sampleSize) || 400,
        task_profile: taskProfile.trim() || null,
      });
      const data = resp.data || {};
      setInferResult(data);
      setProfileResult(data?.profile || null);
      setPreviewResult(null);
      setValidateResult(null);
      const inferredMapping = (data?.inference?.mapping_canvas || {}) as Record<string, string>;
      setMapping(inferredMapping);
      const resolvedAdapter = String(data?.inference?.resolved_adapter_id || '').trim();
      const resolvedTaskProfile = String(data?.inference?.resolved_task_profile || '').trim();
      if (resolvedAdapter) setAdapterId(resolvedAdapter);
      if (resolvedTaskProfile) setTaskProfile(resolvedTaskProfile);
      setStatus('Adapter inference completed. Review mapping and run preview/validate.');
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || 'Failed to infer adapter.'));
    } finally {
      setIsLoading(false);
    }
  };

  const handlePreview = async () => {
    setError('');
    setStatus('');
    setIsLoading(true);
    try {
      const resp = await api.post(`/projects/${projectId}/adapter-studio/preview`, {
        source: sourcePayload,
        adapter_id: adapterId.trim() || 'auto',
        field_mapping: mapping,
        adapter_config: parseJsonObject(adapterConfigText),
        task_profile: taskProfile.trim() || null,
        sample_size: Number(sampleSize) || 300,
        preview_limit: 25,
      });
      setPreviewResult(resp.data || null);
      setValidateResult(null);
      setStatus('Transformed row preview generated.');
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || 'Failed to preview adapter transform.'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleValidate = async () => {
    setError('');
    setStatus('');
    setIsLoading(true);
    try {
      const resp = await api.post(`/projects/${projectId}/adapter-studio/validate`, {
        source: sourcePayload,
        adapter_id: adapterId.trim() || 'auto',
        field_mapping: mapping,
        adapter_config: parseJsonObject(adapterConfigText),
        task_profile: taskProfile.trim() || null,
        sample_size: Number(sampleSize) || 300,
        preview_limit: 25,
      });
      setValidateResult(resp.data || null);
      setStatus('Adapter coverage validation completed.');
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || 'Failed to validate adapter.'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveVersion = async () => {
    setError('');
    setStatus('');
    setIsLoading(true);
    try {
      const resp = await api.post(`/projects/${projectId}/adapter-studio/adapters`, {
        adapter_name: adapterName.trim(),
        source_type: sourceType,
        source_ref: sourceRef.trim() || null,
        base_adapter_id: adapterId.trim() || 'auto',
        task_profile: taskProfile.trim() || null,
        field_mapping: mapping,
        adapter_config: parseJsonObject(adapterConfigText),
        output_contract: contract.output_contract || {},
        schema_profile: profileResult || inferResult?.profile || previewResult?.profile || {},
        inference_summary: inferResult?.inference || {},
        validation_report: validateResult || {},
      });
      setStatus(`Saved adapter version: v${resp.data?.version}`);
      await refreshVersions();
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || 'Failed to save adapter version.'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = async () => {
    setError('');
    setStatus('');
    const version = Number(selectedVersion) || 0;
    if (version <= 0) {
      setError('Choose a saved adapter version before export.');
      return;
    }
    setIsLoading(true);
    try {
      const resp = await api.post(
        `/projects/${projectId}/adapter-studio/adapters/${encodeURIComponent(adapterName)}/versions/${version}/export`,
        {},
      );
      const files = resp.data?.written_files || {};
      setStatus(`Export complete: ${files?.template_json || ''} ${files?.plugin_python || ''}`.trim());
    } catch (err: any) {
      setError(String(err?.response?.data?.detail || err?.message || 'Failed to export adapter scaffold.'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="workspace-page adapter-studio-page">
      <section className="workspace-page-header">
        <div>
          <h2 className="workspace-page-title">Dataset Structure Explorer and Adapter Studio</h2>
          <p className="workspace-page-subtitle">
            Profile schemas, map fields visually, preview transformed rows, and version reusable adapters without writing Python.
          </p>
        </div>
      </section>

      <section className="card studio-source-panel">
        <h3>Source</h3>
        <div className="studio-grid">
          <label>Source Type
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
              <option value="project_dataset">project_dataset</option>
              <option value="csv">csv</option>
              <option value="tsv">tsv</option>
              <option value="json">json</option>
              <option value="jsonl">jsonl</option>
              <option value="parquet">parquet</option>
              <option value="huggingface">huggingface</option>
              <option value="sql_snapshot">sql_snapshot</option>
              <option value="document_corpus">document_corpus</option>
              <option value="chunk_corpus">chunk_corpus</option>
              <option value="chat_transcripts">chat_transcripts</option>
              <option value="pairwise_preference">pairwise_preference</option>
            </select>
          </label>
          <label>Source Ref
            <input value={sourceRef} onChange={(e) => setSourceRef(e.target.value)} placeholder="/path/to/file or hf dataset id" />
          </label>
          <label>Project Dataset Type
            <select value={datasetType} onChange={(e) => setDatasetType(e.target.value)}>
              <option value="raw">raw</option>
              <option value="cleaned">cleaned</option>
              <option value="gold_dev">gold_dev</option>
              <option value="synthetic">synthetic</option>
              <option value="train">train</option>
              <option value="validation">validation</option>
              <option value="test">test</option>
            </select>
          </label>
          <label>Split
            <input value={split} onChange={(e) => setSplit(e.target.value)} placeholder="train" />
          </label>
          <label>Sample Size
            <input value={sampleSize} onChange={(e) => setSampleSize(e.target.value)} placeholder="400" />
          </label>
        </div>
        <div className="studio-actions">
          <button className="btn btn-secondary" onClick={() => void handleProfile()} disabled={isLoading}>Profile Dataset</button>
          <button className="btn btn-primary" onClick={() => void handleInfer()} disabled={isLoading}>Infer Adapter</button>
        </div>
      </section>

      {error && <div className="alert alert-error">{error}</div>}
      {status && <div className="alert alert-info">{status}</div>}

      <section className="card studio-panel">
        <h3>Schema Explorer</h3>
        <div className="studio-table-wrap">
          <table className="studio-table">
            <thead>
              <tr>
                <th>Field</th>
                <th>Type</th>
                <th>Null Rate</th>
                <th>Sensitive</th>
              </tr>
            </thead>
            <tbody>
              {schemaFields.map((row: AnyObj) => (
                <tr key={String(row.path)}>
                  <td>{String(row.path || '')}</td>
                  <td>{String(row.inferred_type || 'unknown')}</td>
                  <td>{Number(row.null_rate || 0).toFixed(3)}</td>
                  <td>{row.sensitive ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card studio-panel">
        <h3>Field Mapping Canvas</h3>
        <div className="studio-grid mapping-grid">
          {CANONICAL_FIELDS.map((canonical) => (
            <label key={canonical}>{canonical}
              <select
                value={mapping[canonical] || ''}
                onChange={(e) => setMapping((prev) => ({ ...prev, [canonical]: e.target.value }))}
              >
                <option value="">(unmapped)</option>
                {sourceFieldOptions.map((field) => (
                  <option key={`${canonical}-${field}`} value={field}>{field}</option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </section>

      <section className="card studio-panel">
        <h3>Task and Output Contract Alignment</h3>
        <div className="studio-kv-grid">
          <div><strong>Resolved Adapter:</strong> {String(inferResult?.inference?.resolved_adapter_id || previewResult?.preview?.resolved_adapter_id || adapterId || 'n/a')}</div>
          <div><strong>Task Profiles:</strong> {Array.isArray(contract.task_profiles) ? contract.task_profiles.join(', ') : 'n/a'}</div>
          <div><strong>Preferred Training Tasks:</strong> {Array.isArray(contract.preferred_training_tasks) ? contract.preferred_training_tasks.join(', ') : 'n/a'}</div>
          <div><strong>Required Output Fields:</strong> {Array.isArray(contract.output_contract?.required_fields) ? contract.output_contract.required_fields.join(', ') : 'n/a'}</div>
        </div>
        <label className="studio-wide">Adapter Config JSON
          <textarea value={adapterConfigText} onChange={(e) => setAdapterConfigText(e.target.value)} placeholder='{"field_mapping":{"text":"body"}}' />
        </label>
        <div className="studio-inline">
          <label>Adapter ID
            <input value={adapterId} onChange={(e) => setAdapterId(e.target.value)} placeholder="auto" />
          </label>
          <label>Task Profile
            <input value={taskProfile} onChange={(e) => setTaskProfile(e.target.value)} placeholder="qa, chat_sft, preference..." />
          </label>
        </div>
        <div className="studio-actions">
          <button className="btn btn-secondary" onClick={() => void handlePreview()} disabled={isLoading}>Preview Transform</button>
          <button className="btn btn-primary" onClick={() => void handleValidate()} disabled={isLoading}>Validate Coverage</button>
        </div>
      </section>

      <section className="card studio-panel">
        <h3>Drop/Error Analysis and Auto-fix Suggestions</h3>
        <div className="studio-kv-grid">
          <div><strong>Sampled:</strong> {dropAnalysis.sampled_records ?? 0}</div>
          <div><strong>Mapped:</strong> {dropAnalysis.mapped_records ?? 0}</div>
          <div><strong>Dropped:</strong> {dropAnalysis.dropped_records ?? 0}</div>
          <div><strong>Drop Rate:</strong> {Number(dropAnalysis.drop_rate || 0).toFixed(3)}</div>
        </div>
        <div className="studio-list">
          {(dropAnalysis.unmapped_fields || []).slice(0, 10).map((field: string) => (
            <div key={field} className="studio-chip">Unmapped: {field}</div>
          ))}
          {((validateResult?.coverage?.type_conflicts || []) as AnyObj[]).slice(0, 6).map((item: AnyObj, idx: number) => (
            <div key={`conflict-${idx}`} className="studio-chip studio-chip-warning">{String(item.message || 'Type conflict')}</div>
          ))}
        </div>
        <div className="studio-suggestion-list">
          {suggestions.slice(0, 8).map((item: AnyObj, idx: number) => (
            <div key={`suggestion-${idx}`} className="studio-suggestion">
              <div>{String(item.message || 'Suggestion')}</div>
              {item.suggested_field_mapping && (
                <button
                  className="btn btn-secondary"
                  onClick={() => setMapping((prev) => ({ ...prev, ...(item.suggested_field_mapping as Record<string, string>) }))}
                >
                  Apply Mapping
                </button>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="card studio-panel">
        <h3>Transformed Row Preview and Error Summary</h3>
        <div className="studio-table-wrap">
          <table className="studio-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Raw</th>
                <th>Mapped</th>
              </tr>
            </thead>
            <tbody>
              {previewRows.slice(0, 8).map((row: AnyObj, idx: number) => (
                <tr key={`preview-${idx}`}>
                  <td>{Number(row.index ?? idx)}</td>
                  <td><pre>{JSON.stringify(row.raw || {}, null, 2)}</pre></td>
                  <td><pre>{JSON.stringify(row.mapped || {}, null, 2)}</pre></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {validateResult && (
          <div className="studio-validation-summary">
            <strong>Validation Status:</strong> {String(validateResult.status || 'n/a')}<br />
            <strong>Reason Codes:</strong> {Array.isArray(validateResult.reason_codes) ? validateResult.reason_codes.join(', ') : 'n/a'}<br />
            <strong>Required Fields Below 100%:</strong> {Array.isArray(conformance.required_fields_below_100) ? conformance.required_fields_below_100.join(', ') : 'none'}
          </div>
        )}
      </section>

      <section className="card studio-panel">
        <h3>Version and Export</h3>
        <div className="studio-inline">
          <label>Adapter Name
            <input value={adapterName} onChange={(e) => setAdapterName(e.target.value)} />
          </label>
          <label>Saved Versions
            <select value={selectedVersion} onChange={(e) => setSelectedVersion(e.target.value)}>
              <option value="">(none)</option>
              {versionList.map((item: AnyObj) => (
                <option key={String(item.id)} value={String(item.version)}>v{String(item.version)}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="studio-actions">
          <button className="btn btn-primary" onClick={() => void handleSaveVersion()} disabled={isLoading}>Save Adapter Version</button>
          <button className="btn btn-secondary" onClick={() => void refreshVersions()} disabled={isLoading}>Refresh Versions</button>
          <button className="btn btn-secondary" onClick={() => void handleExport()} disabled={isLoading}>Export Scaffold</button>
        </div>
      </section>
    </div>
  );
}
