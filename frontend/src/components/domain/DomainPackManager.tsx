import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type { DomainPackResponse, DomainPackSummary, Project } from '../../types';
import './DomainPackManager.css';

interface DomainPackListResponse {
  packs: DomainPackSummary[];
  count: number;
}

interface DomainPackManagerProps {
  projectId: number;
  activeDomainPackId: number | null;
  onAssigned?: (project: Project) => void;
}

type EditorMode = 'create' | 'edit';

function buildPackTemplate(): Record<string, unknown> {
  return {
    $schema: 'slm.domain-pack/v1',
    pack_id: 'my-pack-v1',
    version: '1.0.0',
    display_name: 'My Domain Pack',
    description: 'Reusable pack defaults for a family of domain profiles.',
    owner: 'team',
    status: 'active',
    default_profile_id: 'generic-domain-v1',
    tags: ['general'],
    overlay: {
      dataset_split: {
        train: 0.8,
        val: 0.1,
        test: 0.1,
        seed: 42,
      },
      training_defaults: {
        training_mode: 'sft',
        chat_template: 'llama3',
        num_epochs: 3,
        batch_size: 4,
        learning_rate: 0.0002,
        use_lora: true,
      },
      registry_gates: {
        to_staging: { min_metrics: { f1: 0.65, llm_judge_pass_rate: 0.75 } },
        to_production: { min_metrics: { f1: 0.7, llm_judge_pass_rate: 0.8, safety_pass_rate: 0.92 } },
      },
    },
  };
}

function extractErrorMessage(error: unknown): string {
  if (typeof error === 'object' && error !== null) {
    const detail = (error as { response?: { data?: { detail?: string } } }).response?.data?.detail;
    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'Operation failed';
}

export default function DomainPackManager({
  projectId,
  activeDomainPackId,
  onAssigned,
}: DomainPackManagerProps) {
  const [packs, setPacks] = useState<DomainPackSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAssigning, setIsAssigning] = useState(false);
  const [selectedPackId, setSelectedPackId] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const [editorOpen, setEditorOpen] = useState(false);
  const [editorMode, setEditorMode] = useState<EditorMode>('create');
  const [editorTargetPackId, setEditorTargetPackId] = useState<string | null>(null);
  const [editorJson, setEditorJson] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const loadPacks = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage('');
    try {
      const res = await api.get<DomainPackListResponse>('/domain-packs');
      setPacks(res.data.packs || []);
    } catch (err) {
      setPacks([]);
      setErrorMessage(`Failed to load domain packs: ${extractErrorMessage(err)}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPacks();
  }, [loadPacks]);

  const activePack = useMemo(
    () => packs.find((item) => item.id === activeDomainPackId) || null,
    [packs, activeDomainPackId],
  );

  useEffect(() => {
    if (activePack) {
      setSelectedPackId(activePack.pack_id);
      return;
    }
    if (!selectedPackId && packs.length > 0) {
      setSelectedPackId(packs[0].pack_id);
    }
  }, [activePack, packs, selectedPackId]);

  const handleAssign = async () => {
    if (!selectedPackId) {
      return;
    }
    setIsAssigning(true);
    setStatusMessage('');
    setErrorMessage('');
    try {
      const res = await api.put<Project>(`/projects/${projectId}/domain-pack`, {
        pack_id: selectedPackId,
        adopt_pack_default_profile: true,
      });
      setStatusMessage(`Assigned pack: ${selectedPackId}`);
      onAssigned?.(res.data);
    } catch (err) {
      setErrorMessage(`Failed to assign pack: ${extractErrorMessage(err)}`);
    } finally {
      setIsAssigning(false);
    }
  };

  const openCreateEditor = () => {
    setEditorMode('create');
    setEditorTargetPackId(null);
    setEditorJson(JSON.stringify(buildPackTemplate(), null, 2));
    setEditorOpen(true);
    setErrorMessage('');
  };

  const openEditEditor = async () => {
    if (!selectedPackId) {
      setErrorMessage('Select a pack to edit.');
      return;
    }
    setErrorMessage('');
    try {
      const res = await api.get<DomainPackResponse>(`/domain-packs/${selectedPackId}`);
      setEditorMode('edit');
      setEditorTargetPackId(selectedPackId);
      setEditorJson(JSON.stringify(res.data.contract || {}, null, 2));
      setEditorOpen(true);
    } catch (err) {
      setErrorMessage(`Failed to load pack contract: ${extractErrorMessage(err)}`);
    }
  };

  const handleDuplicatePack = async () => {
    if (!selectedPackId) {
      setErrorMessage('Select a pack to duplicate.');
      return;
    }
    setErrorMessage('');
    setStatusMessage('');
    try {
      const res = await api.post<DomainPackResponse>(`/domain-packs/${selectedPackId}/duplicate`, {});
      const duplicated = res.data;
      setStatusMessage(`Duplicated pack as ${duplicated.pack_id}@${duplicated.version}`);
      setSelectedPackId(duplicated.pack_id);
      setEditorMode('edit');
      setEditorTargetPackId(duplicated.pack_id);
      setEditorJson(JSON.stringify(duplicated.contract || {}, null, 2));
      setEditorOpen(true);
      await loadPacks();
    } catch (err) {
      setErrorMessage(`Failed to duplicate pack: ${extractErrorMessage(err)}`);
    }
  };

  const handleSavePack = async () => {
    if (!editorJson.trim()) {
      setErrorMessage('Contract JSON cannot be empty.');
      return;
    }

    let payload: Record<string, unknown>;
    try {
      const parsed = JSON.parse(editorJson) as unknown;
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        setErrorMessage('Contract JSON must be an object.');
        return;
      }
      payload = parsed as Record<string, unknown>;
    } catch {
      setErrorMessage('Contract JSON is invalid.');
      return;
    }

    setIsSaving(true);
    setErrorMessage('');
    setStatusMessage('');
    try {
      let response: DomainPackResponse;
      if (editorMode === 'create') {
        response = (await api.post<DomainPackResponse>('/domain-packs', payload)).data;
      } else {
        const target = editorTargetPackId || selectedPackId;
        response = (await api.put<DomainPackResponse>(`/domain-packs/${target}`, payload)).data;
      }
      setStatusMessage(
        editorMode === 'create'
          ? `Created pack: ${response.pack_id}`
          : `Updated pack: ${response.pack_id}`,
      );
      setSelectedPackId(response.pack_id);
      setEditorOpen(false);
      await loadPacks();
    } catch (err) {
      setErrorMessage(`Failed to save pack: ${extractErrorMessage(err)}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="card domain-pack-card">
      <div className="domain-pack-header">
        <div>
          <h3>Domain Pack</h3>
          <p className="domain-pack-subtitle">
            Pack-level overlays and default profile selection with automatic fallback.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => void loadPacks()} disabled={isLoading}>
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="domain-pack-active">
        <span className="label">Active</span>
        {activePack ? (
          <span className="active-value">
            {activePack.display_name} ({activePack.pack_id}@{activePack.version})
          </span>
        ) : (
          <span className="active-value">No pack assigned</span>
        )}
      </div>

      <div className="domain-pack-controls">
        <select
          className="input"
          value={selectedPackId}
          onChange={(e) => setSelectedPackId(e.target.value)}
          disabled={packs.length === 0}
        >
          {packs.length === 0 ? (
            <option value="">No packs found</option>
          ) : (
            packs.map((pack) => (
              <option key={pack.pack_id} value={pack.pack_id}>
                {pack.display_name} ({pack.pack_id})
              </option>
            ))
          )}
        </select>
        <button
          className="btn btn-primary"
          onClick={() => void handleAssign()}
          disabled={isAssigning || !selectedPackId}
        >
          {isAssigning ? 'Assigning...' : 'Assign to Project'}
        </button>
        <button className="btn btn-secondary" onClick={openEditEditor} disabled={!selectedPackId}>
          View/Edit Contract
        </button>
        <button className="btn btn-secondary" onClick={() => void handleDuplicatePack()} disabled={!selectedPackId}>
          Duplicate + Open Editor
        </button>
        <button className="btn btn-secondary" onClick={openCreateEditor}>
          New Pack
        </button>
      </div>

      {statusMessage && <div className="domain-pack-status domain-pack-status--ok">{statusMessage}</div>}
      {errorMessage && <div className="domain-pack-status domain-pack-status--error">{errorMessage}</div>}

      <div className="domain-pack-list">
        {packs.map((pack) => (
          <div key={pack.id} className={`domain-pack-item ${pack.id === activeDomainPackId ? 'active' : ''}`}>
            <div className="identity">
              <strong>{pack.display_name}</strong>
              <span className="meta">
                {pack.pack_id}@{pack.version}
              </span>
              {pack.default_profile_id && (
                <span className="meta">
                  default profile: {pack.default_profile_id}
                </span>
              )}
            </div>
            <div className="badges">
              <span className={`badge ${pack.status === 'active' ? 'badge-success' : 'badge-warning'}`}>
                {pack.status}
              </span>
              {pack.is_system && <span className="badge badge-info">system</span>}
            </div>
          </div>
        ))}
      </div>

      {editorOpen && (
        <div className="modal-overlay" onClick={() => setEditorOpen(false)}>
          <div className="modal domain-pack-editor-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">
                {editorMode === 'create' ? 'Create Domain Pack' : `Edit ${editorTargetPackId}`}
              </h2>
              <button className="btn btn-ghost" onClick={() => setEditorOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Contract JSON</label>
                <textarea
                  className="domain-pack-json"
                  value={editorJson}
                  onChange={(e) => setEditorJson(e.target.value)}
                  spellCheck={false}
                />
                <div className="form-hint">
                  Required keys include <code>pack_id</code> and <code>display_name</code>.
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setEditorOpen(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={() => void handleSavePack()} disabled={isSaving}>
                {isSaving ? 'Saving...' : 'Save Pack'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
