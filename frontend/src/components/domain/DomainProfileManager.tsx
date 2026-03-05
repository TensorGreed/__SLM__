import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type { DomainProfileResponse, DomainProfileSummary, Project } from '../../types';
import './DomainProfileManager.css';

interface DomainProfileListResponse {
  profiles: DomainProfileSummary[];
  count: number;
}

interface DomainProfileManagerProps {
  projectId: number;
  activeDomainProfileId: number | null;
  onAssigned?: (project: Project) => void;
}

type EditorMode = 'create' | 'edit';

function buildProfileTemplate(): Record<string, unknown> {
  return {
    $schema: 'slm.domain-profile/v1',
    profile_id: 'my-domain-v1',
    version: '1.0.0',
    display_name: 'My Domain',
    description: 'Describe the target domain and task mix.',
    owner: 'team',
    status: 'active',
    tasks: [
      {
        task_id: 'qa',
        output_mode: 'text',
        required_fields: ['question', 'answer'],
        optional_fields: ['context', 'metadata'],
      },
    ],
    canonical_schema: {
      required: ['input_text', 'target_text'],
      aliases: {
        input_text: ['question', 'prompt', 'input'],
        target_text: ['answer', 'output', 'completion'],
      },
    },
    normalization: {
      trim_whitespace: true,
      drop_empty_records: true,
      dedupe: { enabled: true, method: 'hash(input_text,target_text)' },
      pii_redaction: { enabled: false },
    },
    data_quality: {
      min_records: 1000,
      max_null_ratio: 0.1,
      max_duplicate_ratio: 0.2,
      required_coverage: { input_text: 0.99, target_text: 0.99 },
    },
    dataset_split: {
      train: 0.8,
      val: 0.1,
      test: 0.1,
      seed: 42,
      stratify_by: [],
      leakage_checks: ['exact_text_overlap'],
    },
    training_defaults: {
      training_mode: 'sft',
      chat_template: 'llama3',
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 0.0002,
      use_lora: true,
    },
    evaluation: {
      metrics: [
        { metric_id: 'f1', weight: 0.5, threshold: 0.7 },
        { metric_id: 'safety_pass_rate', weight: 0.5, threshold: 0.9 },
      ],
      required_metrics_for_promotion: ['f1', 'safety_pass_rate'],
    },
    tools: {
      retrieval: { enabled: false, adapter: null },
      function_calling: { enabled: false, adapter: null },
      required_secrets: [],
    },
    registry_gates: {
      to_staging: { min_metrics: { f1: 0.65 } },
      to_production: { min_metrics: { f1: 0.7, safety_pass_rate: 0.92 } },
    },
    audit: {
      require_human_approval_for_production: true,
      notes_required_on_force_promotion: true,
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

export default function DomainProfileManager({
  projectId,
  activeDomainProfileId,
  onAssigned,
}: DomainProfileManagerProps) {
  const [profiles, setProfiles] = useState<DomainProfileSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAssigning, setIsAssigning] = useState(false);
  const [selectedProfileId, setSelectedProfileId] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const [editorOpen, setEditorOpen] = useState(false);
  const [editorMode, setEditorMode] = useState<EditorMode>('create');
  const [editorTargetProfileId, setEditorTargetProfileId] = useState<string | null>(null);
  const [editorJson, setEditorJson] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const loadProfiles = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage('');
    try {
      const res = await api.get<DomainProfileListResponse>('/domain-profiles');
      setProfiles(res.data.profiles || []);
    } catch (err) {
      setProfiles([]);
      setErrorMessage(`Failed to load domain profiles: ${extractErrorMessage(err)}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadProfiles();
  }, [loadProfiles]);

  const activeProfile = useMemo(
    () => profiles.find((item) => item.id === activeDomainProfileId) || null,
    [profiles, activeDomainProfileId],
  );

  useEffect(() => {
    if (activeProfile) {
      setSelectedProfileId(activeProfile.profile_id);
      return;
    }
    if (!selectedProfileId && profiles.length > 0) {
      setSelectedProfileId(profiles[0].profile_id);
    }
  }, [activeProfile, profiles, selectedProfileId]);

  const handleAssign = async () => {
    if (!selectedProfileId) {
      return;
    }
    setIsAssigning(true);
    setStatusMessage('');
    setErrorMessage('');
    try {
      const res = await api.put<Project>(`/projects/${projectId}/domain-profile`, {
        profile_id: selectedProfileId,
      });
      setStatusMessage(`Assigned profile: ${selectedProfileId}`);
      onAssigned?.(res.data);
    } catch (err) {
      setErrorMessage(`Failed to assign profile: ${extractErrorMessage(err)}`);
    } finally {
      setIsAssigning(false);
    }
  };

  const openCreateEditor = () => {
    setEditorMode('create');
    setEditorTargetProfileId(null);
    setEditorJson(JSON.stringify(buildProfileTemplate(), null, 2));
    setEditorOpen(true);
    setErrorMessage('');
  };

  const openEditEditor = async () => {
    if (!selectedProfileId) {
      setErrorMessage('Select a profile to edit.');
      return;
    }
    setErrorMessage('');
    try {
      const res = await api.get<DomainProfileResponse>(`/domain-profiles/${selectedProfileId}`);
      setEditorMode('edit');
      setEditorTargetProfileId(selectedProfileId);
      setEditorJson(JSON.stringify(res.data.contract || {}, null, 2));
      setEditorOpen(true);
    } catch (err) {
      setErrorMessage(`Failed to load profile contract: ${extractErrorMessage(err)}`);
    }
  };

  const handleDuplicateProfile = async () => {
    if (!selectedProfileId) {
      setErrorMessage('Select a profile to duplicate.');
      return;
    }
    setErrorMessage('');
    setStatusMessage('');
    try {
      const res = await api.post<DomainProfileResponse>(
        `/domain-profiles/${selectedProfileId}/duplicate`,
        {},
      );
      const duplicated = res.data;
      setStatusMessage(`Duplicated profile as ${duplicated.profile_id}@${duplicated.version}`);
      setSelectedProfileId(duplicated.profile_id);
      setEditorMode('edit');
      setEditorTargetProfileId(duplicated.profile_id);
      setEditorJson(JSON.stringify(duplicated.contract || {}, null, 2));
      setEditorOpen(true);
      await loadProfiles();
    } catch (err) {
      setErrorMessage(`Failed to duplicate profile: ${extractErrorMessage(err)}`);
    }
  };

  const handleSaveProfile = async () => {
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
      let response: DomainProfileResponse;
      if (editorMode === 'create') {
        response = (await api.post<DomainProfileResponse>('/domain-profiles', payload)).data;
      } else {
        const target = editorTargetProfileId || selectedProfileId;
        response = (await api.put<DomainProfileResponse>(`/domain-profiles/${target}`, payload)).data;
      }
      setStatusMessage(
        editorMode === 'create'
          ? `Created profile: ${response.profile_id}`
          : `Updated profile: ${response.profile_id}`,
      );
      setSelectedProfileId(response.profile_id);
      setEditorOpen(false);
      await loadProfiles();
    } catch (err) {
      setErrorMessage(`Failed to save profile: ${extractErrorMessage(err)}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="card domain-profile-card">
      <div className="domain-profile-header">
        <div>
          <h3>Domain Profile</h3>
          <p className="domain-profile-subtitle">
            Assign contract defaults for dataset split, training config, and registry gates.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => void loadProfiles()} disabled={isLoading}>
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="domain-profile-active">
        <span className="label">Active</span>
        {activeProfile ? (
          <span className="active-value">
            {activeProfile.display_name} ({activeProfile.profile_id}@{activeProfile.version})
          </span>
        ) : (
          <span className="active-value">No profile assigned</span>
        )}
      </div>

      <div className="domain-profile-controls">
        <select
          className="input"
          value={selectedProfileId}
          onChange={(e) => setSelectedProfileId(e.target.value)}
          disabled={profiles.length === 0}
        >
          {profiles.length === 0 ? (
            <option value="">No profiles found</option>
          ) : (
            profiles.map((profile) => (
              <option key={profile.profile_id} value={profile.profile_id}>
                {profile.display_name} ({profile.profile_id})
              </option>
            ))
          )}
        </select>
        <button
          className="btn btn-primary"
          onClick={() => void handleAssign()}
          disabled={isAssigning || !selectedProfileId}
        >
          {isAssigning ? 'Assigning...' : 'Assign to Project'}
        </button>
        <button className="btn btn-secondary" onClick={openEditEditor} disabled={!selectedProfileId}>
          View/Edit Contract
        </button>
        <button className="btn btn-secondary" onClick={() => void handleDuplicateProfile()} disabled={!selectedProfileId}>
          Duplicate + Open Editor
        </button>
        <button className="btn btn-secondary" onClick={openCreateEditor}>
          New Profile
        </button>
      </div>

      {statusMessage && <div className="domain-profile-status domain-profile-status--ok">{statusMessage}</div>}
      {errorMessage && <div className="domain-profile-status domain-profile-status--error">{errorMessage}</div>}

      <div className="domain-profile-list">
        {profiles.map((profile) => (
          <div
            key={profile.id}
            className={`domain-profile-item ${profile.id === activeDomainProfileId ? 'active' : ''}`}
          >
            <div className="identity">
              <strong>{profile.display_name}</strong>
              <span className="meta">
                {profile.profile_id}@{profile.version}
              </span>
            </div>
            <div className="badges">
              <span className={`badge ${profile.status === 'active' ? 'badge-success' : 'badge-warning'}`}>
                {profile.status}
              </span>
              {profile.is_system && <span className="badge badge-info">system</span>}
            </div>
          </div>
        ))}
      </div>

      {editorOpen && (
        <div className="modal-overlay" onClick={() => setEditorOpen(false)}>
          <div className="modal domain-profile-editor-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">
                {editorMode === 'create' ? 'Create Domain Profile' : `Edit ${editorTargetProfileId}`}
              </h2>
              <button className="btn btn-ghost" onClick={() => setEditorOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Contract JSON</label>
                <textarea
                  className="domain-profile-json"
                  value={editorJson}
                  onChange={(e) => setEditorJson(e.target.value)}
                  spellCheck={false}
                />
                <div className="form-hint">
                  Required keys include <code>profile_id</code> and <code>display_name</code>.
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setEditorOpen(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={() => void handleSaveProfile()} disabled={isSaving}>
                {isSaving ? 'Saving...' : 'Save Profile'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
