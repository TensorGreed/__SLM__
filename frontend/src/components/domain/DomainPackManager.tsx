import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type { DomainPackResponse, DomainPackSummary, Project } from '../../types';
import './DomainPackManager.css';

interface DomainPackListResponse {
  packs: DomainPackSummary[];
  count: number;
}

type HookKind = 'normalizers' | 'validators' | 'evaluators';
type ContractHookKey = 'normalizer' | 'validator' | 'evaluator';

interface HookCatalogResponse {
  normalizers: Record<string, string>;
  validators: Record<string, string>;
  evaluators: Record<string, string>;
  plugin_modules_loaded?: string[];
  plugin_load_errors?: Record<string, string>;
  plugin_hook_sources?: Partial<Record<HookKind, Record<string, string>>>;
}

interface HookReloadResponse {
  reload: {
    requested_modules: string[];
    loaded_modules: string[];
    skipped_modules: string[];
    errors: Record<string, string>;
  };
  catalog: HookCatalogResponse;
}

interface DomainPackManagerProps {
  projectId: number;
  activeDomainPackId: number | null;
  onAssigned?: (project: Project) => void;
}

type EditorMode = 'create' | 'edit';

const HOOK_KIND_LABELS: Record<HookKind, string> = {
  normalizers: 'Normalizers',
  validators: 'Validators',
  evaluators: 'Evaluators',
};

const CONTRACT_KEY_TO_HOOK_KIND: Record<ContractHookKey, HookKind> = {
  normalizer: 'normalizers',
  validator: 'validators',
  evaluator: 'evaluators',
};

const CONTRACT_HOOK_LABELS: Record<ContractHookKey, string> = {
  normalizer: 'Normalizer Hook',
  validator: 'Validator Hook',
  evaluator: 'Evaluator Hook',
};

const DEFAULT_HOOK_SELECTION: Record<ContractHookKey, string> = {
  normalizer: 'default-normalizer',
  validator: 'default-validator',
  evaluator: 'default-evaluator',
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeHookId(value: unknown): string {
  if (typeof value !== 'string') {
    return '';
  }
  return value.trim().toLowerCase().replace(/_/g, '-').replace(/\s+/g, '-');
}

function extractHookSelectionFromContract(contract: Record<string, unknown>): Record<ContractHookKey, string> {
  const hooks = isRecord(contract.hooks) ? contract.hooks : {};
  const selection = { ...DEFAULT_HOOK_SELECTION };
  (Object.keys(DEFAULT_HOOK_SELECTION) as ContractHookKey[]).forEach((key) => {
    const spec = hooks[key];
    if (!isRecord(spec)) {
      return;
    }
    const rawId = spec.id ?? spec.hook_id;
    const normalizedId = normalizeHookId(rawId);
    if (normalizedId) {
      selection[key] = normalizedId;
    }
  });
  return selection;
}

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
    hooks: {
      normalizer: { id: 'default-normalizer', config: {} },
      validator: { id: 'default-validator', config: {} },
      evaluator: { id: 'default-evaluator', config: {} },
    },
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
  const [editorHookSelection, setEditorHookSelection] = useState<Record<ContractHookKey, string>>({
    ...DEFAULT_HOOK_SELECTION,
  });
  const [editorHookStatus, setEditorHookStatus] = useState('');
  const [editorHookError, setEditorHookError] = useState('');
  const [hookCatalog, setHookCatalog] = useState<HookCatalogResponse | null>(null);
  const [isHookCatalogLoading, setIsHookCatalogLoading] = useState(false);
  const [isHookReloading, setIsHookReloading] = useState(false);
  const [hookStatusMessage, setHookStatusMessage] = useState('');
  const [hookErrorMessage, setHookErrorMessage] = useState('');

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

  const loadHookCatalog = useCallback(async () => {
    setIsHookCatalogLoading(true);
    try {
      const res = await api.get<HookCatalogResponse>('/domain-packs/hooks/catalog');
      setHookCatalog(res.data);
      setHookErrorMessage('');
    } catch (err) {
      setHookCatalog(null);
      setHookErrorMessage(`Failed to load hook catalog: ${extractErrorMessage(err)}`);
    } finally {
      setIsHookCatalogLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHookCatalog();
  }, [loadHookCatalog]);

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
    const template = buildPackTemplate();
    setEditorMode('create');
    setEditorTargetPackId(null);
    setEditorJson(JSON.stringify(template, null, 2));
    setEditorHookSelection(extractHookSelectionFromContract(template));
    setEditorHookStatus('');
    setEditorHookError('');
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
      const contractPayload = isRecord(res.data.contract) ? res.data.contract : {};
      setEditorMode('edit');
      setEditorTargetPackId(selectedPackId);
      setEditorJson(JSON.stringify(contractPayload, null, 2));
      setEditorHookSelection(extractHookSelectionFromContract(contractPayload));
      setEditorHookStatus('');
      setEditorHookError('');
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
      const duplicatedContract = isRecord(duplicated.contract) ? duplicated.contract : {};
      setStatusMessage(`Duplicated pack as ${duplicated.pack_id}@${duplicated.version}`);
      setSelectedPackId(duplicated.pack_id);
      setEditorMode('edit');
      setEditorTargetPackId(duplicated.pack_id);
      setEditorJson(JSON.stringify(duplicatedContract, null, 2));
      setEditorHookSelection(extractHookSelectionFromContract(duplicatedContract));
      setEditorHookStatus('');
      setEditorHookError('');
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

  const handleRefreshAll = async () => {
    await Promise.all([loadPacks(), loadHookCatalog()]);
  };

  const parseEditorJsonForHookHelper = (): Record<string, unknown> | null => {
    if (!editorJson.trim()) {
      setEditorHookError('Editor JSON is empty.');
      setEditorHookStatus('');
      return null;
    }
    try {
      const parsed = JSON.parse(editorJson) as unknown;
      if (!isRecord(parsed)) {
        setEditorHookError('Editor JSON must be an object.');
        setEditorHookStatus('');
        return null;
      }
      return parsed;
    } catch {
      setEditorHookError('Editor JSON is invalid. Fix JSON before using hook helper.');
      setEditorHookStatus('');
      return null;
    }
  };

  const handleLoadHookSelectionFromJson = () => {
    const parsed = parseEditorJsonForHookHelper();
    if (!parsed) {
      return;
    }
    setEditorHookSelection(extractHookSelectionFromContract(parsed));
    setEditorHookError('');
    setEditorHookStatus('Loaded hook IDs from current JSON.');
  };

  const handleApplyHookSelectionToJson = () => {
    const parsed = parseEditorJsonForHookHelper();
    if (!parsed) {
      return;
    }
    const nextContract: Record<string, unknown> = { ...parsed };
    const hooks = isRecord(nextContract.hooks) ? { ...nextContract.hooks } : {};

    (Object.keys(editorHookSelection) as ContractHookKey[]).forEach((key) => {
      const selected = normalizeHookId(editorHookSelection[key]) || DEFAULT_HOOK_SELECTION[key];
      const existing = hooks[key];
      const nextSpec = isRecord(existing) ? { ...existing } : {};
      nextSpec.id = selected;
      if (!isRecord(nextSpec.config)) {
        nextSpec.config = {};
      }
      hooks[key] = nextSpec;
    });

    nextContract.hooks = hooks;
    setEditorJson(JSON.stringify(nextContract, null, 2));
    setEditorHookError('');
    setEditorHookStatus('Applied selected hooks into contract JSON.');
  };

  const handleReloadHooks = async () => {
    setIsHookReloading(true);
    setHookStatusMessage('');
    setHookErrorMessage('');
    try {
      const res = await api.post<HookReloadResponse>('/domain-packs/hooks/reload', {});
      const reload = res.data.reload || {
        requested_modules: [],
        loaded_modules: [],
        skipped_modules: [],
        errors: {},
      };
      const catalog = res.data.catalog || null;
      setHookCatalog(catalog);

      const requestedCount = (reload.requested_modules || []).length;
      const loadedCount = (reload.loaded_modules || []).length;
      const skippedCount = (reload.skipped_modules || []).length;
      const errorCount = Object.keys(reload.errors || {}).length;

      if (errorCount > 0) {
        setHookErrorMessage(
          `Hook reload completed with ${errorCount} error(s). See plugin load errors below for details.`,
        );
      } else if (requestedCount === 0) {
        setHookStatusMessage('Hook reload complete. No plugin modules configured.');
      } else {
        setHookStatusMessage(
          `Hook reload complete. Loaded ${loadedCount} module(s), skipped ${skippedCount} module(s).`,
        );
      }
    } catch (err) {
      setHookErrorMessage(`Failed to reload hook plugins: ${extractErrorMessage(err)}`);
    } finally {
      setIsHookReloading(false);
    }
  };

  const pluginModules = hookCatalog?.plugin_modules_loaded || [];
  const pluginErrors = Object.entries(hookCatalog?.plugin_load_errors || {});
  const hookOptionsByKind = useMemo(() => {
    return {
      normalizers: Object.entries(hookCatalog?.normalizers || {}),
      validators: Object.entries(hookCatalog?.validators || {}),
      evaluators: Object.entries(hookCatalog?.evaluators || {}),
    };
  }, [hookCatalog]);

  return (
    <div className="card domain-pack-card">
      <div className="domain-pack-header">
        <div>
          <h3>Domain Pack</h3>
          <p className="domain-pack-subtitle">
            Pack-level overlays and default profile selection with automatic fallback.
          </p>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => void handleRefreshAll()}
          disabled={isLoading || isHookCatalogLoading}
        >
          {isLoading || isHookCatalogLoading ? 'Refreshing...' : 'Refresh'}
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

      <div className="domain-pack-hooks-panel">
        <div className="domain-pack-hooks-header">
          <div>
            <h4>Hook Catalog</h4>
            <p className="domain-pack-hooks-subtitle">
              Use these IDs in <code>hooks.normalizer.id</code>, <code>hooks.validator.id</code>, and{' '}
              <code>hooks.evaluator.id</code> inside pack contracts.
            </p>
          </div>
          <div className="domain-pack-hooks-actions">
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => void loadHookCatalog()}
              disabled={isHookCatalogLoading}
            >
              {isHookCatalogLoading ? 'Loading...' : 'Refresh Catalog'}
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => void handleReloadHooks()}
              disabled={isHookReloading}
            >
              {isHookReloading ? 'Reloading...' : 'Reload Plugins'}
            </button>
          </div>
        </div>

        {hookStatusMessage && <div className="domain-pack-status domain-pack-status--ok">{hookStatusMessage}</div>}
        {hookErrorMessage && <div className="domain-pack-status domain-pack-status--error">{hookErrorMessage}</div>}

        <div className="domain-pack-hooks-grid">
          {(['normalizers', 'validators', 'evaluators'] as HookKind[]).map((kind) => {
            const entries = Object.entries(hookCatalog?.[kind] || {});
            return (
              <div key={kind} className="domain-pack-hooks-group">
                <h5>{HOOK_KIND_LABELS[kind]}</h5>
                {entries.length === 0 ? (
                  <div className="domain-pack-hooks-empty">No hooks available.</div>
                ) : (
                  <ul>
                    {entries.map(([hookId, description]) => {
                      const source = hookCatalog?.plugin_hook_sources?.[kind]?.[hookId];
                      return (
                        <li key={`${kind}-${hookId}`}>
                          <code>{hookId}</code>
                          <span>{description}</span>
                          {source && <em>plugin: {source}</em>}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            );
          })}
        </div>

        <div className="domain-pack-plugin-status-grid">
          <div className="domain-pack-plugin-status-card">
            <h5>Loaded Plugin Modules</h5>
            {pluginModules.length === 0 ? (
              <div className="domain-pack-hooks-empty">None</div>
            ) : (
              <ul>
                {pluginModules.map((modulePath) => (
                  <li key={modulePath}>
                    <code>{modulePath}</code>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="domain-pack-plugin-status-card">
            <h5>Plugin Load Errors</h5>
            {pluginErrors.length === 0 ? (
              <div className="domain-pack-hooks-empty">None</div>
            ) : (
              <ul className="domain-pack-plugin-errors">
                {pluginErrors.map(([modulePath, message]) => (
                  <li key={modulePath}>
                    <code>{modulePath}</code>
                    <span>{message}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
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
              <div className="domain-pack-hook-editor">
                <div className="domain-pack-hook-editor-header">
                  <span className="form-label">Hook Helper</span>
                  <span className="domain-pack-hook-editor-note">Apply IDs directly into contract JSON</span>
                </div>
                <div className="domain-pack-hook-editor-grid">
                  {(Object.keys(CONTRACT_HOOK_LABELS) as ContractHookKey[]).map((key) => {
                    const kind = CONTRACT_KEY_TO_HOOK_KIND[key];
                    const entries = hookOptionsByKind[kind];
                    const selected = editorHookSelection[key];
                    const hasSelected = entries.some(([hookId]) => hookId === selected);
                    return (
                      <div key={key} className="form-group">
                        <label className="form-label">{CONTRACT_HOOK_LABELS[key]}</label>
                        <select
                          className="input"
                          value={selected}
                          onChange={(e) => {
                            const nextId = normalizeHookId(e.target.value) || DEFAULT_HOOK_SELECTION[key];
                            setEditorHookSelection((prev) => ({ ...prev, [key]: nextId }));
                          }}
                        >
                          {!hasSelected && selected && <option value={selected}>{selected} (custom)</option>}
                          {entries.map(([hookId, description]) => (
                            <option key={`${key}-${hookId}`} value={hookId}>
                              {hookId} - {description}
                            </option>
                          ))}
                        </select>
                      </div>
                    );
                  })}
                </div>
                <div className="domain-pack-hook-editor-actions">
                  <button className="btn btn-secondary btn-sm" onClick={handleLoadHookSelectionFromJson}>
                    Load From JSON
                  </button>
                  <button className="btn btn-secondary btn-sm" onClick={handleApplyHookSelectionToJson}>
                    Apply To JSON
                  </button>
                </div>
                {editorHookStatus && <div className="domain-pack-hook-editor-status">{editorHookStatus}</div>}
                {editorHookError && <div className="domain-pack-hook-editor-error">{editorHookError}</div>}
              </div>
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
