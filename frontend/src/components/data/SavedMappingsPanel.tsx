/**
 * Saved dataset-import mappings panel (Phase G of DATASET_IMPORT_PLAN).
 *
 * Lists every saved config for the project — locator, mapper,
 * last-run stats — and offers per-row "Re-run" + "Delete" actions.
 * The wizard's "Save mapping" button writes to the same endpoint
 * that backs this list, so newly-saved configs appear after the
 * wizard closes.
 *
 * Hidden when the project has no saved configs (the empty state
 * isn't useful before the user has done at least one import).
 */

import { useCallback, useEffect, useState } from 'react';
import {
    deleteSavedConfig,
    listSavedConfigs,
    runSavedConfig,
    type ImportResultDict,
    type SavedConfig,
} from '../../api/datasetImport';

interface SavedMappingsPanelProps {
    projectId: number;
    refreshKey?: number;
    onRunComplete?: (result: ImportResultDict) => void;
}

function extractErrorMessage(err: unknown): string {
    if (typeof err === 'object' && err !== null) {
        const data = (err as { response?: { data?: { detail?: unknown } } }).response?.data
            ?.detail;
        if (typeof data === 'string' && data.trim()) {
            return data;
        }
        const message = (err as { message?: unknown }).message;
        if (typeof message === 'string' && message.trim()) {
            return message;
        }
    }
    return 'Operation failed';
}

function formatRelative(iso: string | null): string {
    if (!iso) return 'never';
    const ts = Date.parse(iso);
    if (Number.isNaN(ts)) return iso;
    const seconds = Math.floor((Date.now() - ts) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

export default function SavedMappingsPanel({
    projectId,
    refreshKey,
    onRunComplete,
}: SavedMappingsPanelProps) {
    const [configs, setConfigs] = useState<SavedConfig[]>([]);
    const [loading, setLoading] = useState(false);
    const [listError, setListError] = useState<string>('');
    const [runningId, setRunningId] = useState<number | null>(null);
    const [actionError, setActionError] = useState<string>('');
    const [lastRunMsg, setLastRunMsg] = useState<string>('');

    const fetchConfigs = useCallback(async () => {
        setLoading(true);
        setListError('');
        try {
            const list = await listSavedConfigs(projectId);
            setConfigs(list);
        } catch (err) {
            setListError(extractErrorMessage(err));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchConfigs();
    }, [fetchConfigs, refreshKey]);

    const handleRun = async (config: SavedConfig) => {
        setRunningId(config.id);
        setActionError('');
        setLastRunMsg('');
        try {
            const result = await runSavedConfig(projectId, config.id);
            setLastRunMsg(
                `Re-imported ${result.accepted_count} row(s) via ${config.name}.`,
            );
            onRunComplete?.(result);
            // Refresh so last_run_* updates immediately.
            void fetchConfigs();
        } catch (err) {
            setActionError(extractErrorMessage(err));
        } finally {
            setRunningId(null);
        }
    };

    const handleDelete = async (config: SavedConfig) => {
        if (!window.confirm(`Delete saved mapping "${config.name}"?`)) return;
        setActionError('');
        try {
            await deleteSavedConfig(projectId, config.id);
            void fetchConfigs();
        } catch (err) {
            setActionError(extractErrorMessage(err));
        }
    };

    if (!loading && configs.length === 0 && !listError) {
        // Empty state stays hidden — a user who hasn't saved anything
        // yet doesn't need a "no saved mappings" panel cluttering the
        // Data tab.
        return null;
    }

    return (
        <div
            className="card"
            style={{ marginBottom: 'var(--space-lg)' }}
            data-testid="saved-mappings-panel"
        >
            <div
                style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    marginBottom: 'var(--space-sm)',
                }}
            >
                <div>
                    <h3 style={{ margin: 0 }}>Saved mappings</h3>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        Re-run the dataset-import pipeline against a refreshed source — no
                        re-introspecting needed.
                    </div>
                </div>
            </div>

            {lastRunMsg && (
                <div
                    style={{
                        padding: 'var(--space-sm) var(--space-md)',
                        background: 'rgba(34, 197, 94, 0.08)',
                        border: '1px solid rgba(34, 197, 94, 0.3)',
                        borderRadius: 'var(--radius-md)',
                        marginBottom: 'var(--space-sm)',
                        fontSize: '0.9rem',
                    }}
                    role="status"
                >
                    {lastRunMsg}
                </div>
            )}
            {listError && (
                <div className="error-banner" data-testid="saved-mappings-list-error">
                    {listError}
                </div>
            )}
            {actionError && (
                <div className="error-banner" data-testid="saved-mappings-action-error">
                    {actionError}
                </div>
            )}

            <table className="table" style={{ width: '100%', fontSize: '0.9rem' }}>
                <thead>
                    <tr>
                        <th style={{ textAlign: 'left' }}>Name</th>
                        <th style={{ textAlign: 'left' }}>Locator</th>
                        <th style={{ textAlign: 'left' }}>Mapper</th>
                        <th style={{ textAlign: 'right' }}>Last run</th>
                        <th style={{ textAlign: 'right' }}>Rows</th>
                        <th style={{ width: 200 }}></th>
                    </tr>
                </thead>
                <tbody>
                    {configs.map((cfg) => (
                        <tr key={cfg.id} data-testid={`saved-config-row-${cfg.id}`}>
                            <td>
                                <div>
                                    <strong>{cfg.name}</strong>
                                </div>
                                {cfg.description && (
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                        {cfg.description}
                                    </div>
                                )}
                            </td>
                            <td>
                                <code style={{ fontSize: '0.85rem' }}>{cfg.locator}</code>
                            </td>
                            <td>
                                <code style={{ fontSize: '0.85rem' }}>{cfg.mapper_id}</code>
                            </td>
                            <td style={{ textAlign: 'right' }}>
                                {formatRelative(cfg.last_run_at)}
                            </td>
                            <td style={{ textAlign: 'right' }}>
                                {cfg.last_run_accepted ?? '—'}
                            </td>
                            <td style={{ textAlign: 'right' }}>
                                <button
                                    type="button"
                                    className="btn btn-primary"
                                    onClick={() => handleRun(cfg)}
                                    disabled={runningId !== null}
                                    style={{ marginRight: 4 }}
                                    data-testid={`run-config-${cfg.id}`}
                                >
                                    {runningId === cfg.id ? 'Running…' : 'Re-run'}
                                </button>
                                <button
                                    type="button"
                                    className="btn btn-ghost"
                                    onClick={() => handleDelete(cfg)}
                                    disabled={runningId !== null}
                                    data-testid={`delete-config-${cfg.id}`}
                                >
                                    Delete
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
