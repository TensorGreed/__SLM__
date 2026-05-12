/**
 * ProjectExtensionStudioPage — Extension Studio for Wave H power users
 * (priority.md P40). Hidden in beginner mode (the sidebar skips the
 * link unless ``isBeginner === false``).
 *
 * Surface includes:
 *   - Plugin-kind list with configured / loaded module counts and
 *     reload status badges.
 *   - Per-kind detail panel showing the contract version, recognised
 *     hook names, settings key, and inline load errors.
 *   - Scaffold form (POST /api/extensions/scaffold) with file preview
 *     + per-file blob download.
 *   - Validate panel (POST /api/extensions/validate) that exposes the
 *     full P37 check suite.
 *   - "Reload" button per-kind and "Reload all" header action.
 *
 * Endpoints used:
 *   GET  /api/extensions
 *   POST /api/extensions/scaffold
 *   POST /api/extensions/validate
 *   POST /api/extensions/reload
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';

import api from '../api/client';
import ExtensionKindList from '../components/extensions/ExtensionKindList';
import ScaffoldForm from '../components/extensions/ScaffoldForm';
import ScaffoldPreview from '../components/extensions/ScaffoldPreview';
import ValidatePanel from '../components/extensions/ValidatePanel';
import type {
    ExtensionListResponse,
    PluginContractReport,
    PluginKind,
    ReloadKindResult,
    ReloadResponse,
    ScaffoldRequest,
    ScaffoldResponse,
} from '../types/extensions';
import { PLUGIN_KIND_LABEL } from '../types/extensions';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

import './ProjectExtensionStudioPage.css';

interface ApiErrorShape {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
}

function extractErrorMessage(err: unknown, fallback = 'Request failed.'): string {
    const e = err as ApiErrorShape;
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    return e?.message || fallback;
}

export default function ProjectExtensionStudioPage() {
    const params = useParams();
    const workspace = useOutletContext<ProjectWorkspaceContextValue | null>();
    const routeProjectId = params.id ? Number.parseInt(params.id, 10) : null;
    const projectId =
        workspace?.projectId
        ?? (Number.isFinite(routeProjectId) ? routeProjectId : null);

    const [catalog, setCatalog] = useState<ExtensionListResponse | null>(null);
    const [selectedKind, setSelectedKind] = useState<PluginKind>('data_adapter');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [reloadResults, setReloadResults] = useState<ReloadKindResult[]>([]);
    const [reloading, setReloading] = useState(false);

    const [scaffold, setScaffold] = useState<ScaffoldResponse | null>(null);

    const fetchCatalog = useCallback(async () => {
        setLoading(true);
        try {
            const response = await api.get<ExtensionListResponse>('/extensions');
            setCatalog(response.data);
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load extensions.'));
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void fetchCatalog();
    }, [fetchCatalog]);

    const handleReload = useCallback(
        async (kind: PluginKind | null) => {
            setReloading(true);
            setError(null);
            try {
                const response = await api.post<ReloadResponse>(
                    '/extensions/reload',
                    kind ? { kind } : {},
                );
                setReloadResults(response.data.results || []);
                await fetchCatalog();
            } catch (err) {
                setError(extractErrorMessage(err, 'Reload failed.'));
            } finally {
                setReloading(false);
            }
        },
        [fetchCatalog],
    );

    const submitScaffold = useCallback(
        async (body: ScaffoldRequest): Promise<ScaffoldResponse> => {
            const response = await api.post<ScaffoldResponse>(
                '/extensions/scaffold',
                body,
            );
            return response.data;
        },
        [],
    );

    const submitValidate = useCallback(
        async (body: { kind: PluginKind; module: string; force_reload: boolean }) => {
            const response = await api.post<PluginContractReport>(
                '/extensions/validate',
                body,
            );
            return response.data;
        },
        [],
    );

    const selectedStatus = useMemo(() => {
        if (!catalog) return null;
        return catalog.kinds.find((row) => row.kind === selectedKind) ?? null;
    }, [catalog, selectedKind]);

    if (projectId == null) {
        return (
            <div className="workspace-page extension-studio-page">
                <div className="deployment-status is-error" role="alert">
                    Project context is not available.
                </div>
            </div>
        );
    }

    return (
        <div className="workspace-page extension-studio-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Extension Studio</h2>
                    <p className="workspace-page-subtitle">
                        Build, validate, and reload custom data adapters, training
                        runtimes, domain packs, and evaluation packs without
                        leaving the workspace. Hidden in beginner mode (Wave H,
                        priority.md P37/P38/P40).
                    </p>
                </div>
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => void fetchCatalog()}
                    disabled={loading}
                >
                    {loading ? 'Refreshing…' : 'Refresh'}
                </button>
            </section>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            <section className="extension-studio-layout">
                <aside className="extension-studio-sidebar card">
                    {catalog ? (
                        <ExtensionKindList
                            kinds={catalog.kinds}
                            selectedKind={selectedKind}
                            onSelectKind={setSelectedKind}
                            reloadResults={reloadResults}
                            onReload={handleReload}
                            reloading={reloading}
                        />
                    ) : (
                        <div className="dim">Loading kinds…</div>
                    )}
                </aside>

                <div className="extension-studio-main">
                    {selectedStatus && (
                        <div className="card extension-studio-detail">
                            <header className="extension-studio-detail-head">
                                <h3 className="observability-heading">
                                    {PLUGIN_KIND_LABEL[selectedStatus.kind]}
                                </h3>
                                <div className="extension-studio-detail-meta dim">
                                    Contract version{' '}
                                    <code>{selectedStatus.contract_version}</code>
                                    {selectedStatus.settings_key && (
                                        <>
                                            {' · settings key '}
                                            <code>{selectedStatus.settings_key}</code>
                                        </>
                                    )}
                                </div>
                                <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => void handleReload(selectedStatus.kind)}
                                    disabled={
                                        reloading
                                        || !selectedStatus.supports_safe_reload
                                    }
                                    title={
                                        selectedStatus.supports_safe_reload
                                            ? 'Reload modules listed in settings.'
                                            : 'Reload is not implemented for this kind yet.'
                                    }
                                >
                                    Reload kind
                                </button>
                            </header>
                            <ul className="extension-studio-detail-exports">
                                <li>
                                    <span className="dim">Recognised hooks:</span>{' '}
                                    {selectedStatus.recognized_exports.map((name) => (
                                        <code key={name}>{name}</code>
                                    ))}
                                </li>
                                <li>
                                    <span className="dim">Loaded modules:</span>{' '}
                                    {selectedStatus.loaded_modules.length === 0 ? (
                                        <span className="dim">none</span>
                                    ) : (
                                        selectedStatus.loaded_modules.map((module) => (
                                            <code key={module}>{module}</code>
                                        ))
                                    )}
                                </li>
                            </ul>
                            {Object.keys(selectedStatus.load_errors || {}).length > 0 && (
                                <ul className="extension-studio-load-errors">
                                    {Object.entries(selectedStatus.load_errors).map(
                                        ([moduleName, message]) => (
                                            <li key={moduleName}>
                                                <span className="badge badge-danger">
                                                    load error
                                                </span>{' '}
                                                <code>{moduleName}</code>
                                                <span className="dim">{message}</span>
                                            </li>
                                        ),
                                    )}
                                </ul>
                            )}
                            {selectedStatus.note && (
                                <p className="dim extension-studio-detail-note">
                                    {selectedStatus.note}
                                </p>
                            )}
                        </div>
                    )}

                    <div className="card extension-studio-card">
                        <h3 className="observability-heading">Generate scaffold</h3>
                        <p className="dim">
                            Scaffold a contract-valid starter module for{' '}
                            <strong>{PLUGIN_KIND_LABEL[selectedKind]}</strong>.
                            The bundle ships a module file, a self-validating
                            test stub, and a README.
                        </p>
                        <ScaffoldForm
                            kind={selectedKind}
                            submit={submitScaffold}
                            onGenerated={setScaffold}
                            onError={setError}
                        />
                        {scaffold && scaffold.kind === selectedKind && (
                            <ScaffoldPreview scaffold={scaffold} />
                        )}
                    </div>

                    <div className="card extension-studio-card">
                        <h3 className="observability-heading">Validate module</h3>
                        <p className="dim">
                            Run the P37 contract suite against a module path
                            without registering it into the live registry. Use
                            this before adding to{' '}
                            {selectedStatus?.settings_key ? (
                                <code>{selectedStatus.settings_key}</code>
                            ) : (
                                'settings'
                            )}
                            .
                        </p>
                        <ValidatePanel
                            kind={selectedKind}
                            onValidate={submitValidate}
                        />
                    </div>
                </div>
            </section>
        </div>
    );
}
