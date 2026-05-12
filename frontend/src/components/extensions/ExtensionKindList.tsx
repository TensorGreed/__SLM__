/**
 * ExtensionKindList — left column of the Extension Studio page
 * (priority.md P40). Lists the four plugin kinds (P37) with their
 * configured + loaded module counts and inline load errors. The
 * "Reload" button drives the project-wide reload pathway (P37).
 */

import { useCallback } from 'react';

import type {
    ExtensionKindStatus,
    PluginKind,
    ReloadKindResult,
} from '../../types/extensions';
import { PLUGIN_KIND_LABEL } from '../../types/extensions';

interface Props {
    kinds: ExtensionKindStatus[];
    selectedKind: PluginKind;
    onSelectKind: (kind: PluginKind) => void;
    reloadResults: ReloadKindResult[];
    onReload: (kind: PluginKind | null) => Promise<void> | void;
    reloading: boolean;
}

function ReloadBadge({ status }: { status: ReloadKindResult['status'] }) {
    const variant =
        status === 'ok'
            ? 'badge-success'
            : status === 'not_supported'
                ? 'badge-warning'
                : status === 'partial'
                    ? 'badge-warning'
                    : 'badge-danger';
    return <span className={`badge ${variant}`}>{status}</span>;
}

export default function ExtensionKindList({
    kinds,
    selectedKind,
    onSelectKind,
    reloadResults,
    onReload,
    reloading,
}: Props) {
    const resultByKind = new Map<PluginKind, ReloadKindResult>();
    for (const result of reloadResults) {
        resultByKind.set(result.kind, result);
    }

    const handleReloadAll = useCallback(() => {
        void onReload(null);
    }, [onReload]);

    return (
        <div className="extension-kind-list">
            <div className="extension-kind-list-header">
                <h3 className="observability-heading">Plugin kinds</h3>
                <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleReloadAll}
                    disabled={reloading}
                >
                    {reloading ? 'Reloading…' : 'Reload all'}
                </button>
            </div>
            <ul className="extension-kind-rows" aria-label="Plugin kinds">
                {kinds.map((row) => {
                    const errorCount = Object.keys(row.load_errors || {}).length;
                    const isSelected = row.kind === selectedKind;
                    const reload = resultByKind.get(row.kind);
                    return (
                        <li
                            key={row.kind}
                            className={`extension-kind-row ${isSelected ? 'is-selected' : ''}`}
                        >
                            <button
                                type="button"
                                className="extension-kind-button"
                                onClick={() => onSelectKind(row.kind)}
                                aria-pressed={isSelected}
                                aria-label={`Select ${PLUGIN_KIND_LABEL[row.kind]} kind`}
                            >
                                <div className="extension-kind-row-title">
                                    <span className="extension-kind-name">
                                        {PLUGIN_KIND_LABEL[row.kind]}
                                    </span>
                                    {!row.has_module_loader && (
                                        <span className="badge badge-warning">
                                            loader pending
                                        </span>
                                    )}
                                </div>
                                <div className="extension-kind-row-meta dim">
                                    <code>{row.contract_version}</code>
                                </div>
                                <div className="extension-kind-row-stats">
                                    <span className="badge badge-info">
                                        {row.registered_count} registered
                                    </span>
                                    <span className="badge badge-info">
                                        {row.loaded_modules.length} module(s)
                                    </span>
                                    {errorCount > 0 && (
                                        <span className="badge badge-danger">
                                            {errorCount} error(s)
                                        </span>
                                    )}
                                    {reload && (
                                        <ReloadBadge status={reload.status} />
                                    )}
                                </div>
                            </button>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}
