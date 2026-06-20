/**
 * Panel showing prepared dataset version history with reproducibility and reuse-signal checks.
 */

import { useEffect, useMemo, useState } from 'react';
import {
    AlertTriangle,
    CheckCircle2,
    Download,
    ExternalLink,
    FileCheck2,
    GitBranch,
    History,
    Play,
    RefreshCw,
    ShieldCheck,
    Workflow,
} from 'lucide-react';

import api from '../../api/client';
import {
    activatePreparedVersion,
    comparePreparedVersions,
    getDataStudioDatasetVersions,
    runDataStudioPrepareDataset,
} from '../../api/dataStudio';
import type {
    DataStudioDatasetVersionArtifact,
    DataStudioDatasetVersionHistoryItem,
    DataStudioDatasetVersionSignal,
    DataStudioDatasetVersions,
    DataStudioIssue,
    PreparedVersionComparison,
    RunPrepareDatasetResult,
} from '../../api/dataStudio';
import './DataStudioDatasetVersionsPanel.css';

interface DataStudioDatasetVersionsPanelProps {
    projectId: number;
    onOpenTarget: (target: string) => void;
}

const VERSION_VERDICT_COPY: Record<DataStudioDatasetVersions['verdict'], { label: string; detail: string }> = {
    empty: {
        label: 'No versions',
        detail: 'Run Dataset Prep to create reusable train, validation, and test dataset versions.',
    },
    attention: {
        label: 'Check versions',
        detail: 'Prepared versions exist, but reproducibility or reuse signals need review.',
    },
    ready: {
        label: 'Reusable',
        detail: 'Prepared dataset versions are aligned with the manifest and ready to reuse downstream.',
    },
};

function formatNumber(value: number | undefined): string {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function labelForToken(value: string | undefined | null): string {
    if (!value) return 'Unknown';
    return value.replace(/_/g, ' ');
}

function compactJson(value: unknown): string {
    return JSON.stringify(value || {}, null, 2);
}

function statusClass(status: string): string {
    if (status === 'met' || status === 'ready') {
        return 'ready';
    }
    if (status === 'attention') {
        return 'attention';
    }
    return 'missing';
}

function statusIcon(status: string) {
    const normalized = statusClass(status);
    if (normalized === 'ready') {
        return <CheckCircle2 size={16} aria-hidden="true" />;
    }
    return <AlertTriangle size={16} aria-hidden="true" />;
}

function issueIcon(issue: DataStudioIssue) {
    if (issue.severity === 'info') {
        return <CheckCircle2 size={15} aria-hidden="true" />;
    }
    return <AlertTriangle size={15} aria-hidden="true" />;
}

function SignalRow({
    signal,
    onOpenTarget,
}: {
    signal: DataStudioDatasetVersionSignal;
    onOpenTarget: (target: string) => void;
}) {
    return (
        <button
            type="button"
            className={`data-studio-versions__signal data-studio-versions__signal--${statusClass(signal.status)}`}
            onClick={() => onOpenTarget(signal.target_tab)}
        >
            <span>{statusIcon(signal.status)}</span>
            <span>
                <strong>{signal.label}</strong>
                <small>{signal.message}</small>
            </span>
            <b>{labelForToken(signal.status)}</b>
        </button>
    );
}

function ArtifactCard({
    artifact,
    projectId,
}: {
    artifact: DataStudioDatasetVersionArtifact;
    projectId: number;
}) {
    const [downloading, setDownloading] = useState(false);
    const [downloadError, setDownloadError] = useState<string | null>(null);
    const versionLabel = artifact.latest_version_number
        ? `v${artifact.latest_version_number}`
        : 'No version';
    const manifestLabel = artifact.manifest_version
        ? `manifest v${artifact.manifest_version}`
        : 'no manifest ref';

    const handleDownload = async () => {
        setDownloading(true);
        setDownloadError(null);
        try {
            // Auth-aware download: fetch the JSONL as a blob through the axios
            // client (carries the token) then trigger a client-side save.
            const resp = await api.get(
                `/projects/${projectId}/data-studio/dataset-versions/${artifact.key}/export`,
                { responseType: 'blob' },
            );
            const url = URL.createObjectURL(resp.data as Blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `project-${projectId}-${artifact.key}.jsonl`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (err: any) {
            setDownloadError(err?.response?.data?.detail || err?.message || 'Download failed.');
        } finally {
            setDownloading(false);
        }
    };
    return (
        <article className="data-studio-versions__artifact">
            <div className="data-studio-versions__artifact-head">
                <div>
                    <strong>{artifact.label}</strong>
                    <small>{labelForToken(artifact.dataset_type)}</small>
                </div>
                <span>{versionLabel}</span>
            </div>
            <dl>
                <div>
                    <dt>Rows</dt>
                    <dd>{formatNumber(artifact.row_count)}</dd>
                </div>
                <div>
                    <dt>Manifest</dt>
                    <dd>{formatNumber(artifact.manifest_count)}</dd>
                </div>
                <div>
                    <dt>Versions</dt>
                    <dd>{formatNumber(artifact.version_count)}</dd>
                </div>
            </dl>
            <div className="data-studio-versions__artifact-flags">
                <span className={artifact.file_exists ? 'is-ready' : 'is-missing'}>
                    {artifact.file_exists ? 'File found' : 'File missing'}
                </span>
                <span className={artifact.version_matches_manifest ? 'is-ready' : 'is-attention'}>
                    {manifestLabel}
                </span>
                <span className={artifact.row_count_matches_manifest ? 'is-ready' : 'is-attention'}>
                    {artifact.row_count_matches_manifest ? 'Counts match' : 'Counts differ'}
                </span>
            </div>
            {artifact.file_exists && (
                <div className="data-studio-versions__artifact-actions">
                    <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => void handleDownload()}
                        disabled={downloading}
                    >
                        <Download size={13} aria-hidden="true" />
                        {downloading ? 'Exporting…' : 'Export JSONL'}
                    </button>
                    {downloadError && (
                        <span className="badge badge-danger" role="alert">{downloadError}</span>
                    )}
                </div>
            )}
        </article>
    );
}

function HistoryRow({ item }: { item: DataStudioDatasetVersionHistoryItem }) {
    const latest = item.latest_version;
    return (
        <div className="data-studio-versions__history-row">
            <div>
                <strong>{item.dataset_name}</strong>
                <small>
                    {labelForToken(item.dataset_type)}
                    {' · '}
                    {item.file_exists ? 'file found' : 'file missing'}
                </small>
            </div>
            <div>
                <span>{latest ? `v${latest.version}` : 'No version'}</span>
                <small>{formatNumber(item.version_count)} total</small>
            </div>
        </div>
    );
}

export default function DataStudioDatasetVersionsPanel({
    projectId,
    onOpenTarget,
}: DataStudioDatasetVersionsPanelProps) {
    const [versions, setVersions] = useState<DataStudioDatasetVersions | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    // Arc A — inline re-prepare when manifest drift is detected.
    // Same backend call as the Prepare panel's "Run prepare now"; the
    // UI shows it only when there's actually drift to fix.
    const [running, setRunning] = useState(false);
    const [runFlash, setRunFlash] = useState<string | null>(null);
    const [runError, setRunError] = useState<string | null>(null);
    // Epic E — activate/retrain a prepared-version snapshot.
    const [activateBusy, setActivateBusy] = useState<number | null>(null);
    const [activateFlash, setActivateFlash] = useState<string | null>(null);
    const [activateError, setActivateError] = useState<string | null>(null);
    // Epic E — compare two prepared-version snapshots.
    const [compareA, setCompareA] = useState<number | null>(null);
    const [compareB, setCompareB] = useState<number | null>(null);
    const [comparison, setComparison] = useState<PreparedVersionComparison | null>(null);
    const [compareError, setCompareError] = useState<string | null>(null);

    const loadVersions = async () => {
        setLoading(true);
        try {
            const data = await getDataStudioDatasetVersions(projectId);
            setVersions(data);
            setError(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to load Data Studio dataset versions.');
        } finally {
            setLoading(false);
        }
    };

    const handleRePrepare = async () => {
        setRunning(true);
        setRunFlash(null);
        setRunError(null);
        try {
            const result: RunPrepareDatasetResult = await runDataStudioPrepareDataset(projectId);
            const total =
                Number(result.train_count || 0)
                + Number(result.val_count || 0)
                + Number(result.test_count || 0);
            setRunFlash(
                total > 0
                    ? `Re-prepared ${total.toLocaleString()} rows — manifest now matches train/val/test.`
                    : 'Re-prepared dataset splits — refreshing version checks.',
            );
            await loadVersions();
        } catch (err: any) {
            setRunError(
                err?.response?.data?.detail
                || err?.message
                || 'Re-prepare failed.',
            );
        } finally {
            setRunning(false);
        }
    };

    const handleActivate = async (version: number, retrain: boolean) => {
        setActivateBusy(version);
        setActivateFlash(null);
        setActivateError(null);
        try {
            const result = await activatePreparedVersion(projectId, version);
            const total = Object.values(result.restored_counts).reduce((a, b) => a + b, 0);
            setActivateFlash(
                `v${version} is now the active prepared dataset (${total.toLocaleString()} rows). `
                + (retrain ? 'Opening Training…' : 'Training will use this version.'),
            );
            await loadVersions();
            if (retrain) {
                onOpenTarget('training');
            }
        } catch (err: any) {
            setActivateError(
                err?.response?.data?.detail || err?.message || `Failed to activate v${version}.`,
            );
        } finally {
            setActivateBusy(null);
        }
    };

    const handleCompare = async () => {
        if (compareA == null || compareB == null || compareA === compareB) {
            return;
        }
        setCompareError(null);
        setComparison(null);
        try {
            setComparison(await comparePreparedVersions(projectId, compareA, compareB));
        } catch (err: any) {
            setCompareError(err?.response?.data?.detail || err?.message || 'Compare failed.');
        }
    };

    useEffect(() => {
        void loadVersions();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [projectId]);

    const topIssues = useMemo(
        () => versions?.issues.slice(0, 5) ?? [],
        [versions],
    );
    const entryPoints = useMemo(
        () => versions?.entry_points.slice(0, 3) ?? [],
        [versions],
    );
    // Arc A — only surface the "Re-prepare" button when an artifact
    // actually disagrees with the manifest (version ref OR row count).
    // No drift → don't pollute the toolbar with a button that would
    // re-build identical files.
    const hasManifestDrift = useMemo(
        () =>
            (versions?.latest_artifacts ?? []).some(
                (a) =>
                    a.file_exists
                    && (!a.version_matches_manifest || !a.row_count_matches_manifest),
            ),
        [versions],
    );

    if (loading && !versions) {
        return (
            <section className="data-studio-versions data-studio-versions--loading">
                <span>Loading dataset versions...</span>
            </section>
        );
    }

    if (error && !versions) {
        return (
            <section className="data-studio-versions data-studio-versions--error">
                <div>
                    <h3>Dataset Versions</h3>
                    <p>{error}</p>
                </div>
                <button type="button" className="btn btn-secondary" onClick={() => void loadVersions()}>
                    Retry
                </button>
            </section>
        );
    }

    if (!versions) {
        return null;
    }

    const verdict = VERSION_VERDICT_COPY[versions.verdict];
    const recipeName = versions.source_context.recipe?.name || 'No recipe';
    const domainName = versions.source_context.domain?.profile_display_name
        || versions.source_context.domain?.pack_display_name
        || 'Generic domain';

    return (
        <section
            className={`data-studio-versions data-studio-versions--${versions.verdict}`}
            data-testid="data-studio-dataset-versions"
        >
            <div className="data-studio-versions__header">
                <div>
                    <p className="data-studio-versions__eyebrow">Versions</p>
                    <h3>Dataset Versions</h3>
                    <p>{verdict.detail}</p>
                </div>
                <div className="data-studio-versions__actions">
                    <span className={`data-studio-versions__verdict data-studio-versions__verdict--${versions.verdict}`}>
                        {verdict.label}
                    </span>
                    <button
                        type="button"
                        className="btn btn-ghost data-studio-versions__refresh"
                        onClick={() => void loadVersions()}
                        aria-label="Refresh Data Studio dataset versions"
                    >
                        <RefreshCw size={16} aria-hidden="true" />
                    </button>
                </div>
            </div>

            <div className="data-studio-versions__metrics" aria-label="Dataset version metrics">
                <div className="data-studio-versions__metric">
                    <History size={18} aria-hidden="true" />
                    <span>Versions</span>
                    <strong>{formatNumber(versions.summary.total_version_count)}</strong>
                </div>
                <div className="data-studio-versions__metric">
                    <GitBranch size={18} aria-hidden="true" />
                    <span>Latest rows</span>
                    <strong>{formatNumber(versions.summary.latest_total_rows)}</strong>
                </div>
                <div className="data-studio-versions__metric">
                    <FileCheck2 size={18} aria-hidden="true" />
                    <span>Manifest refs</span>
                    <strong>{formatNumber(versions.summary.manifest_version_ref_count)}</strong>
                </div>
                <div className="data-studio-versions__metric">
                    <ShieldCheck size={18} aria-hidden="true" />
                    <span>Reuse</span>
                    <strong>{versions.summary.training_reuse_ready ? 'Ready' : 'Review'}</strong>
                </div>
            </div>

            <div className="data-studio-versions__signals">
                <span>{recipeName}</span>
                <span>{domainName}</span>
                <span>{versions.source_context.adapter_id || 'No adapter in manifest'}</span>
                <span>{versions.manifest.readable ? 'Manifest readable' : 'Manifest missing'}</span>
                {versions.read_only ? <span>Read-only check</span> : null}
            </div>

            <div className="data-studio-versions__entrypoints">
                {hasManifestDrift && (
                    <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => void handleRePrepare()}
                        disabled={running}
                        title="Re-build train/validation/test JSONLs so the manifest version + row counts realign."
                        data-testid="data-studio-versions-re-prepare"
                    >
                        <Play size={15} aria-hidden="true" />
                        {running ? 'Re-preparing…' : 'Re-prepare to fix drift'}
                    </button>
                )}
                {entryPoints.map((entry) => (
                    <button
                        type="button"
                        className={
                            entry.target_tab === 'dataprep' && !hasManifestDrift
                                ? 'btn btn-primary'
                                : 'btn btn-secondary'
                        }
                        key={entry.target_tab}
                        onClick={() => onOpenTarget(entry.target_tab)}
                    >
                        <ExternalLink size={15} aria-hidden="true" />
                        {entry.label}
                    </button>
                ))}
            </div>
            {runFlash && (
                <p
                    className="data-studio-versions__run-flash"
                    data-testid="data-studio-versions-run-flash"
                >
                    {runFlash}
                </p>
            )}
            {runError && (
                <p
                    className="data-studio-versions__run-error"
                    data-testid="data-studio-versions-run-error"
                >
                    {runError}
                </p>
            )}

            <div className="data-studio-versions__reuse">
                <div className={`data-studio-versions__reuse-item data-studio-versions__reuse-item--${statusClass(versions.reuse_readiness.training.status)}`}>
                    <Workflow size={18} aria-hidden="true" />
                    <div>
                        <strong>Training reuse</strong>
                        <small>{versions.reuse_readiness.training.message}</small>
                    </div>
                    <button type="button" className="btn btn-ghost" onClick={() => onOpenTarget('training')}>
                        Training
                    </button>
                </div>
                <div className={`data-studio-versions__reuse-item data-studio-versions__reuse-item--${statusClass(versions.reuse_readiness.evaluation.status)}`}>
                    <FileCheck2 size={18} aria-hidden="true" />
                    <div>
                        <strong>Eval reuse</strong>
                        <small>{versions.reuse_readiness.evaluation.message}</small>
                    </div>
                    <button type="button" className="btn btn-ghost" onClick={() => onOpenTarget('eval')}>
                        Eval
                    </button>
                </div>
            </div>

            <div className="data-studio-versions__body">
                <div className="data-studio-versions__artifacts">
                    <h4>Latest artifacts</h4>
                    <div className="data-studio-versions__artifact-list">
                        {versions.latest_artifacts.map((artifact) => (
                            <ArtifactCard artifact={artifact} projectId={projectId} key={artifact.key} />
                        ))}
                    </div>
                </div>

                <div className="data-studio-versions__checks">
                    <h4>Reproducibility</h4>
                    <div className="data-studio-versions__signal-list">
                        {versions.reproducibility.map((signal) => (
                            <SignalRow signal={signal} key={signal.id} onOpenTarget={onOpenTarget} />
                        ))}
                    </div>
                </div>
            </div>

            {versions.prepared_versions && versions.prepared_versions.available.length > 0 && (
                <div className="data-studio-versions__snapshots" data-testid="prepared-version-snapshots">
                    <h4>
                        <GitBranch size={15} aria-hidden="true" />
                        Prepared snapshots — activate or retrain from any version
                    </h4>
                    {activateFlash && (
                        <p className="data-studio-versions__activate-flash" role="status" data-testid="activate-flash">
                            {activateFlash}
                        </p>
                    )}
                    {activateError && (
                        <p className="data-studio-versions__activate-error" role="alert">{activateError}</p>
                    )}
                    <div className="data-studio-versions__snapshot-list">
                        {versions.prepared_versions.available.map((snap) => (
                            <div className="data-studio-versions__snapshot" key={snap.version}>
                                <span className="data-studio-versions__snapshot-label">
                                    <strong>v{snap.version}</strong>
                                    {snap.is_active && (
                                        <span className="badge badge-success" data-testid={`snapshot-active-${snap.version}`}>
                                            Active
                                        </span>
                                    )}
                                </span>
                                <div className="data-studio-versions__snapshot-actions">
                                    <button
                                        type="button"
                                        className="btn btn-secondary btn-sm"
                                        disabled={activateBusy !== null || snap.is_active}
                                        onClick={() => void handleActivate(snap.version, false)}
                                    >
                                        {activateBusy === snap.version ? 'Activating…' : 'Make active'}
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn-primary btn-sm"
                                        disabled={activateBusy !== null}
                                        onClick={() => void handleActivate(snap.version, true)}
                                    >
                                        <Play size={13} aria-hidden="true" />
                                        Retrain from this
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>

                    {versions.prepared_versions.available.length >= 2 && (
                        <div className="data-studio-versions__compare" data-testid="version-compare">
                            <div className="data-studio-versions__compare-controls">
                                <span>Compare</span>
                                <select
                                    aria-label="Compare version A"
                                    value={compareA ?? ''}
                                    onChange={(e) => setCompareA(e.target.value ? Number(e.target.value) : null)}
                                >
                                    <option value="">v…</option>
                                    {versions.prepared_versions.available.map((s) => (
                                        <option key={s.version} value={s.version}>v{s.version}</option>
                                    ))}
                                </select>
                                <span>vs</span>
                                <select
                                    aria-label="Compare version B"
                                    value={compareB ?? ''}
                                    onChange={(e) => setCompareB(e.target.value ? Number(e.target.value) : null)}
                                >
                                    <option value="">v…</option>
                                    {versions.prepared_versions.available.map((s) => (
                                        <option key={s.version} value={s.version}>v{s.version}</option>
                                    ))}
                                </select>
                                <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    disabled={compareA == null || compareB == null || compareA === compareB}
                                    onClick={() => void handleCompare()}
                                >
                                    Compare
                                </button>
                            </div>
                            {compareError && (
                                <p className="data-studio-versions__activate-error" role="alert">{compareError}</p>
                            )}
                            {comparison && (
                                <div className="data-studio-versions__compare-result" data-testid="version-compare-result">
                                    <p>
                                        v{comparison.a.version} → v{comparison.b.version}:{' '}
                                        <strong>
                                            {comparison.diff.total_delta >= 0 ? '+' : ''}
                                            {comparison.diff.total_delta}
                                        </strong>{' '}
                                        rows total
                                        {Object.entries(comparison.diff.split_deltas).map(([k, d]) => (
                                            <span key={k}> · {k} {d >= 0 ? '+' : ''}{d}</span>
                                        ))}
                                    </p>
                                    {comparison.diff.sources_added.length > 0 && (
                                        <p>Sources added: {comparison.diff.sources_added.join(', ')}</p>
                                    )}
                                    {comparison.diff.sources_removed.length > 0 && (
                                        <p>Sources removed: {comparison.diff.sources_removed.join(', ')}</p>
                                    )}
                                    {(comparison.diff.seed_changed
                                        || comparison.diff.ratios_changed
                                        || comparison.diff.strategy_changed) && (
                                        <p>
                                            Changed:
                                            {comparison.diff.seed_changed && ' seed'}
                                            {comparison.diff.ratios_changed && ' ratios'}
                                            {comparison.diff.strategy_changed && ' split-strategy'}
                                        </p>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            <div className="data-studio-versions__history">
                <h4>Version history</h4>
                {versions.version_history.length > 0 ? (
                    <div className="data-studio-versions__history-list">
                        {versions.version_history.map((item) => (
                            <HistoryRow item={item} key={item.dataset_id} />
                        ))}
                    </div>
                ) : (
                    <p className="data-studio-versions__empty">
                        Version history appears after Dataset Prep writes prepared split versions.
                    </p>
                )}
            </div>

            {topIssues.length > 0 ? (
                <ul className="data-studio-versions__issues">
                    {topIssues.map((issue) => (
                        <li key={issue.id} className={`data-studio-versions__issue data-studio-versions__issue--${issue.severity}`}>
                            <span>{issueIcon(issue)}</span>
                            <div>
                                <strong>{issue.title}</strong>
                                <small>{issue.message}</small>
                            </div>
                            <button type="button" className="btn btn-ghost" onClick={() => onOpenTarget(issue.target_tab)}>
                                {issue.action_label}
                            </button>
                        </li>
                    ))}
                </ul>
            ) : null}

            <details className="data-studio-versions__details">
                <summary>Power details</summary>
                <pre>{compactJson(versions)}</pre>
            </details>
        </section>
    );
}
