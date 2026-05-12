/**
 * SupportBundleCard — generate + list + download surface for the
 * project's support bundles (priority.md P36, P34).
 *
 * Generate a redacted bundle for the current project and surface a
 * preview of what's in it: per-section row counts and per-section
 * redaction stats (which secret-shape patterns hit, how many times).
 * Recent bundles list with their redaction summaries so the operator
 * can verify a bundle was scrubbed before forwarding it to support.
 *
 * Download links open in a new tab; the URL embeds the per-bundle
 * download token returned at create time.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import api from '../../api/client';
import type {
    SupportBundleListItem,
    SupportBundleListResponse,
    SupportBundleMetadata,
} from '../../types/observability';
import CommandSnippet from '../shared/CommandSnippet';
import EmptyState from '../shared/EmptyState';

interface Props {
    projectId: number;
}

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

function formatTs(value: string | null): string {
    if (!value) return '—';
    try {
        const d = new Date(value);
        if (Number.isNaN(d.getTime())) return value;
        return d.toLocaleString();
    } catch {
        return value;
    }
}

function formatBytes(n: number): string {
    if (!Number.isFinite(n) || n <= 0) return '—';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function summariseRedactions(stats: SupportBundleListItem['redactions_applied']): {
    total: number;
    bySection: Array<{ section: string; total: number }>;
} {
    let total = 0;
    const bySection: Array<{ section: string; total: number }> = [];
    for (const [section, payload] of Object.entries(stats || {})) {
        const sectionTotal = Number(payload?.total ?? 0);
        total += sectionTotal;
        if (sectionTotal > 0) {
            bySection.push({ section, total: sectionTotal });
        }
    }
    bySection.sort((a, b) => b.total - a.total);
    return { total, bySection };
}

export default function SupportBundleCard({ projectId }: Props) {
    const [bundles, setBundles] = useState<SupportBundleListItem[]>([]);
    const [latest, setLatest] = useState<SupportBundleMetadata | null>(null);
    const [loading, setLoading] = useState(false);
    const [creating, setCreating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchBundles = useCallback(async () => {
        setLoading(true);
        try {
            const response = await api.get<SupportBundleListResponse>(
                `/projects/${projectId}/support-bundles`,
            );
            setBundles(response.data.bundles || []);
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load bundles.'));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchBundles();
    }, [fetchBundles]);

    const createBundle = useCallback(async () => {
        setCreating(true);
        setError(null);
        try {
            const response = await api.post<SupportBundleMetadata>(
                `/projects/${projectId}/support-bundle`,
                {},
            );
            setLatest(response.data);
            await fetchBundles();
        } catch (err) {
            setError(extractErrorMessage(err, 'Bundle creation failed.'));
        } finally {
            setCreating(false);
        }
    }, [fetchBundles, projectId]);

    const latestSummary = useMemo(() => {
        if (!latest) return null;
        return {
            sectionCounts: latest.section_counts,
            redactions: summariseRedactions(latest.redactions_applied),
            sizeBytes: latest.size_bytes,
            expiresAt: latest.expires_at,
            downloadUrl: latest.download_url,
            bundleUid: latest.bundle_uid,
        };
    }, [latest]);

    return (
        <div className="card support-bundle-card">
            <div className="support-bundle-header">
                <div>
                    <h3 className="observability-heading">Support bundle</h3>
                    <div className="dim">
                        Packs decision logs, recent run-events, training manifests,
                        deployment versions, and failure clusters into a single
                        zip. Sensitive keys + secret patterns are scrubbed before
                        write.
                    </div>
                </div>
                <div className="support-bundle-header-actions">
                    <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={creating}
                        onClick={() => void createBundle()}
                    >
                        {creating ? 'Generating…' : 'Generate bundle'}
                    </button>
                    <CommandSnippet
                        cli={`brewslm support-bundle create --project ${projectId} --download`}
                        api={{
                            method: 'POST',
                            path: `/projects/${projectId}/support-bundle`,
                            body: {},
                        }}
                    />
                </div>
            </div>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            {latestSummary && (
                <div className="support-bundle-latest">
                    <div className="support-bundle-latest-head">
                        <span className="badge badge-success">just generated</span>
                        <code>{latestSummary.bundleUid}</code>
                        <span className="dim">
                            {formatBytes(latestSummary.sizeBytes)} · expires{' '}
                            {formatTs(latestSummary.expiresAt)}
                        </span>
                        <a
                            className="btn btn-secondary btn-sm"
                            href={latestSummary.downloadUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Download zip
                        </a>
                    </div>
                    <div className="support-bundle-preview">
                        <div className="support-bundle-section-counts">
                            <div className="dim">Sections (rows)</div>
                            <ul>
                                {Object.entries(
                                    latestSummary.sectionCounts || {},
                                ).map(([section, count]) => (
                                    <li key={section}>
                                        <code>{section}</code>{' '}
                                        <span className="dim">{count}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                        <div className="support-bundle-redaction-summary">
                            <div className="dim">
                                Redactions ({latestSummary.redactions.total} total)
                            </div>
                            {latestSummary.redactions.total === 0 ? (
                                <div className="dim">
                                    No sensitive values detected.
                                </div>
                            ) : (
                                <ul>
                                    {latestSummary.redactions.bySection.map((row) => (
                                        <li key={row.section}>
                                            <code>{row.section}</code>{' '}
                                            <span className="badge badge-warning">
                                                {row.total} scrubbed
                                            </span>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    </div>
                </div>
            )}

            <div className="support-bundle-history">
                <h4>Recent bundles</h4>
                {loading && !bundles.length && (
                    <div className="dim">Loading bundles…</div>
                )}
                {!loading && !bundles.length && (
                    <EmptyState
                        title="No support bundles yet"
                        description="A support bundle packages recent events, failure clusters, and deployment state into one redacted zip you can hand to oncall. Generate one when you hit an issue worth forwarding."
                        primary={{ label: 'Generate first bundle', onClick: () => void createBundle() }}
                        docsHref="http://localhost:3001/docs/observability/support-bundles"
                    />
                )}
                {bundles.length > 0 && (
                    <table className="deployment-table" aria-label="Support bundles">
                        <thead>
                            <tr>
                                <th>UID</th>
                                <th>Created</th>
                                <th>Size</th>
                                <th>Expires</th>
                                <th>Sections</th>
                                <th>Redactions</th>
                                <th>Actor</th>
                            </tr>
                        </thead>
                        <tbody>
                            {bundles.map((bundle) => {
                                const summary = summariseRedactions(
                                    bundle.redactions_applied,
                                );
                                const sectionCount = Object.keys(
                                    bundle.section_counts || {},
                                ).length;
                                return (
                                    <tr key={bundle.bundle_uid}>
                                        <td>
                                            <code>
                                                {bundle.bundle_uid.slice(0, 12)}…
                                            </code>
                                        </td>
                                        <td>{formatTs(bundle.created_at)}</td>
                                        <td>{formatBytes(bundle.size_bytes)}</td>
                                        <td>{formatTs(bundle.expires_at)}</td>
                                        <td>{sectionCount}</td>
                                        <td>{summary.total}</td>
                                        <td>{bundle.actor}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
