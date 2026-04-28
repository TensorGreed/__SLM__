/**
 * ManifestSummaryCard — compact "Pipeline-as-Code" preview rendered at
 * the top of the manifest page (priority.md P24).
 *
 * Pulls `GET /api/projects/{id}/manifest/export?format=json` and renders
 * a small grid of high-level facts (api_version, blueprint domain, data
 * sources count, adapters count, target profile) plus an inline export
 * button so an operator can pull the YAML without leaving the card.
 */

import { useCallback, useEffect, useState } from 'react';

import api from '../../api/client';
import type { BrewslmManifest } from '../../types/manifest';
import ManifestExportButton from './ManifestExportButton';

interface ManifestSummaryCardProps {
    projectId: number;
    projectName: string;
}

function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
}

export default function ManifestSummaryCard({ projectId, projectName }: ManifestSummaryCardProps) {
    const [manifest, setManifest] = useState<BrewslmManifest | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const fetchManifest = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await api.get<BrewslmManifest>(
                `/projects/${projectId}/manifest/export`,
                { params: { format: 'json' } },
            );
            setManifest(response.data);
        } catch (err) {
            setError(errorDetail(err, 'Failed to load manifest summary.'));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchManifest();
    }, [fetchManifest]);

    const spec = manifest?.spec ?? {};
    const dataSourceCount = spec.data_sources?.length ?? 0;
    const adaptersCount = spec.adapters?.length ?? 0;
    const targetProfile = spec.workflow?.target_profile_id ?? '—';
    const blueprintDomain = spec.blueprint?.domain_name || '—';
    const apiVersion = manifest?.api_version || '—';

    return (
        <section className="card manifest-summary-card" aria-label="Manifest summary">
            <div className="manifest-summary-card-head">
                <div>
                    <h3 style={{ margin: 0 }}>Pipeline as Code</h3>
                    <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                        The project's current state, exportable as a checked-in <code>brewslm.yaml</code>.
                    </p>
                </div>
                <ManifestExportButton projectId={projectId} projectName={projectName} size="sm" />
            </div>

            {error && (
                <div className="manifest-status is-error" role="alert">
                    {error}
                </div>
            )}

            {loading && !manifest && (
                <div className="skeleton" style={{ height: 80 }} />
            )}

            {manifest && (
                <div className="manifest-summary-grid">
                    <div className="manifest-summary-cell">
                        <span className="label">api_version</span>
                        <span className="value">{apiVersion}</span>
                    </div>
                    <div className="manifest-summary-cell">
                        <span className="label">blueprint domain</span>
                        <span className="value">{blueprintDomain}</span>
                    </div>
                    <div className="manifest-summary-cell">
                        <span className="label">data sources</span>
                        <span className="value">{dataSourceCount}</span>
                    </div>
                    <div className="manifest-summary-cell">
                        <span className="label">adapters</span>
                        <span className="value">{adaptersCount}</span>
                    </div>
                    <div className="manifest-summary-cell">
                        <span className="label">target profile</span>
                        <span className="value">{targetProfile}</span>
                    </div>
                </div>
            )}
        </section>
    );
}
