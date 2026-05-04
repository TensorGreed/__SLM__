/**
 * ProjectDeploymentsPage — Deployment Assistant (priority.md P30).
 *
 * Single-page surface for the Wave F deployment plane:
 *   - Versions list with promote / reject / rollback (P25).
 *   - Deployability score card with per-component provenance (P28).
 *   - Live telemetry KPIs + percentile chart (P26).
 *   - Drift verdict + history + offline launcher (P27).
 *
 * Endpoints used:
 *   GET /api/projects/{id}/deployments
 *   POST /api/deployments/{id}/{promote,reject,rollback}
 *   GET /api/deployments/{id}/score
 *   POST /api/deployments/{id}/score/compute
 *   GET /api/deployments/{id}/telemetry
 *   GET /api/deployments/{id}/drift/checks
 *   POST /api/deployments/{id}/drift/check
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';

import api from '../api/client';
import DeployabilityScoreCard from '../components/deployment/DeployabilityScoreCard';
import DeployedVersionsList from '../components/deployment/DeployedVersionsList';
import DriftPanel from '../components/deployment/DriftPanel';
import TelemetryPanel from '../components/deployment/TelemetryPanel';
import type {
    DeploymentVersion,
    DeploymentVersionListResponse,
} from '../types/deployment';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';

import './ProjectDeploymentsPage.css';

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

function pickPreferredDeployment(
    versions: DeploymentVersion[],
    current: number | null,
): number | null {
    if (current != null && versions.some((v) => v.id === current)) return current;
    const promoted = versions.find((v) => v.status === 'promoted');
    if (promoted) return promoted.id;
    const pending = versions.find((v) => v.status === 'pending');
    if (pending) return pending.id;
    return versions[0]?.id ?? null;
}

export default function ProjectDeploymentsPage() {
    const params = useParams();
    const workspace = useOutletContext<ProjectWorkspaceContextValue | null>();
    const routeProjectId = params.id ? Number.parseInt(params.id, 10) : null;
    const projectId =
        workspace?.projectId ?? (Number.isFinite(routeProjectId) ? routeProjectId : null);

    const [versions, setVersions] = useState<DeploymentVersion[]>([]);
    const [selectedDeploymentId, setSelectedDeploymentId] = useState<number | null>(
        null,
    );
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchVersions = useCallback(async () => {
        if (projectId == null) return;
        setLoading(true);
        try {
            const response = await api.get<DeploymentVersionListResponse>(
                `/projects/${projectId}/deployments`,
            );
            const list = response.data.deployment_versions || [];
            setVersions(list);
            setSelectedDeploymentId((current) => pickPreferredDeployment(list, current));
            setError(null);
        } catch (err) {
            setError(extractErrorMessage(err, 'Failed to load deployment versions.'));
        } finally {
            setLoading(false);
        }
    }, [projectId]);

    useEffect(() => {
        void fetchVersions();
    }, [fetchVersions]);

    const counts = useMemo(() => {
        const total = versions.length;
        const promoted = versions.filter((v) => v.status === 'promoted').length;
        const pending = versions.filter((v) => v.status === 'pending').length;
        return { total, promoted, pending };
    }, [versions]);

    if (projectId == null) {
        return (
            <div className="workspace-page deployments-page">
                <div className="deployment-status is-error" role="alert">
                    Project context is not available.
                </div>
            </div>
        );
    }

    return (
        <div className="workspace-page deployments-page">
            <section className="workspace-page-header">
                <div>
                    <h2 className="workspace-page-title">Deployment Assistant</h2>
                    <p className="workspace-page-subtitle">
                        Track served versions, promote / reject / rollback, and watch
                        telemetry, drift, and the deployability score in one place.
                    </p>
                </div>
                <div className="deployment-summary-counts">
                    <span className="badge badge-info">total: {counts.total}</span>
                    <span className="badge badge-success">promoted: {counts.promoted}</span>
                    <span className="badge badge-warning">pending: {counts.pending}</span>
                </div>
            </section>

            {error && (
                <div className="deployment-status is-error" role="alert">
                    {error}
                </div>
            )}

            <section className="card">
                <div className="deployment-section-header">
                    <h3 style={{ marginTop: 0 }}>Served versions</h3>
                    <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => void fetchVersions()}
                        disabled={loading}
                    >
                        {loading ? 'Refreshing…' : 'Refresh'}
                    </button>
                </div>
                <DeployedVersionsList
                    versions={versions}
                    selectedDeploymentId={selectedDeploymentId}
                    onSelect={setSelectedDeploymentId}
                    onRefresh={fetchVersions}
                />
            </section>

            {selectedDeploymentId != null && (
                <>
                    <DeployabilityScoreCard deploymentVersionId={selectedDeploymentId} />
                    <TelemetryPanel deploymentVersionId={selectedDeploymentId} />
                    <DriftPanel deploymentVersionId={selectedDeploymentId} />
                </>
            )}
        </div>
    );
}
