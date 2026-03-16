import React, { useEffect, useState } from 'react';
import api from '../../api/client';
import './ReadinessPanel.css';

interface ReadinessCheck {
  id: string;
  name: string;
  status: 'pass' | 'warn' | 'fail';
  message: string;
  type: 'blocker' | 'warning';
  fix?: string;
}

interface ReadinessResponse {
  project_id: number;
  status: 'pass' | 'warn' | 'fail';
  strict_mode: boolean;
  checks: ReadinessCheck[];
}

interface ReadinessPanelProps {
  projectId: number;
  refreshInterval?: number;
  className?: string;
}

const normalizeStatus = (value: unknown): 'pass' | 'warn' | 'fail' => {
  const token = String(value || '').toLowerCase();
  if (token === 'pass' || token === 'warn' || token === 'fail') {
    return token;
  }
  if (token === 'error') {
    return 'fail';
  }
  return 'warn';
};

export const ReadinessPanel: React.FC<ReadinessPanelProps> = ({
  projectId,
  refreshInterval = 30000,
  className = '',
}) => {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReadiness = async () => {
    try {
      const response = await api.get(`/projects/${projectId}/runtime/readiness`);
      const payload = response.data || {};
      const checks = Array.isArray(payload.checks) ? payload.checks : [];
      const normalizedChecks: ReadinessCheck[] = checks.map((check: any, idx: number) => ({
        id: String(check?.id || `check-${idx}`),
        name: String(check?.name || 'Readiness Check'),
        status: normalizeStatus(check?.status),
        message: String(check?.message || ''),
        type: check?.type === 'blocker' ? 'blocker' : 'warning',
        fix: check?.fix ? String(check.fix) : undefined,
      }));

      if (normalizedChecks.length === 0 && payload?.message) {
        normalizedChecks.push({
          id: 'summary',
          name: 'Readiness',
          status: normalizeStatus(payload?.status),
          message: String(payload.message),
          type: 'warning',
        });
      }

      setReadiness({
        project_id: Number(payload?.project_id || projectId),
        status: normalizeStatus(payload?.status),
        strict_mode: Boolean(payload?.strict_mode),
        checks: normalizedChecks,
      });
      setError(null);
    } catch (err) {
      console.error('Failed to fetch readiness:', err);
      setError('Failed to load system readiness status.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReadiness();
    if (refreshInterval > 0) {
      const timer = setInterval(fetchReadiness, refreshInterval);
      return () => clearInterval(timer);
    }
  }, [projectId, refreshInterval]);

  if (loading && !readiness) {
    return <div className="readiness-panel loading">Checking system readiness...</div>;
  }

  if (error && !readiness) {
    return <div className="readiness-panel error">{error}</div>;
  }

  if (!readiness) return null;
  const readinessStatus = normalizeStatus(readiness.status);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass': return '✅';
      case 'warn': return '⚠️';
      case 'fail': return '❌';
      default: return '❓';
    }
  };

  return (
    <div className={`readiness-panel ${readinessStatus} ${className}`}>
      <div className="readiness-header">
        <span className="readiness-title">System Readiness</span>
        <div className="readiness-badges">
          {readiness.strict_mode && <span className="badge strict">Strict Mode</span>}
          <span className={`badge status-${readinessStatus}`}>
            {readinessStatus.toUpperCase()}
          </span>
        </div>
      </div>

      <div className="readiness-checks">
        {readiness.checks.map((check) => (
          <div key={check.id} className={`readiness-check ${check.status}`}>
            <div className="check-main">
              <span className="check-icon">{getStatusIcon(check.status)}</span>
              <div className="check-info">
                <span className="check-name">{check.name}</span>
                <span className="check-message">{check.message}</span>
              </div>
              {check.type === 'blocker' && check.status === 'fail' && (
                <span className="check-type-tag blocker">BLOCKER</span>
              )}
            </div>
            {check.fix && (check.status === 'fail' || check.status === 'warn') && (
              <div className="check-fix">
                <strong>Fix:</strong> {check.fix}
              </div>
            )}
          </div>
        ))}
      </div>
      
      <button className="readiness-refresh" onClick={fetchReadiness}>
        Refresh Status
      </button>
    </div>
  );
};
