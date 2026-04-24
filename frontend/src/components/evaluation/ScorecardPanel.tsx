import React, { useEffect, useState } from 'react';
import api from '../../api/client';
import './ScorecardPanel.css';

interface GateCheck {
  gate_id: string;
  metric_id: string;
  operator: string;
  threshold: number;
  required: boolean;
  actual: number | null;
  passed: boolean;
  reason?: string;
}

interface GateReport {
  passed: boolean;
  checks: GateCheck[];
  missing_required_metrics: string[];
  failed_gate_ids: string[];
}

interface Scorecard {
  experiment_id: number;
  is_ship: boolean;
  decision: 'SHIP' | 'NO-SHIP';
  reasons: string[];
  failed_gates: string[];
  missing_metrics: string[];
  gate_report: GateReport;
}

interface ScorecardPanelProps {
  projectId: number;
  experimentId: number;
}

const ScorecardPanel: React.FC<ScorecardPanelProps> = ({ projectId, experimentId }) => {
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchScorecard = async () => {
      try {
        setLoading(true);
        const response = await api.get<Scorecard>(`/projects/${projectId}/evaluation/scorecard/${experimentId}`);
        setScorecard(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to load scorecard');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchScorecard();
  }, [projectId, experimentId]);

  if (loading) return <div className="scorecard-loading">Loading scorecard...</div>;
  if (error) return <div className="scorecard-error">{error}</div>;
  if (!scorecard) return null;

  const { is_ship, decision, reasons, gate_report } = scorecard;

  return (
    <div className={`scorecard-container ${is_ship ? 'ship' : 'no-ship'}`}>
      <div className="scorecard-header">
        <div className="decision-badge">{decision}</div>
        <h2>Experiment Scorecard</h2>
      </div>

      {reasons.length > 0 && (
        <div className="scorecard-reasons">
          <h3>Blockers:</h3>
          <ul>
            {reasons.map((reason, idx) => (
              <li key={idx} className="blocker-item">{reason}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="gates-grid">
        <div className="gates-section">
          <h3>Quality Gates</h3>
          <table className="gates-table">
            <thead>
              <tr>
                <th>Gate ID</th>
                <th>Metric</th>
                <th>Target</th>
                <th>Actual</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {gate_report.checks.map((gate) => {
                const notMeasured = gate.actual === null;
                const rowClass = notMeasured
                  ? (gate.required ? 'failed' : 'warn')
                  : (gate.passed ? 'passed' : gate.required ? 'failed' : 'warn');
                let statusLabel: string;
                if (notMeasured) {
                  statusLabel = gate.required ? '⚠️ Not measured' : '⏭️ Skipped';
                } else if (gate.passed) {
                  statusLabel = '✅ Pass';
                } else {
                  statusLabel = gate.required ? '❌ Fail' : '⚠️ Low';
                }
                return (
                  <tr key={gate.gate_id} className={rowClass}>
                    <td>{gate.gate_id}</td>
                    <td>{gate.metric_id}</td>
                    <td>{gate.operator} {gate.threshold}</td>
                    <td>{notMeasured ? 'N/A' : gate.actual!.toFixed(4)}</td>
                    <td>{statusLabel}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {gate_report.missing_required_metrics.length > 0 && (
          <div className="missing-metrics-section">
            <h3>Missing Metrics</h3>
            <div className="missing-metrics-list">
              {gate_report.missing_required_metrics.map(m => (
                <div key={m} className="missing-metric-tag">{m}</div>
              ))}
            </div>
            <p className="missing-diagnostics">
              Some required metrics are missing from evaluation results. Run full evaluation suite to clear.
            </p>
          </div>
        )}
      </div>

      <div className="scorecard-footer">
        <p>Decisions are deterministic and reproducible based on project gate policy.</p>
      </div>
    </div>
  );
};

export default ScorecardPanel;
