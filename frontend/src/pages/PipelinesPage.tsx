/**
 * Pipelines — list of multi-stage evaluation pipelines (chains of projects
 * scored per stage and end to end), with "New pipeline".
 */

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

import { listPipelines, percent, runStatusLabel, type PipelineSummary } from '../api/pipelines';
import TopBar from '../components/layout/TopBar';
import '../components/pipelines/pipelines.css';

export default function PipelinesPage() {
    const [items, setItems] = useState<PipelineSummary[] | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        listPipelines().then(setItems).catch(() => { setItems([]); setError('Could not load pipelines'); });
    }, []);

    return (
        <div className="main-content">
            <TopBar title="Pipelines" subtitle="Chain projects and score them end to end" />
            <div className="page-container" data-testid="pipelines-page">
                <div className="pipelines__page-head">
                    <Link to="/" className="pipelines__back">Back to projects</Link>
                    <Link to="/pipelines/new" className="btn btn-primary" data-testid="pipeline-new">New pipeline</Link>
                </div>
                {error && <p className="pipelines__error" role="alert">{error}</p>}
                {items && items.length === 0 && !error && (
                    <p className="pipelines__note" data-testid="pipelines-empty">No pipelines yet. A pipeline chains projects, for example one that extracts items and one that rewrites each item.</p>
                )}
                <ul className="pipelines__cards">
                    {(items ?? []).map((p) => (
                        <li key={p.id} className="pipelines__card" data-testid={`pipeline-item-${p.id}`}>
                            <h4><Link to={`/pipelines/${p.id}`}>{p.name}</Link></h4>
                            <p className="pipelines__note">
                                {p.stages.map((s, i) => (
                                    <span key={s.name}>
                                        {i > 0 ? ' → ' : ''}{s.name} (#{s.project_id}{s.project_name ? ` ${s.project_name}` : ''})
                                        {s.project_missing && (
                                            <span className="pipelines__badge" data-testid={`pipeline-missing-${p.id}-${s.name}`}>project missing</span>
                                        )}
                                    </span>
                                ))}
                            </p>
                            <p className="pipelines__note">
                                {p.heldout_rows} held-out row{p.heldout_rows === 1 ? '' : 's'}
                                {!p.last_run ? ' · never run'
                                    : (p.last_run.status ?? 'done') === 'done'
                                        ? ` · last run ${percent(p.last_run.share_fully_correct)} fully correct`
                                        : ` · last run ${runStatusLabel(p.last_run.status)}`}
                            </p>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}
