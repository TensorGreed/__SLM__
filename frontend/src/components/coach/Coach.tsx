/**
 * The Coach — the one guidance surface (Wave 3).
 *
 * Mounted once in the project workspace layout, above every page. It
 * replaces the flow hint, the guided-learning rail, the getting-started
 * overlay, the per-tab video strip and the per-panel Coach strips:
 *
 *   - where you are (stage + progress) and the single next step,
 *   - a short tip + walkthrough video for the tab you're on,
 *   - the rule-based suggestions for this stage (``GET /coach/{stage}``),
 *   - a link to the full plan (the project home: quickstart + checklist).
 *
 * Off when Coach Mode is toggled off for the project (top-bar toggle).
 */

import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import type { PipelineStatusResponse, Project } from '../../types';
import { getRecommendedAction, PIPELINE_STAGE_LABEL } from '../../utils/flowGuide';
import TabVideoLink from '../video/TabVideoLink';
import CoachStrip from './CoachStrip';
import { coachLocationFor, TAB_TIP } from './coachContext';
import { useCoachMode } from './useCoachMode';
import './Coach.css';

interface CoachProps {
    projectId: number;
    project: Project;
    pipelineStatus: PipelineStatusResponse | null;
}

function tipKey(projectId: number, tab: string): string {
    return `brewslm.coach.tip.${projectId}.${tab}`;
}

function readFlag(key: string): boolean {
    try {
        return window.localStorage.getItem(key) === '1';
    } catch {
        return false;
    }
}

function writeFlag(key: string): void {
    try {
        window.localStorage.setItem(key, '1');
    } catch {
        // private mode / quota — the tip just shows again next time
    }
}

export default function Coach({ projectId, project, pipelineStatus }: CoachProps) {
    const location = useLocation();
    const navigate = useNavigate();
    const { isOn } = useCoachMode(projectId, project.beginner_mode);
    const { tab, stage } = useMemo(() => coachLocationFor(location.pathname), [location.pathname]);
    const next = useMemo(
        () => getRecommendedAction(projectId, project, pipelineStatus),
        [projectId, project, pipelineStatus],
    );
    const [tipDismissed, setTipDismissed] = useState(true);

    useEffect(() => {
        setTipDismissed(tab ? readFlag(tipKey(projectId, tab)) : true);
    }, [projectId, tab]);

    // The project home IS the Coach's full view.
    if (!isOn || location.pathname.endsWith('/guide')) {
        return null;
    }

    const currentStage = pipelineStatus?.current_stage || project.pipeline_stage;
    const onNextTarget = location.pathname.startsWith(next.path.split('?')[0]);
    const tip = tab ? TAB_TIP[tab] : null;

    return (
        <section className="card coach-bar" data-testid="coach-bar" aria-label="Coach">
            <div className="coach-bar__row">
                <span className="coach-bar__badge">Coach</span>
                <span className="coach-bar__where" data-testid="coach-where">
                    {PIPELINE_STAGE_LABEL[currentStage] ?? currentStage}
                    {' · '}
                    {pipelineStatus?.progress_percent ?? 0}% done
                </span>
                <span className="coach-bar__next" data-testid="coach-next">
                    Next: <strong>{next.title}</strong> — {next.description}
                </span>
                <span className="coach-bar__actions">
                    {!onNextTarget && (
                        <button
                            type="button"
                            className="btn btn-primary btn-sm"
                            onClick={() => navigate(next.path)}
                            data-testid="coach-continue"
                        >
                            Continue →
                        </button>
                    )}
                    <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => navigate(`/project/${projectId}/guide`)}
                        data-testid="coach-plan"
                    >
                        Full plan
                    </button>
                </span>
            </div>
            {tab && tip && !tipDismissed && (
                <div className="coach-bar__tip" data-testid="coach-tip">
                    <span>💡 {tip}</span>
                    <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        aria-label="Dismiss tip"
                        onClick={() => {
                            writeFlag(tipKey(projectId, tab));
                            setTipDismissed(true);
                        }}
                    >
                        Got it
                    </button>
                </div>
            )}
            {tab && <TabVideoLink tabKey={tab} />}
            {stage && <CoachStrip projectId={projectId} stage={stage} beginnerMode={project.beginner_mode} />}
        </section>
    );
}
