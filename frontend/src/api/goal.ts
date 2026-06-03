/**
 * Arc H — Project end-goal contract + progress ledger client.
 *
 * The goal is one shape: a target metric (f1 / pass_rate / accuracy),
 * a threshold (0-1), an optional deadline, and an optional title.
 * Coach + Data Studio compute the progress ledger from the project's
 * current state vs. that target.
 */

import api from './client';


export type GoalMetric = 'f1' | 'pass_rate' | 'accuracy';
export type GoalLedgerStatus = 'ready_to_ship' | 'in_progress' | 'blocked';
export type GoalComponentStatus = 'met' | 'attention' | 'pending';

export interface ProjectGoal {
    target_metric: GoalMetric;
    target_threshold: number;
    deadline: string | null;
    title: string | null;
    /** ISO timestamp; server-side stamp at PUT time. Null on default
     *  fallback (when the project has never had a goal set). */
    stated_at: string | null;
}

export interface GoalProgressComponent {
    id: 'data_ready' | 'gold_set' | 'predicted_pass' | 'eval_pass_rate' | string;
    label: string;
    /** 0.0–1.0 fraction toward this component's target. Null when
     *  the component hasn't been computed yet (e.g. no eval has run). */
    value: number | null;
    status: GoalComponentStatus;
    /** Short human-facing detail line ("12 gold rows · 100 recommended"). */
    detail: string;
    /** Concept id in the frontend Term registry. The card renders
     *  ``<Term id={concept_id}>`` so each component carries an inline
     *  "Learn more on BrewSLM Academy" link (Arc G). */
    concept_id: string;
}

export interface GoalProgressResponse {
    project_id: number;
    /** Always present; backend falls back to ``f1 ≥ 0.70`` when the
     *  user hasn't stated a goal yet. ``has_explicit_goal`` tells the
     *  UI whether to nudge for goal-setting. */
    goal: ProjectGoal;
    has_explicit_goal: boolean;
    components: GoalProgressComponent[];
    /** Equal-weight mean over known component values. 0-1. */
    overall_progress: number;
    /** Component IDs whose value is still null. */
    pending_components: string[];
    /** Short human-facing blocker strings for surfacing in the UI. */
    blockers: string[];
    status: GoalLedgerStatus;
}


export async function getProjectGoalProgress(
    projectId: number,
): Promise<GoalProgressResponse> {
    const resp = await api.get(`/projects/${projectId}/goal/progress`);
    return resp.data as GoalProgressResponse;
}


export interface SetGoalArgs {
    targetMetric: GoalMetric;
    targetThreshold: number;
    deadline?: string | null;
    title?: string | null;
}


export async function setProjectGoal(
    projectId: number,
    args: SetGoalArgs,
): Promise<{ project_id: number; goal: ProjectGoal }> {
    const resp = await api.put(`/projects/${projectId}/goal`, {
        target_metric: args.targetMetric,
        target_threshold: args.targetThreshold,
        deadline: args.deadline ?? null,
        title: args.title ?? null,
    });
    return resp.data as { project_id: number; goal: ProjectGoal };
}


export async function clearProjectGoal(
    projectId: number,
): Promise<void> {
    await api.delete(`/projects/${projectId}/goal`);
}
