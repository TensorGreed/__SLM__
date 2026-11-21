/**
 * The project's scoring settings, in one place: Results › Verdict › How it's scored › "Change how it's
 * scored" (UI simplification pass, phase 4; decision 5). The same cards Train › More options held
 * (tool-call scoring, plan schema and rubric, remediation checks, SQL execution match, the automatic
 * base-vs-fine-tuned check), plus the project's null-as-absent switch. Same endpoints and bodies; the
 * old Train ``?open=`` links redirect here (`ProjectTrainPage`).
 *
 * Which task-specific cards show follows the old Train rules: tool-call scoring for tool-calling and
 * structured-plan projects, plan rubric for structured-plan, remediation checks for code remediation.
 * The task the project evaluates as comes from the backend (`effective_task_profile`).
 */

import { useEffect, useState, type ReactNode } from 'react';

import { fetchNullAsAbsent, saveNullAsAbsent } from '../../api/nullScoring';
import { getToolScoringConfig } from '../../api/toolScoring';
import { useProjectStore } from '../../stores/projectStore';
import AutoLiftSettingCard from '../training/AutoLiftSettingCard';
import RemediationConfigCard from '../training/RemediationConfigCard';
import SqlExecutionConfigCard from '../training/SqlExecutionConfigCard';
import StructuredPlanConfigCard from '../training/StructuredPlanConfigCard';
import ToolScoringConfigCard from '../training/ToolScoringConfigCard';

/** The settings this section holds; each is ``?open=how-scored-settings&setting=<id>``. */
export const SCORING_SETTING_IDS = ['tool-scoring', 'structured-plan', 'remediation', 'null-as-absent', 'sql-execution', 'auto-lift'] as const;
export type ScoringSettingId = (typeof SCORING_SETTING_IDS)[number];

export function isScoringSettingId(value: string | null | undefined): value is ScoringSettingId {
    return !!value && (SCORING_SETTING_IDS as readonly string[]).includes(value);
}

/** The project-wide "null counts as absent" switch (the eval form keeps the per-eval override). */
export function NullAsAbsentProjectSetting({ projectId }: { projectId: number }) {
    const [enabled, setEnabled] = useState<boolean | null>(null);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    useEffect(() => {
        let alive = true;
        fetchNullAsAbsent(projectId)
            .then((s) => { if (alive) setEnabled(Boolean(s.enabled)); })
            .catch(() => { if (alive) setError('Could not load the setting.'); });
        return () => { alive = false; };
    }, [projectId]);
    const save = async (next: boolean) => {
        setSaving(true);
        setError(null);
        setEnabled(next);
        try {
            const s = await saveNullAsAbsent(projectId, next ? true : null);
            setEnabled(Boolean(s.enabled));
        } catch {
            setError('Could not save the setting.');
        } finally {
            setSaving(false);
        }
    };
    return (
        <div data-testid="null-as-absent-project-card">
            <p className="u-hint">
                For answers in JSON (extraction, tool-call arguments): count <code>&quot;key&quot;: null</code> the same
                as the key being left out when comparing with the answer key. Off by default. Results scored with it are
                labelled, and Compare and the lift check do not pair them with results scored without it. One eval can
                override it under Score this run › Advanced.
            </p>
            {enabled !== null && (
                <label className="eval-panel__toggle">
                    <input
                        type="checkbox"
                        checked={enabled}
                        disabled={saving}
                        onChange={(e) => void save(e.target.checked)}
                        data-testid="null-as-absent-project-toggle"
                    />
                    Project setting: null counts as absent
                </label>
            )}
            {error && <p className="u-hint u-mt0" data-testid="null-as-absent-project-error">{error}</p>}
        </div>
    );
}

interface Props {
    projectId: number;
    /** ``?setting=`` — scroll to that card. */
    focus?: string | null;
}

export default function ScoringSettingsSection({ projectId, focus = null }: Props) {
    const project = useProjectStore((s) => s.activeProject);
    const recipe = (project && project.id === projectId ? project.selected_recipe : null) as
        | { recipe_id?: string; task_profile?: string } | null | undefined;
    const [effectiveProfile, setEffectiveProfile] = useState<string | null | undefined>(undefined);
    useEffect(() => {
        let alive = true;
        getToolScoringConfig(projectId)
            .then((res) => { if (alive) setEffectiveProfile(res.effective_task_profile ?? null); })
            .catch(() => { if (alive) setEffectiveProfile(undefined); });
        return () => { alive = false; };
    }, [projectId]);
    const taskProfile = String(effectiveProfile ?? recipe?.task_profile ?? '');
    const isStructuredPlan = recipe?.recipe_id === 'structured-plan';
    const isRemediation = recipe?.recipe_id === 'code-remediation';
    const isToolCalling = taskProfile === 'tool_calling';

    useEffect(() => {
        if (!isScoringSettingId(focus)) return;
        const timer = window.setTimeout(() => {
            document.getElementById(`how-scored-setting-${focus}`)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
        }, 80);
        return () => window.clearTimeout(timer);
    }, [focus, isToolCalling, isStructuredPlan]);

    const block = (id: ScoringSettingId, title: string, body: ReactNode) => (
        <div className="how-scored__setting" id={`how-scored-setting-${id}`} data-testid={`how-scored-setting-${id}`}>
            <h4 className="u-m0">{title}</h4>
            {body}
        </div>
    );

    return (
        <div className="how-scored__settings" data-testid="how-scored-settings-body">
            <p className="u-hint u-m0">Project settings. New settings apply to evals run after the change.</p>
            {(isToolCalling || isStructuredPlan) && block('tool-scoring', 'Tool-call scoring', <ToolScoringConfigCard projectId={projectId} isPlan={isStructuredPlan} />)}
            {isStructuredPlan && block('structured-plan', 'Plan schema and rubric', <StructuredPlanConfigCard projectId={projectId} />)}
            {isRemediation && block('remediation', 'Remediation checks', <RemediationConfigCard projectId={projectId} />)}
            {block('null-as-absent', 'Null counts as absent', <NullAsAbsentProjectSetting projectId={projectId} />)}
            {block('sql-execution', 'SQL execution match', <SqlExecutionConfigCard projectId={projectId} />)}
            {block('auto-lift', 'Automatic base-vs-fine-tuned check', <AutoLiftSettingCard projectId={projectId} />)}
        </div>
    );
}
