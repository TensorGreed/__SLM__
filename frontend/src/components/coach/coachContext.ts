/**
 * Where the user is, from the Coach's point of view: which pipeline tab
 * (for the tip + walkthrough video) and which Coach stage (for the
 * rule-based suggestions from ``GET /coach/{stage}``).
 */

import { PIPELINE_TABS } from '../../types';
import type { TabKey } from '../../types';
import type { CoachStage } from '../../api/coach';

export const TAB_TIP: Record<TabKey, string> = {
    data: 'This is where your raw examples live. Import a CSV or launch a sample dataset to get started.',
    cleaning: 'Tidy the data — drop broken rows, dedupe, normalize. One-click autofixes handle the common cases.',
    goldset: 'Build a small, trusted answer key. Evaluation scores your model against it, so the quality here matters most.',
    synthetic: 'Short on data? Generate more examples from a playbook that matches your task shape.',
    dataprep: 'Split your data into train / validation / test sets the trainer can consume.',
    tokenization: 'See how your text becomes tokens — catch truncation and out-of-vocabulary surprises before you train.',
    training: 'Pick a base model and hyperparameters on the Training Config page, then launch a run. Live metrics appear here.',
    eval: 'Score the trained model against your answer key and the built-in checks — honest numbers, pass/fail rules that can fail.',
    compression: 'Shrink the model (quantize / distill) so it fits your deployment target.',
    export: 'Package the finished model for download or deployment. You made it!',
};

const TAB_KEYS = new Set<string>(PIPELINE_TABS.map((tab) => tab.key));

const COACH_STAGE_BY_TAB: Partial<Record<TabKey, CoachStage>> = {
    data: 'data',
    cleaning: 'cleaning',
    goldset: 'gold_set',
    synthetic: 'synthetic',
    dataprep: 'dataprep',
    training: 'training',
    eval: 'eval',
    export: 'export',
};

export interface CoachLocation {
    tab: TabKey | null;
    stage: CoachStage | null;
}

/** Map a workspace URL to the tab + Coach stage it belongs to. */
export function coachLocationFor(pathname: string): CoachLocation {
    const pipeline = pathname.match(/\/pipeline\/([^/?#]+)/);
    if (pipeline && TAB_KEYS.has(pipeline[1])) {
        const tab = pipeline[1] as TabKey;
        return { tab, stage: COACH_STAGE_BY_TAB[tab] ?? null };
    }
    if (/\/training-config(\/|$)/.test(pathname)) {
        return { tab: 'training', stage: 'training' };
    }
    if (/\/data-studio(\/|$)/.test(pathname)) {
        return { tab: 'data', stage: 'data' };
    }
    return { tab: null, stage: null };
}
