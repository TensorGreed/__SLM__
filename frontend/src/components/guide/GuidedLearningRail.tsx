/**
 * GuidedLearningRail — Guided Learning Mode contextual band (Epic G phase G3).
 *
 * A thin orchestration layer over pieces that already exist (the pipeline
 * tab sequence, `project.beginner_mode`, the `goToNextTab` advance handler).
 * Rendered under the pipeline tab bar and gated on beginner mode, it gives a
 * first-timer three things on every tab:
 *
 *   1. a plain-language **checkpoint line** — "Guided step N of M · <stage>" —
 *      so they always know where they are in the data → train → eval → ship arc
 *      (complements the visual `PipelineProgress` bar, doesn't duplicate it);
 *   2. a **first-visit contextual tip** explaining what the current stage is
 *      for, dismissable per (project, tab) via localStorage so it never nags;
 *   3. a **"next step" CTA** that advances to the next stage (reusing the same
 *      `goToNextTab` the panels already call), so the path is one click forward.
 *
 * Advanced users (beginner_mode off) see none of this — the component renders
 * null. No new API surface, no new types.
 */

import { useEffect, useState } from 'react';
import { GraduationCap, Lightbulb, ArrowRight, X } from 'lucide-react';

import { PIPELINE_TABS } from '../../types';
import type { Project, TabKey } from '../../types';
import './GuidedLearningRail.css';

interface GuidedLearningRailProps {
    project: Project | null | undefined;
    activeTab: TabKey;
    onNextStep: () => void;
}

/** Per-tab newbie orientation: what this stage is for, in one line. */
const TAB_TIP: Record<TabKey, string> = {
    data: 'This is where your raw examples live. Import a CSV or launch a sample dataset to get started.',
    cleaning: 'Tidy the data — drop broken rows, dedupe, normalize. One-click autofixes handle the common cases.',
    goldset: 'Build a small, trusted answer key. Evaluation scores your model against it, so the quality here matters most.',
    synthetic: 'Short on data? Generate more examples from a playbook that matches your task shape.',
    dataprep: 'Split your data into train / validation / test sets the trainer can consume.',
    tokenization: 'See how your text becomes tokens — catch truncation and out-of-vocabulary surprises before you train.',
    training: 'Pick a base model and hyperparameters on the Training Config page, then launch a run. Live metrics appear here.',
    eval: 'Score the trained model against your gold set and the independent probe pack — honest numbers, gates that can fail.',
    compression: 'Shrink the model (quantize / distill) so it fits your deployment target.',
    export: 'Package the finished model for download or deployment. You made it!',
};

export function guidedTipDismissKey(projectId: number, tab: TabKey): string {
    return `brewslm.guided.tip.${projectId}.${tab}`;
}

function readTipDismissed(projectId: number, tab: TabKey): boolean {
    try {
        return window.localStorage.getItem(guidedTipDismissKey(projectId, tab)) === '1';
    } catch {
        return false;
    }
}

function writeTipDismissed(projectId: number, tab: TabKey): void {
    try {
        window.localStorage.setItem(guidedTipDismissKey(projectId, tab), '1');
    } catch {
        // ignore storage failures (private mode, quota)
    }
}

export default function GuidedLearningRail({ project, activeTab, onNextStep }: GuidedLearningRailProps) {
    // `null` until the first-visit check runs on mount, so we never flash the
    // tip before reading localStorage.
    const [tipDismissed, setTipDismissed] = useState<boolean | null>(null);

    const projectId = project?.id ?? null;

    useEffect(() => {
        if (projectId == null) {
            return;
        }
        setTipDismissed(readTipDismissed(projectId, activeTab));
    }, [projectId, activeTab]);

    // Guided Learning Mode is beginner-only orchestration.
    if (!project || !project.beginner_mode) {
        return null;
    }

    const stepIndex = PIPELINE_TABS.findIndex((t) => t.key === activeTab);
    const total = PIPELINE_TABS.length;
    const current = PIPELINE_TABS[stepIndex >= 0 ? stepIndex : 0];
    const nextTab = stepIndex >= 0 && stepIndex < total - 1 ? PIPELINE_TABS[stepIndex + 1] : null;
    const tip = TAB_TIP[activeTab];

    const handleDismissTip = () => {
        if (projectId != null) {
            writeTipDismissed(projectId, activeTab);
        }
        setTipDismissed(true);
    };

    return (
        <section className="guided-rail" role="region" aria-label="Guided Learning Mode">
            <div className="guided-rail-head">
                <span className="guided-rail-badge">
                    <GraduationCap size={14} aria-hidden="true" />
                    Guided
                </span>
                <span className="guided-rail-step">
                    Step {Math.max(stepIndex, 0) + 1} of {total} · <strong>{current.label}</strong>
                </span>
                {nextTab ? (
                    <button
                        type="button"
                        className="btn btn-primary btn-sm guided-rail-next"
                        onClick={onNextStep}
                    >
                        Next: {nextTab.label} <ArrowRight size={14} aria-hidden="true" />
                    </button>
                ) : (
                    <span className="guided-rail-final">Final step 🎉</span>
                )}
            </div>
            {tip && tipDismissed === false && (
                <div className="guided-rail-tip">
                    <Lightbulb size={14} aria-hidden="true" className="guided-rail-tip-icon" />
                    <p className="guided-rail-tip-text">{tip}</p>
                    <button
                        type="button"
                        className="guided-rail-tip-dismiss"
                        onClick={handleDismissTip}
                        aria-label={`Dismiss the ${current.label} tip`}
                    >
                        <X size={14} aria-hidden="true" />
                    </button>
                </div>
            )}
        </section>
    );
}
