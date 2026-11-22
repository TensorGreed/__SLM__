/**
 * One-line help under a step bar (Epic U, phase U6): the step's tip and its
 * walkthrough video. What the Coach bar's tip strip used to be.
 *
 * UI simplification phase 6 (decision 4): the tip shows in full on the first
 * visit to a step only; afterwards (or after "Got it") it is a "?" line that
 * opens the tip for that visit. Remembered per project and step in
 * localStorage: ``brewslm.coach.tip.<pid>.<step>`` = "Got it",
 * ``brewslm.coach.tip.seen.<pid>.<step>`` = shown once (written only after the
 * full tip has been on screen for ``STEP_HELP_SEEN_AFTER_MS``, so a quick
 * remount during route churn does not use up the first visit). Every access
 * is wrapped; with storage blocked the tip shows in full every time.
 */

import { useEffect, useState } from 'react';

import type { TabKey } from '../../types';
import TabVideoLink from '../video/TabVideoLink';
import { useCoachMode } from './useCoachMode';
import { SUB_STEP_TIP, TAB_TIP } from './coachContext';
import './StepHelp.css';

interface StepHelpProps {
    projectId: number;
    tab: TabKey;
    /** The active Results tab / Ship stage, so the tip and video chapter follow it. */
    sub?: string;
}

/** How long the full tip must stay on screen before the step counts as seen. */
export const STEP_HELP_SEEN_AFTER_MS = 1500;

function tipKey(projectId: number, step: string): string {
    return `brewslm.coach.tip.${projectId}.${step}`;
}

function seenKey(projectId: number, step: string): string {
    return `brewslm.coach.tip.seen.${projectId}.${step}`;
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
        // Storage blocked: the tip shows in full next time too.
    }
}

export default function StepHelp({ projectId, tab, sub }: StepHelpProps) {
    const subTip = sub ? SUB_STEP_TIP[tab]?.[sub] : undefined;
    const step = subTip ? `${tab}.${sub}` : tab;
    const dismissKey = tipKey(projectId, step);
    const firstVisitKey = seenKey(projectId, step);
    const { isOn } = useCoachMode(projectId);
    const readCollapsed = () => readFlag(dismissKey) || readFlag(firstVisitKey);
    const [collapsed, setCollapsed] = useState(readCollapsed);
    const [shownKey, setShownKey] = useState(dismissKey);
    if (shownKey !== dismissKey) {
        setShownKey(dismissKey);
        setCollapsed(readCollapsed());
    }
    const tip = subTip ?? TAB_TIP[tab];
    const visible = isOn && Boolean(tip);

    // A full showing that stays up a moment counts as the visit: from the next
    // one on, the tip is a "?" line. Unmounting earlier cancels it.
    useEffect(() => {
        if (!visible || collapsed) return undefined;
        const timer = window.setTimeout(() => writeFlag(firstVisitKey), STEP_HELP_SEEN_AFTER_MS);
        return () => window.clearTimeout(timer);
    }, [visible, collapsed, firstVisitKey]);

    if (!visible) return null;
    if (collapsed) {
        return (
            <div className="step-help step-help--collapsed" data-testid={`step-help-${tab}`}>
                <button
                    type="button"
                    className="step-help__open"
                    onClick={() => setCollapsed(false)}
                    aria-label="Show the tip for this step"
                    title="Show the tip for this step"
                    data-testid={`step-help-${tab}-open`}
                >
                    ?
                </button>
            </div>
        );
    }
    return (
        <div className="step-help" data-testid={`step-help-${tab}`}>
            <span className="step-help__tip">{tip}</span>
            <TabVideoLink tabKey={tab} subStep={subTip ? sub : undefined} />
            <button
                type="button"
                className="btn btn-ghost btn-sm"
                aria-label="Dismiss tip"
                onClick={() => {
                    writeFlag(dismissKey);
                    setCollapsed(true);
                }}
            >
                Got it
            </button>
        </div>
    );
}
