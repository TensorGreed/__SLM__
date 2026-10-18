/**
 * One-line help under a step bar (Epic U, phase U6): the step's tip and its
 * walkthrough video, dismissible per project and step. What the Coach bar's
 * tip strip used to be.
 */

import { useState } from 'react';

import type { TabKey } from '../../types';
import TabVideoLink from '../video/TabVideoLink';
import { useCoachMode } from './useCoachMode';
import { TAB_TIP } from './coachContext';

interface StepHelpProps {
    projectId: number;
    tab: TabKey;
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

export default function StepHelp({ projectId, tab }: StepHelpProps) {
    const { isOn } = useCoachMode(projectId);
    const [dismissed, setDismissed] = useState(() => readFlag(tipKey(projectId, tab)));
    const [seenTab, setSeenTab] = useState(tab);
    if (seenTab !== tab) {
        setSeenTab(tab);
        setDismissed(readFlag(tipKey(projectId, tab)));
    }
    if (!isOn || dismissed) return null;
    const tip = TAB_TIP[tab];
    if (!tip) return null;
    return (
        <div className="step-help" data-testid={`step-help-${tab}`}>
            <span className="step-help__tip">{tip}</span>
            <TabVideoLink tabKey={tab} />
            <button
                type="button"
                className="btn btn-ghost btn-sm"
                aria-label="Dismiss tip"
                onClick={() => {
                    try {
                        window.localStorage.setItem(tipKey(projectId, tab), '1');
                    } catch {
                        // ignore
                    }
                    setDismissed(true);
                }}
            >
                Got it
            </button>
        </div>
    );
}
