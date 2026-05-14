/**
 * CRT achievement toast contract.
 *
 * Pins:
 * - 'achievement' toasts go through AchievementToast, not the
 *   standard pill renderer.
 * - Level-up payload shows the level number + LEVEL UP header.
 * - +XP delta renders when present; omitted otherwise.
 * - Close button removes the toast from the store.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import ToastContainer from '../shared/Toast';
import {
    useToastStore,
    type AchievementToastPayload,
} from '../../stores/toastStore';

function pushAchievement(payload: AchievementToastPayload) {
    useToastStore.getState().addAchievementToast(payload, 0);
}

describe('AchievementToast', () => {
    beforeEach(() => {
        useToastStore.setState({ toasts: [] });
    });
    afterEach(() => {
        useToastStore.setState({ toasts: [] });
    });

    it('renders the achievement title, description, and XP delta', () => {
        pushAchievement({
            kind: 'achievement',
            title: 'First Forge',
            description: 'Trained your first model to completion.',
            xp_awarded: 200,
            tier: 'onboarding',
        });
        render(<ToastContainer />);
        expect(screen.getByText('First Forge')).toBeInTheDocument();
        expect(
            screen.getByText('Trained your first model to completion.'),
        ).toBeInTheDocument();
        expect(screen.getByText(/\+200 XP/)).toBeInTheDocument();
        // CRT frame markers present.
        expect(screen.getByText(/ACHIEVEMENT UNLOCKED/)).toBeInTheDocument();
    });

    it('renders level-up toasts with LEVEL UP header + level number', () => {
        pushAchievement({
            kind: 'level_up',
            title: 'Senior',
            level_after: 5,
        });
        render(<ToastContainer />);
        expect(screen.getByText(/LEVEL UP/)).toBeInTheDocument();
        expect(screen.getByText(/Level 5 — Senior/)).toBeInTheDocument();
    });

    it('renders both the achievement title and a "LV X reached" line when level changed', () => {
        pushAchievement({
            kind: 'achievement',
            title: 'Decent Hacker',
            description: 'Eval pass rate crossed 80%.',
            xp_awarded: 200,
            level_after: 3,
        });
        render(<ToastContainer />);
        expect(screen.getByText('Decent Hacker')).toBeInTheDocument();
        expect(screen.getByText(/LV 3 reached/)).toBeInTheDocument();
    });

    it('omits the XP line when xp_awarded is missing', () => {
        pushAchievement({
            kind: 'level_up',
            title: 'Principal',
            level_after: 10,
        });
        render(<ToastContainer />);
        expect(screen.queryByText(/\+0 XP/)).not.toBeInTheDocument();
        // No +N XP line at all.
        const xpHits = screen.queryAllByText(/\+\d+ XP/);
        expect(xpHits).toHaveLength(0);
    });

    it('the close button removes the toast from the store', async () => {
        pushAchievement({
            kind: 'achievement',
            title: 'Quantized',
            xp_awarded: 200,
        });
        render(<ToastContainer />);
        const closeBtn = screen.getByLabelText('Close');
        const user = userEvent.setup();
        await user.click(closeBtn);
        // Wait for the 200ms fade-out timer that lives inside the
        // component; the store update happens after.
        await new Promise((r) => setTimeout(r, 250));
        expect(useToastStore.getState().toasts.length).toBe(0);
    });
});
