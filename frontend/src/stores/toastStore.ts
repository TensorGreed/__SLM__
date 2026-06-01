/**
 * Zustand store for toast notifications with optional achievement payload and auto-dismiss timers.
 */

import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'info' | 'warning' | 'achievement';

/**
 * Optional payload for the 'achievement' toast variant — the Lab
 * Journal (gamification) pushes one of these on every newly-unlocked
 * achievement or level-up. Standard toasts (success/error/info/
 * warning) leave it undefined and render plain text.
 */
export interface AchievementToastPayload {
    kind: 'achievement' | 'level_up';
    title: string;
    description?: string;
    xp_awarded?: number;
    level_after?: number | null;
    tier?: 'onboarding' | 'mastery' | 'discovery';
}

/**
 * Optional one-click action attached to a toast — used by flows
 * that want to close a workflow loop without making the user
 * context-switch. Example: the bell kill-switch's cancel+clone
 * path attaches a "Start retry now" action so the user can
 * launch the cloned experiment from the toast itself instead of
 * navigating to the training panel and finding the row.
 *
 * Click semantics: the toast removes itself after ``onClick``
 * resolves (the action is the dismissal), so the renderer doesn't
 * need a separate close. ``onClick`` is fire-and-forget; if it
 * throws the toast still dismisses so the user isn't stuck with
 * a stale action.
 */
export interface ToastAction {
    label: string;
    onClick: () => void | Promise<void>;
}

export interface ToastMessage {
    id: string;
    type: ToastType;
    message: string;
    payload?: AchievementToastPayload;
    action?: ToastAction;
}

interface ToastState {
    toasts: ToastMessage[];
    addToast: (
        message: string,
        type?: ToastType,
        duration?: number,
        action?: ToastAction,
    ) => void;
    addAchievementToast: (
        payload: AchievementToastPayload,
        duration?: number,
    ) => void;
    removeToast: (id: string) => void;
}

export const useToastStore = create<ToastState>((set) => ({
    toasts: [],
    addToast: (message, type = 'info', duration = 3000, action) => {
        const id = Math.random().toString(36).substring(2, 9);
        set((state) => ({
            toasts: [
                ...state.toasts,
                action ? { id, type, message, action } : { id, type, message },
            ],
        }));

        if (duration > 0) {
            setTimeout(() => {
                set((state) => ({ toasts: state.toasts.filter(t => t.id !== id) }));
            }, duration);
        }
    },
    addAchievementToast: (payload, duration = 5500) => {
        // Achievement toasts linger longer than standard ones (5.5s
        // vs 3s) — they're celebratory and the user usually wants to
        // read the title + XP delta before they fade.
        const id = Math.random().toString(36).substring(2, 9);
        set((state) => ({
            toasts: [
                ...state.toasts,
                { id, type: 'achievement', message: payload.title, payload },
            ],
        }));
        if (duration > 0) {
            setTimeout(() => {
                set((state) => ({ toasts: state.toasts.filter(t => t.id !== id) }));
            }, duration);
        }
    },
    removeToast: (id) => set((state) => ({ toasts: state.toasts.filter(t => t.id !== id) })),
}));

// Helper function for non-React files. The optional ``action``
// parameter renders as a button inside the toast pill — clicking
// it runs ``onClick`` and dismisses the toast. Used by flows
// like the bell kill-switch ("Start retry now" after cancel+
// clone) that want to close a multi-step UX loop in-toast.
export const toast = {
    success: (msg: string, dur?: number, action?: ToastAction) =>
        useToastStore.getState().addToast(msg, 'success', dur, action),
    error: (msg: string, dur?: number, action?: ToastAction) =>
        useToastStore.getState().addToast(msg, 'error', dur, action),
    info: (msg: string, dur?: number, action?: ToastAction) =>
        useToastStore.getState().addToast(msg, 'info', dur, action),
    warning: (msg: string, dur?: number, action?: ToastAction) =>
        useToastStore.getState().addToast(msg, 'warning', dur, action),
    achievement: (payload: AchievementToastPayload, dur?: number) =>
        useToastStore.getState().addAchievementToast(payload, dur),
};
