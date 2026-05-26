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

export interface ToastMessage {
    id: string;
    type: ToastType;
    message: string;
    payload?: AchievementToastPayload;
}

interface ToastState {
    toasts: ToastMessage[];
    addToast: (message: string, type?: ToastType, duration?: number) => void;
    addAchievementToast: (
        payload: AchievementToastPayload,
        duration?: number,
    ) => void;
    removeToast: (id: string) => void;
}

export const useToastStore = create<ToastState>((set) => ({
    toasts: [],
    addToast: (message, type = 'info', duration = 3000) => {
        const id = Math.random().toString(36).substring(2, 9);
        set((state) => ({ toasts: [...state.toasts, { id, type, message }] }));

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

// Helper function for non-React files
export const toast = {
    success: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'success', dur),
    error: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'error', dur),
    info: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'info', dur),
    warning: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'warning', dur),
    achievement: (payload: AchievementToastPayload, dur?: number) =>
        useToastStore.getState().addAchievementToast(payload, dur),
};
