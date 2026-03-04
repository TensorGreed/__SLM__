import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastMessage {
    id: string;
    type: ToastType;
    message: string;
}

interface ToastState {
    toasts: ToastMessage[];
    addToast: (message: string, type?: ToastType, duration?: number) => void;
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
    removeToast: (id) => set((state) => ({ toasts: state.toasts.filter(t => t.id !== id) })),
}));

// Helper function for non-React files
export const toast = {
    success: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'success', dur),
    error: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'error', dur),
    info: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'info', dur),
    warning: (msg: string, dur?: number) => useToastStore.getState().addToast(msg, 'warning', dur),
};
