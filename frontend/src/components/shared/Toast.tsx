/**
 * Toast notification renderer with achievement support and auto-dismiss capability.
 */

import { useState } from 'react';
import { useToastStore, type ToastMessage } from '../../stores/toastStore';
import AchievementToast from '../gamification/AchievementToast';
import './Toast.css';

const ICONS: Record<string, string> = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️',
};

const ToastItem = ({ toast }: { toast: ToastMessage }) => {
    const removeToast = useToastStore(s => s.removeToast);
    const [isRemoving, setIsRemoving] = useState(false);

    // Gamification toasts get their own CRT-styled renderer. Standard
    // pill style stays for success / error / info / warning.
    if (toast.type === 'achievement') {
        return <AchievementToast toast={toast} />;
    }

    const handleClose = () => {
        setIsRemoving(true);
        setTimeout(() => removeToast(toast.id), 200); // Wait for fade-out animation
    };

    const handleAction = async () => {
        // Action runs first, then the toast dismisses. We dismiss
        // regardless of action success/failure so the user isn't
        // stuck with a stale "Start retry now" button if the
        // start endpoint 404s — the action's purpose is satisfied
        // (they tried) and any follow-up shows up via the bell.
        if (!toast.action) return;
        try {
            await toast.action.onClick();
        } catch {
            // swallow — handler is fire-and-forget by contract.
        }
        handleClose();
    };

    return (
        <div className={`toast ${toast.type} ${isRemoving ? 'removing' : ''}`} role="alert">
            <span className="toast-icon">{ICONS[toast.type] ?? 'ℹ️'}</span>
            <span className="toast-message">{toast.message}</span>
            {toast.action && (
                <button
                    type="button"
                    className="toast-action"
                    onClick={() => void handleAction()}
                    data-testid={`toast-${toast.id}-action`}
                >
                    {toast.action.label}
                </button>
            )}
            <button className="toast-close" onClick={handleClose} aria-label="Close">×</button>
        </div>
    );
};

export default function ToastContainer() {
    const toasts = useToastStore(s => s.toasts);

    if (toasts.length === 0) return null;

    return (
        <div className="toast-container">
            {toasts.map(t => (
                <ToastItem key={t.id} toast={t} />
            ))}
        </div>
    );
}
