import { useState } from 'react';
import { useToastStore, type ToastMessage } from '../../stores/toastStore';
import './Toast.css';

const ICONS = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️'
};

const ToastItem = ({ toast }: { toast: ToastMessage }) => {
    const removeToast = useToastStore(s => s.removeToast);
    const [isRemoving, setIsRemoving] = useState(false);

    const handleClose = () => {
        setIsRemoving(true);
        setTimeout(() => removeToast(toast.id), 200); // Wait for fade-out animation
    };

    return (
        <div className={`toast ${toast.type} ${isRemoving ? 'removing' : ''}`} role="alert">
            <span className="toast-icon">{ICONS[toast.type]}</span>
            <span className="toast-message">{toast.message}</span>
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
