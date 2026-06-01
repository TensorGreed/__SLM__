/**
 * Toast renderer — focused on the new optional ``action`` field
 * (used by the bell kill-switch's "Start retry now" shortcut).
 * Existing toast types (success/error/info/warning) keep their
 * existing string-only contract; the action is purely additive.
 */

import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import ToastContainer from './Toast';
import { toast, useToastStore } from '../../stores/toastStore';


function _resetToasts() {
    useToastStore.setState({ toasts: [] });
}


describe('Toast renderer with optional action', () => {
    beforeEach(() => {
        _resetToasts();
    });

    it('renders a plain toast without an action button', () => {
        // Regression guard: the existing toast.success/info/error
        // callers must keep producing toasts that don't carry an
        // action button. The button is purely additive on opt-in.
        render(<ToastContainer />);
        act(() => {
            toast.info('hello world');
        });
        expect(screen.getByText('hello world')).toBeInTheDocument();
        // No element with the toast-action class.
        expect(document.querySelector('.toast-action')).toBeNull();
    });

    it('renders an action button when provided', () => {
        render(<ToastContainer />);
        const onClick = vi.fn();
        act(() => {
            toast.success('done', 5000, { label: 'Start retry now', onClick });
        });
        const btn = document.querySelector('.toast-action');
        expect(btn).not.toBeNull();
        expect(btn!.textContent).toContain('Start retry now');
    });

    it('clicking the action fires onClick and dismisses the toast', async () => {
        const user = userEvent.setup();
        render(<ToastContainer />);
        const onClick = vi.fn();
        act(() => {
            toast.success('done', 5000, { label: 'do thing', onClick });
        });
        const btn = document.querySelector('.toast-action') as HTMLButtonElement;
        await user.click(btn);
        expect(onClick).toHaveBeenCalledTimes(1);
        // Toast dismisses after the action runs — see Toast.tsx's
        // handleAction → handleClose flow.
        await new Promise((r) => setTimeout(r, 250));
        expect(document.querySelector('.toast-action')).toBeNull();
    });

    it('async onClick fires when the user clicks the action', async () => {
        // Used by the kill-switch's "Start retry now" → POST /start.
        // The click handler is fire-and-forget (we don't make the
        // user wait for the network); the regression guard is that
        // the action does in fact fire AND the toast eventually
        // dismisses regardless of the async work's duration.
        const user = userEvent.setup();
        render(<ToastContainer />);
        let resolved = false;
        const onClick = vi.fn(async () => {
            await new Promise((r) => setTimeout(r, 10));
            resolved = true;
        });
        act(() => {
            toast.success('done', 5000, { label: 'go', onClick });
        });
        await user.click(
            document.querySelector('.toast-action') as HTMLButtonElement,
        );
        expect(onClick).toHaveBeenCalledTimes(1);
        // Wait long enough for the async onClick + dismissal
        // animation (200ms in Toast.tsx) to complete.
        await new Promise((r) => setTimeout(r, 300));
        expect(resolved).toBe(true);
        expect(document.querySelector('.toast-action')).toBeNull();
    });

    it('action throwing does not leave the toast stuck', async () => {
        // Failure swallowed by handleAction (fire-and-forget by
        // contract). Toast still dismisses; any follow-up surfaces
        // through a separate error toast the caller emits.
        const user = userEvent.setup();
        render(<ToastContainer />);
        const onClick = vi.fn(() => {
            throw new Error('boom');
        });
        act(() => {
            toast.success('done', 5000, { label: 'fail', onClick });
        });
        const btn = document.querySelector('.toast-action') as HTMLButtonElement;
        await user.click(btn);
        await new Promise((r) => setTimeout(r, 250));
        // Toast container is empty after dismissal — no stale
        // action button hangs around.
        expect(document.querySelector('.toast-action')).toBeNull();
    });
});
