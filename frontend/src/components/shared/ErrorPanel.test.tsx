import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { ErrorEnvelope } from '../../api/errors';
import ErrorPanel from './ErrorPanel';


function makeEnvelope(overrides: Partial<ErrorEnvelope> = {}): ErrorEnvelope {
    return {
        errorCode: 'SYNTHETIC_LLM_REFUSAL',
        stage: 'synthetic',
        message: 'Model refused on guardrails.',
        actionableFix: 'Switch to Qwen 2.5.',
        docsUrl: '/docs/troubleshooting/synthetic',
        troubleshootingId: 'err_a3f8d4e2',
        metadata: { raw_llm_snippet: 'I cannot...' },
        statusCode: 400,
        isFallback: false,
        ...overrides,
    };
}


describe('ErrorPanel', () => {
    it('renders headline, status, remediation, and troubleshooting id', () => {
        render(<ErrorPanel envelope={makeEnvelope()} />);
        expect(screen.getByTestId('error-panel-message').textContent)
            .toMatch(/Model refused/);
        expect(screen.getByTestId('error-panel-status').textContent).toMatch(/400/);
        expect(screen.getByTestId('error-panel-remediation').textContent)
            .toMatch(/Switch to Qwen/);
        expect(screen.getByTestId('error-panel-trace-id').textContent)
            .toMatch(/err_a3f8d4e2/);
    });

    it('shows the "Show technical details" expander only when metadata is present', () => {
        const { rerender } = render(
            <ErrorPanel envelope={makeEnvelope({ metadata: { raw_llm_snippet: 'x' } })} />,
        );
        expect(screen.getByTestId('error-panel-details')).toBeInTheDocument();
        rerender(<ErrorPanel envelope={makeEnvelope({ metadata: null })} />);
        expect(screen.queryByTestId('error-panel-details')).not.toBeInTheDocument();
    });

    it('renders each metadata key as a labelled grid row', () => {
        render(
            <ErrorPanel envelope={makeEnvelope({
                metadata: {
                    raw_llm_snippet: 'I cannot...',
                    backend_used: 'ollama:llama3:latest',
                    elapsed_sec: 0.6,
                },
            })} />,
        );
        const details = screen.getByTestId('error-panel-details');
        expect(details.textContent).toMatch(/raw_llm_snippet/);
        expect(details.textContent).toMatch(/I cannot/);
        expect(details.textContent).toMatch(/backend_used/);
        expect(details.textContent).toMatch(/ollama:llama3:latest/);
        expect(details.textContent).toMatch(/elapsed_sec/);
    });

    it('uses the critical badge style for 5xx statuses', () => {
        const { container } = render(
            <ErrorPanel envelope={makeEnvelope({ statusCode: 500 })} />,
        );
        const panel = container.querySelector('.error-panel');
        expect(panel?.className).toMatch(/error-panel--critical/);
    });

    it('uses the warning badge style for 4xx statuses', () => {
        const { container } = render(
            <ErrorPanel envelope={makeEnvelope({ statusCode: 422 })} />,
        );
        const panel = container.querySelector('.error-panel');
        expect(panel?.className).toMatch(/error-panel--warning/);
    });

    it('fires onDismiss when the close button is clicked', async () => {
        const onDismiss = vi.fn();
        render(<ErrorPanel envelope={makeEnvelope()} onDismiss={onDismiss} />);
        await userEvent.click(screen.getByTestId('error-panel-dismiss'));
        expect(onDismiss).toHaveBeenCalledOnce();
    });

    it('does not render the dismiss button when onDismiss is not provided', () => {
        render(<ErrorPanel envelope={makeEnvelope()} />);
        expect(screen.queryByTestId('error-panel-dismiss')).not.toBeInTheDocument();
    });

    it('copies the troubleshooting id to clipboard when copy is clicked', async () => {
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.assign(navigator, { clipboard: { writeText } });
        render(<ErrorPanel envelope={makeEnvelope()} />);
        await userEvent.click(screen.getByTestId('error-panel-copy'));
        expect(writeText).toHaveBeenCalledWith('err_a3f8d4e2');
    });

    it('renders the fallback-note when the envelope is a client-side fallback', () => {
        render(<ErrorPanel envelope={makeEnvelope({ isFallback: true })} />);
        expect(screen.getByTestId('error-panel-fallback-note')).toBeInTheDocument();
    });

    it('renders supplied action buttons in the footer', () => {
        render(
            <ErrorPanel
                envelope={makeEnvelope()}
                actions={<button data-testid="retry-btn">Retry</button>}
            />,
        );
        expect(screen.getByTestId('retry-btn')).toBeInTheDocument();
    });

    it('exposes the error_code on the root element so dispatchers can branch', () => {
        const { container } = render(
            <ErrorPanel envelope={makeEnvelope({ errorCode: 'SYNTHETIC_LLM_REFUSAL' })} />,
        );
        const panel = container.querySelector('.error-panel');
        expect(panel?.getAttribute('data-error-code')).toBe('SYNTHETIC_LLM_REFUSAL');
    });
});
