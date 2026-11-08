import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), patch: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import RunFailureBanner from './RunFailureBanner';
import { runFailure } from './checkpointsPanelUtils';

const MESSAGE =
    'Stopped at step 60 of 250: reached the 6-hour limit (EXTERNAL_COMMAND_TIMEOUT_SECONDS). Last checkpoint: step 50.';

function failedExp(failure: Record<string, unknown> | null, status = 'failed') {
    return {
        id: 12,
        status,
        config: { _runtime: { backend: 'external', returncode: -15, ...(failure ? { failure } : {}) } },
    };
}

describe('RunFailureBanner', () => {
    beforeEach(() => {
        apiMock.post.mockReset();
    });

    it('shows the saved reason and continues from the resumable checkpoint', async () => {
        apiMock.post.mockResolvedValueOnce({
            data: {
                experiment_id: 13,
                resume_from: { checkpoint_step: 50, resumes_training_state: true, has_optimizer_state: true },
            },
        });
        const onContinued = vi.fn();
        render(
            <RunFailureBanner
                projectId={5}
                experiment={failedExp({ code: 'timeout', message: MESSAGE, resumable_checkpoint_step: 50 })}
                onContinued={onContinued}
            />,
        );
        expect(screen.getByText('Training hit its time limit')).toBeInTheDocument();
        expect(screen.getByText(MESSAGE)).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Continue from step 50' }));
        await waitFor(() => expect(onContinued).toHaveBeenCalledWith(13));
        expect(apiMock.post).toHaveBeenCalledWith('/projects/5/training/runs/12/resume-from/50', {});
        expect(screen.getByText(/Run #13 created\. It continues at step 50/)).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Continue from step/ })).not.toBeInTheDocument();
    });

    it('explains a refused continue', async () => {
        apiMock.post.mockRejectedValueOnce({ response: { data: { detail: 'checkpoint_not_resumable' } } });
        render(
            <RunFailureBanner
                projectId={5}
                experiment={failedExp({ code: 'signal', message: 'Stopped at step 60.', resumable_checkpoint_step: 50 })}
            />,
        );
        expect(screen.getByText('Training was stopped')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Continue from step 50' }));
        expect(await screen.findByText(/can't be resumed/)).toBeInTheDocument();
    });

    it('has no button without a resumable checkpoint', () => {
        render(
            <RunFailureBanner
                projectId={5}
                experiment={failedExp({ code: 'exit', message: 'Stopped before the first logged step: x. No checkpoint was saved.' })}
            />,
        );
        expect(screen.getByText('Training exited with an error')).toBeInTheDocument();
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('renders nothing for runs that did not fail or have no saved reason', () => {
        const { container: running } = render(
            <RunFailureBanner projectId={5} experiment={failedExp({ code: 'timeout', message: MESSAGE }, 'running')} />,
        );
        expect(running).toBeEmptyDOMElement();
        const { container: legacy } = render(<RunFailureBanner projectId={5} experiment={failedExp(null)} />);
        expect(legacy).toBeEmptyDOMElement();
        expect(runFailure({ _runtime: { failure: { code: 'exit' } } })).toBeNull();
    });
});
