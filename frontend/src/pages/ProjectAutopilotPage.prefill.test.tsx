/**
 * Newbie UX Phase 3.1 — pre-fill the Autopilot intent textarea from the
 * demo project's suggested_brief on first mount.
 *
 * Separate from ProjectAutopilotPage.test.tsx so we can parameterise
 * the project context per test without disturbing the (already-large)
 * existing test file's fixed mock.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock, contextHolder } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn() },
    contextHolder: {
        current: {
            projectId: 77,
            project: { id: 77, name: 'Test', dataset_adapter_preset: null },
            pipelineStatus: null,
        } as Record<string, unknown>,
    },
}));

vi.mock('../api/client', () => ({ default: apiMock }));

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
    return {
        ...actual,
        useOutletContext: () => contextHolder.current,
    };
});

import ProjectAutopilotPage from './ProjectAutopilotPage';

function setProject(project: Record<string, unknown>) {
    contextHolder.current = {
        projectId: project.id ?? 77,
        project,
        pipelineStatus: null,
    };
}

describe('ProjectAutopilotPage — Phase 3.1 demo-project intent pre-fill', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('pre-fills the intent textarea from project.dataset_adapter_preset.suggested_brief', () => {
        setProject({
            id: 99,
            name: 'Demo · Support FAQ',
            dataset_adapter_preset: {
                demo_slug: 'support-faq',
                suggested_brief: 'Build a support FAQ assistant — concise tone, no hallucinations.',
            },
        });
        render(<ProjectAutopilotPage />);
        const textarea = screen.getByRole('textbox', { name: /Autopilot intent/i }) as HTMLTextAreaElement;
        expect(textarea.value).toBe(
            'Build a support FAQ assistant — concise tone, no hallucinations.',
        );
    });

    it('shows the "pre-filled from demo project" hint banner', () => {
        setProject({
            id: 99,
            name: 'Demo · Support FAQ',
            dataset_adapter_preset: {
                demo_slug: 'support-faq',
                suggested_brief: 'Build a support FAQ assistant.',
            },
        });
        render(<ProjectAutopilotPage />);
        expect(
            screen.getByRole('status', { name: /Intent pre-filled from demo project/i }),
        ).toBeInTheDocument();
    });

    it('falls back to the default placeholder when no suggested_brief is set', () => {
        setProject({
            id: 77,
            name: 'Plain Project',
            dataset_adapter_preset: null,
        });
        render(<ProjectAutopilotPage />);
        const textarea = screen.getByRole('textbox', { name: /Autopilot intent/i }) as HTMLTextAreaElement;
        expect(textarea.value).toBe(
            'Train a concise support assistant with JSON output.',
        );
        // And no hint banner.
        expect(
            screen.queryByRole('status', { name: /pre-filled from demo project/i }),
        ).not.toBeInTheDocument();
    });

    it('still lets the user edit the pre-filled intent', async () => {
        setProject({
            id: 99,
            name: 'Demo · Support FAQ',
            dataset_adapter_preset: { suggested_brief: 'Initial brief.' },
        });
        render(<ProjectAutopilotPage />);
        const textarea = screen.getByRole('textbox', { name: /Autopilot intent/i }) as HTMLTextAreaElement;
        const user = userEvent.setup();
        await user.clear(textarea);
        await user.type(textarea, 'My own intent.');
        expect(textarea.value).toBe('My own intent.');
    });

    it('ignores a non-string suggested_brief value', () => {
        setProject({
            id: 99,
            name: 'Weird preset',
            // suggested_brief is something we wouldn't normally produce —
            // confirm we don't crash and fall back to the default.
            dataset_adapter_preset: { suggested_brief: 12345 },
        });
        render(<ProjectAutopilotPage />);
        const textarea = screen.getByRole('textbox', { name: /Autopilot intent/i }) as HTMLTextAreaElement;
        expect(textarea.value).toBe(
            'Train a concise support assistant with JSON output.',
        );
    });
});
