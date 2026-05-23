import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        patch: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({ default: apiMock }));

import CoachToggle from './CoachToggle';
import { _resetCoachModeStoreForTests } from '../../stores/coachModeStore';


describe('CoachToggle', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
        _resetCoachModeStoreForTests();
    });

    it('renders the "on" pill by default for a newbie (level < 3)', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                xp_balance: 50,
                level: 1,
                level_title: 'Intern',
                xp_into_level: 50,
                xp_to_next_level: 50,
                achievements_unlocked: [],
                milestones: {},
                recent_unlocks: [],
                counters: {
                    base_models_trained: [],
                    import_sources_used: [],
                    successful_training_runs: 0,
                },
            },
        });
        render(<CoachToggle projectId={1} />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-toggle').getAttribute('aria-pressed')).toBe('true');
        });
        expect(screen.getByTestId('coach-toggle').textContent).toMatch(/on/i);
    });

    it('renders the "off" pill by default for a power user (level >= 3)', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                xp_balance: 1200,
                level: 5,
                level_title: 'Senior',
                xp_into_level: 100,
                xp_to_next_level: 500,
                achievements_unlocked: [],
                milestones: {},
                recent_unlocks: [],
                counters: {
                    base_models_trained: [],
                    import_sources_used: [],
                    successful_training_runs: 4,
                },
            },
        });
        render(<CoachToggle projectId={2} />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-toggle').getAttribute('aria-pressed')).toBe('false');
        });
        expect(screen.getByTestId('coach-toggle').textContent).toMatch(/off/i);
    });

    it('clicking flips the toggle and persists the override', async () => {
        apiMock.get.mockResolvedValue({
            data: {
                xp_balance: 50,
                level: 1,
                level_title: 'Intern',
                xp_into_level: 50,
                xp_to_next_level: 50,
                achievements_unlocked: [],
                milestones: {},
                recent_unlocks: [],
                counters: {
                    base_models_trained: [],
                    import_sources_used: [],
                    successful_training_runs: 0,
                },
            },
        });
        render(<CoachToggle projectId={42} />);
        await waitFor(() => {
            expect(screen.getByTestId('coach-toggle').getAttribute('aria-pressed')).toBe('true');
        });
        await userEvent.click(screen.getByTestId('coach-toggle'));
        await waitFor(() => {
            expect(screen.getByTestId('coach-toggle').getAttribute('aria-pressed')).toBe('false');
        });
        // The persisted override survives a "reload" — confirm by
        // reading localStorage directly.
        const raw = localStorage.getItem('brewslm:coach-mode-overrides');
        expect(raw).not.toBeNull();
        const parsed = JSON.parse(raw!);
        expect(parsed['42']).toBe('off');
    });

    it('falls back to default-on when the gamification fetch errors', async () => {
        apiMock.get.mockRejectedValue({ response: { status: 500 } });
        render(<CoachToggle projectId={9} />);
        await waitFor(() => {
            // Failed fetch → level null → computeCoachDefaultOn returns true.
            expect(screen.getByTestId('coach-toggle').getAttribute('aria-pressed')).toBe('true');
        });
    });
});
