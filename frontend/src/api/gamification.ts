/**
 * Typed wrappers for the Lab Journal (gamification) API.
 *
 * Two read-only endpoints feed the UI:
 *   GET /api/projects/{id}/gamification              — chip + drawer status
 *   GET /api/projects/{id}/gamification/achievements — full catalog
 *
 * Mutations happen server-side via the RunEvent tap; the client only
 * reads. No POST/PUT/DELETE surface here.
 */

import api from './client';

export type AchievementTier = 'onboarding' | 'mastery' | 'discovery';

export interface RecentUnlock {
    kind: 'achievement' | 'level_up';
    achievement_id?: string;
    title?: string;
    description?: string;
    tier?: AchievementTier;
    xp_awarded?: number;
    unlocked_at: string;
    level_after?: number | null;
}

export interface ProgressionState {
    xp_balance: number;
    level: number;
    level_title: string;
    xp_into_level: number;
    xp_to_next_level: number;
    achievements_unlocked: string[];
    milestones: Record<string, string>;
    recent_unlocks: RecentUnlock[];
    counters: {
        base_models_trained: string[];
        import_sources_used: string[];
        successful_training_runs: number;
        [key: string]: unknown;
    };
}

export interface AchievementListItem {
    id: string;
    title: string;
    description: string;
    xp: number;
    tier: AchievementTier;
    hidden: boolean;
    unlocked: boolean;
    unlocked_at: string | null;
}

export interface AchievementListResponse {
    achievements: AchievementListItem[];
    summary: {
        total: number;
        unlocked: number;
        level: number;
        level_title: string;
        xp_balance: number;
    };
}

export async function fetchProgression(
    projectId: number,
): Promise<ProgressionState> {
    const res = await api.get<ProgressionState>(
        `/projects/${projectId}/gamification`,
    );
    return res.data;
}

export async function fetchAchievements(
    projectId: number,
): Promise<AchievementListResponse> {
    const res = await api.get<AchievementListResponse>(
        `/projects/${projectId}/gamification/achievements`,
    );
    return res.data;
}
