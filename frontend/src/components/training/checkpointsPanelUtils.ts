/** Pure helpers for CheckpointsPanel (kept out of the component file for fast refresh). */

export interface ResumeFromResponse {
    experiment_id?: number;
    resume_from?: {
        checkpoint_step?: number;
        resumes_training_state?: boolean;
        has_optimizer_state?: boolean;
    };
}

/** What the fork carries over — so the user knows it continues rather than restarts. */
export function forkNotice(newId: number, step: number, data: ResumeFromResponse | undefined): string {
    const block = data?.resume_from;
    if (!block?.resumes_training_state) {
        return `Run #${newId} created from step ${step}. Start it from the runs list.`;
    }
    const state = block.has_optimizer_state
        ? 'its weights, optimizer and scheduler state'
        : 'its weights (the optimizer restarts: this checkpoint has no optimizer state)';
    return `Run #${newId} created. It continues at step ${step} with ${state}. Start it from the runs list.`;
}
