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

const ERROR_TEXT: Record<string, string> = {
    checkpoint_not_resumable:
        "This checkpoint can't be resumed: its training state or weights are missing on disk, so a new run would start again from step 0.",
};

/** The API's error detail as a sentence, with known reason codes spelled out. */
export function errorDetail(err: unknown, fallback: string): string {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
    if (typeof detail === 'string' && ERROR_TEXT[detail]) return ERROR_TEXT[detail];
    return typeof detail === 'string' && detail ? detail : fallback;
}

/** ``config._runtime.failure`` — why a real run stopped (saved by the backend). */
export interface RunFailure {
    code?: 'timeout' | 'signal' | 'exit' | 'error' | string;
    message?: string;
    last_step?: number | null;
    total_steps?: number | null;
    last_checkpoint_step?: number | null;
    resumable_checkpoint_step?: number | null;
    error_line?: string | null;
}

export function runFailure(config: Record<string, unknown> | undefined | null): RunFailure | null {
    const runtime = config?.['_runtime'];
    if (!runtime || typeof runtime !== 'object') return null;
    const failure = (runtime as Record<string, unknown>)['failure'];
    if (!failure || typeof failure !== 'object') return null;
    const record = failure as RunFailure;
    return typeof record.message === 'string' && record.message ? record : null;
}
