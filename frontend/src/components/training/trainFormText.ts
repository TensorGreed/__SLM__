/** Text + types for the Train form's readiness line (pure, no React). */

export interface TrainReadiness {
    state: 'idle' | 'checking' | 'ready' | 'blocked' | 'error';
    /** Blocking issues (preflight errors). */
    errors: string[];
    warnings: string[];
    message?: string | null;
}

export function readinessText(readiness: TrainReadiness): string {
    switch (readiness.state) {
        case 'idle':
            return 'Not checked yet.';
        case 'checking':
            return 'Checking the data, the model and the runtime…';
        case 'ready':
            return readiness.warnings.length
                ? `Ready to train, ${readiness.warnings.length} warning${readiness.warnings.length === 1 ? '' : 's'}.`
                : 'Ready to train.';
        case 'blocked':
            return `${readiness.errors.length} blocking issue${readiness.errors.length === 1 ? '' : 's'}.`;
        case 'error':
            return readiness.message || 'The check could not run.';
        default:
            return '';
    }
}
