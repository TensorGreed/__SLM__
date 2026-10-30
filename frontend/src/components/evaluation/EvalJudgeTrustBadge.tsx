/**
 * EvalJudgeTrustBadge (Judge calibration V4): how far the eval judge can be trusted, from blind labels.
 * Shown next to the judge on Settings → Judge model and on the eval summary card. Nothing is gated by it.
 */

import { TRUST_BADGE, evalJudgeTrustText, type EvalJudgeTrust } from '../../api/evalJudgeCalibration';
import './EvalJudgeTrustBadge.css';

interface Props {
    trust: EvalJudgeTrust | null | undefined;
    /** What the labeller labels ("answers", "plans"). */
    what?: string;
    testId?: string;
}

export default function EvalJudgeTrustBadge({ trust, what = 'answers', testId = 'eval-judge-trust' }: Props) {
    if (!trust || !trust.status) return null;
    return (
        <span className="eval-judge-trust" data-testid={testId}>
            <span className={`eval-judge-trust__badge eval-judge-trust__badge--${trust.status}`} data-testid={`${testId}-status`}>
                {TRUST_BADGE[trust.status] ?? trust.status}
            </span>{' '}
            <span className="eval-judge-trust__text">{evalJudgeTrustText(trust, what)}</span>
        </span>
    );
}
