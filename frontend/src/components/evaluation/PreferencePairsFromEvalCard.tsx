/**
 * PreferencePairsFromEvalCard — turn one eval result's wrong / partial answers
 * into a preference-pair dataset (chosen = the answer key, rejected = what the
 * model said).
 *
 * Provenance: the header names the run, dataset and eval result id the pairs
 * come from. Skipped rows are shown grouped by reason, including when nothing
 * could be built (the server answers 400 with the same breakdown).
 */

import { useState } from 'react';

import {
    SKIP_REASON_LABELS,
    TEST_SPLITS,
    buildPairsFromEval,
    skipBreakdownFromError,
    type PairsFromEvalResponse,
    type PairVerdict,
} from '../../api/preferencePairs';
import { parseErrorEnvelope } from '../../api/errors';
import './PreferencePairsFromEvalCard.css';

interface Props {
    projectId: number;
    evalResultId: number;
    /** The split the eval ran on (its dataset name, e.g. "validation" / "test"). */
    split?: string;
    /** Where the numbers come from, e.g. "Run #21 · gold_dev (result 1001)". */
    provenance: string;
}

function BreakdownList({ skipped, testId }: { skipped: Record<string, number>; testId: string }) {
    const entries = Object.entries(skipped).filter(([, n]) => n > 0);
    if (entries.length === 0) return null;
    return (
        <ul className="pairs-from-eval__skips" data-testid={testId}>
            {entries.map(([reason, n]) => (
                <li key={reason} data-testid={`${testId}-${reason}`}>
                    {SKIP_REASON_LABELS[reason] ?? reason}: {n}
                </li>
            ))}
        </ul>
    );
}

export default function PreferencePairsFromEvalCard({ projectId, evalResultId, split, provenance }: Props) {
    const isTestSplit = TEST_SPLITS.includes((split ?? '').toLowerCase());
    const [includePartial, setIncludePartial] = useState(true);
    const [allowTest, setAllowTest] = useState(false);
    const [busy, setBusy] = useState(false);
    const [result, setResult] = useState<PairsFromEvalResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [errorSkips, setErrorSkips] = useState<Record<string, number> | null>(null);

    const run = async () => {
        setBusy(true);
        setError(null);
        setErrorSkips(null);
        setResult(null);
        const include: PairVerdict[] = includePartial ? ['wrong', 'partial'] : ['wrong'];
        try {
            setResult(await buildPairsFromEval(projectId, {
                eval_result_id: evalResultId,
                include,
                ...(isTestSplit && allowTest ? { allow_test_split: true } : {}),
            }));
        } catch (e) {
            setError(parseErrorEnvelope(e).message || 'No preference pairs could be built.');
            setErrorSkips(skipBreakdownFromError(e));
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="pairs-from-eval" data-testid={`pairs-from-eval-${evalResultId}`}>
            <h4 className="pairs-from-eval__title">Preference pairs from judged answers</h4>
            <p className="u-hint u-mt0" data-testid={`pairs-from-eval-provenance-${evalResultId}`}>{provenance}</p>
            <label className="pairs-from-eval__opt">
                <input
                    type="checkbox"
                    checked={includePartial}
                    onChange={(e) => setIncludePartial(e.target.checked)}
                    data-testid={`pairs-include-partial-${evalResultId}`}
                />
                Include partial answers (otherwise wrong only)
            </label>
            {isTestSplit && (
                <label className="pairs-from-eval__opt">
                    <input
                        type="checkbox"
                        checked={allowTest}
                        onChange={(e) => setAllowTest(e.target.checked)}
                        data-testid={`pairs-allow-test-${evalResultId}`}
                    />
                    Allow test split (measures memorisation)
                </label>
            )}
            <button
                type="button"
                className="btn btn-secondary"
                disabled={busy}
                onClick={() => { void run(); }}
                data-testid={`pairs-from-eval-run-${evalResultId}`}
            >
                {busy ? 'Building pairs...' : 'Build preference pairs from wrong/partial answers'}
            </button>
            {result && (
                <div data-testid={`pairs-from-eval-result-${evalResultId}`}>
                    <p>
                        Created <strong data-testid={`pairs-created-${evalResultId}`}>{result.created}</strong> pair
                        {result.created === 1 ? '' : 's'} named{' '}
                        <strong data-testid={`pairs-dataset-${evalResultId}`}>{result.dataset_name}</strong>
                        {result.source_split ? <span className="u-hint"> (split: {result.source_split})</span> : null}
                    </p>
                    {result.dataset_path && (
                        <p className="u-hint" data-testid={`pairs-path-${evalResultId}`}>File: {result.dataset_path}</p>
                    )}
                    <p className="u-hint">
                        To train DPO/ORPO on these pairs, set it as the alignment dataset path in Train.
                    </p>
                    {result.warning && (
                        <p role="alert" className="pairs-from-eval__error" data-testid={`pairs-warning-${evalResultId}`}>
                            {result.warning}
                        </p>
                    )}
                    <BreakdownList skipped={result.skipped ?? {}} testId={`pairs-skipped-${evalResultId}`} />
                </div>
            )}
            {error && (
                <div role="alert" data-testid={`pairs-from-eval-error-${evalResultId}`}>
                    <p className="pairs-from-eval__error">{error}</p>
                    {errorSkips && <BreakdownList skipped={errorSkips} testId={`pairs-error-skipped-${evalResultId}`} />}
                </div>
            )}
        </div>
    );
}
