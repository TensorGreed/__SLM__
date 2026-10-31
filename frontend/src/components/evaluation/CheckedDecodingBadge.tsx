/**
 * CheckedDecodingBadge (Epic Y4): says that an eval was decoded with checked decoding
 * ("Checked decoding · up to 3 tries · resample") and how many rows a retry fixed, with the
 * eval it came from. Renders nothing for an eval that ran without it.
 */

import { checkedDecodingBadgeText, rowsFixedText, type CheckedDecodingSummary } from '../../api/checkedDecoding';
import './CheckedDecoding.css';

interface Props {
    summary: CheckedDecodingSummary | null | undefined;
    /** Which eval the numbers come from ("eval #91", "fine-tuned eval #91"). */
    source?: string | null;
    testId?: string;
}

export default function CheckedDecodingBadge({ summary, source, testId = 'checked-decoding-badge' }: Props) {
    const label = checkedDecodingBadgeText(summary?.policy);
    if (!label) return null;
    const fixed = rowsFixedText(summary);
    return (
        <span className="checked-decoding" data-testid={testId}>
            <span className="checked-decoding__badge" data-testid={`${testId}-label`}>{label}</span>
            {fixed && <span className="checked-decoding__text" data-testid={`${testId}-fixed`}>{fixed}</span>}
            {source && <span className="checked-decoding__source">({source})</span>}
        </span>
    );
}
