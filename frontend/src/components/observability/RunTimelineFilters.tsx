/**
 * RunTimelineFilters — controlled filter bar for the timeline page
 * (priority.md P36, P32 filters).
 *
 * Emits a single ``onChange`` with the full :class:`TimelineFilters`
 * shape on each control change. The parent owns the filters state +
 * fetch lifecycle so the same filters can drive the tree fetch and
 * the failure-cluster + run-events drilldown surfaces in the same
 * page.
 */

import { useCallback } from 'react';

import type {
    Severity,
    Stage,
    TimelineFilters,
} from '../../types/observability';
import {
    KNOWN_SEVERITIES,
    KNOWN_STAGES,
} from '../../types/observability';

interface Props {
    value: TimelineFilters;
    onChange: (next: TimelineFilters) => void;
    onRefresh: () => void;
    loading?: boolean;
    truncated?: boolean;
}

export default function RunTimelineFilters({
    value,
    onChange,
    onRefresh,
    loading = false,
    truncated = false,
}: Props) {
    const update = useCallback(
        (patch: Partial<TimelineFilters>) => {
            onChange({ ...value, ...patch });
        },
        [onChange, value],
    );

    return (
        <div
            className="timeline-filters"
            role="group"
            aria-label="Timeline filters"
        >
            <label>
                <span>Stage</span>
                <select
                    value={value.stage ?? ''}
                    onChange={(e) =>
                        update({ stage: (e.target.value || '') as Stage | '' })
                    }
                    aria-label="Filter by stage"
                >
                    <option value="">all stages</option>
                    {KNOWN_STAGES.map((s) => (
                        <option key={s} value={s}>
                            {s}
                        </option>
                    ))}
                </select>
            </label>
            <label>
                <span>Severity</span>
                <select
                    value={value.severity ?? ''}
                    onChange={(e) =>
                        update({
                            severity: (e.target.value || '') as Severity | '',
                        })
                    }
                    aria-label="Filter by severity"
                >
                    <option value="">all severities</option>
                    {KNOWN_SEVERITIES.map((s) => (
                        <option key={s} value={s}>
                            {s}
                        </option>
                    ))}
                </select>
            </label>
            <label>
                <span>Run id (anchor)</span>
                <input
                    type="text"
                    value={value.run_id ?? ''}
                    placeholder="exp-42"
                    onChange={(e) => update({ run_id: e.target.value })}
                    aria-label="Anchor on run id"
                />
            </label>
            <label>
                <span>Since</span>
                <input
                    type="text"
                    value={value.since ?? ''}
                    placeholder="ISO timestamp"
                    onChange={(e) => update({ since: e.target.value })}
                    aria-label="Since (ISO timestamp)"
                />
            </label>
            <label>
                <span>Until</span>
                <input
                    type="text"
                    value={value.until ?? ''}
                    placeholder="ISO timestamp"
                    onChange={(e) => update({ until: e.target.value })}
                    aria-label="Until (ISO timestamp)"
                />
            </label>
            <label>
                <span>Limit</span>
                <input
                    type="number"
                    min={1}
                    max={2000}
                    value={value.limit ?? 500}
                    onChange={(e) =>
                        update({
                            limit: Math.max(
                                1,
                                Math.min(2000, Number(e.target.value) || 500),
                            ),
                        })
                    }
                    aria-label="Max events"
                />
            </label>
            <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={onRefresh}
                disabled={loading}
            >
                {loading ? 'Loading…' : 'Refresh'}
            </button>
            {truncated && (
                <span className="badge badge-warning" role="status">
                    truncated — narrow the window
                </span>
            )}
        </div>
    );
}
