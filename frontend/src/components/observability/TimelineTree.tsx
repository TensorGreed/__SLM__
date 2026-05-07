/**
 * TimelineTree — recursive tree view of P32 timeline nodes
 * (priority.md P36).
 *
 * One row per ``run_id`` with severity badge, stage chip, summary,
 * event count, and an "Open" link that surfaces the per-run event
 * stream via the parent's ``onSelectRun`` callback (the parent owns
 * the drill-in drawer so multiple consumers can share the timeline
 * tree without coupling). Branches are collapsible per row;
 * top-level expansion state is owned here so a deep tree doesn't
 * force the parent to track every node.
 */

import { useCallback, useMemo, useState } from 'react';

import type { TimelineNode } from '../../types/observability';

interface Props {
    tree: TimelineNode[];
    selectedRunId: string | null;
    onSelectRun: (runId: string) => void;
    /**
     * Optional initial expansion state. When unset, only roots are
     * expanded by default (children collapsed) so a wide tree fits
     * on screen.
     */
    defaultAllExpanded?: boolean;
}

function severityBadgeClass(severity: string): string {
    switch (severity) {
        case 'critical':
            return 'badge badge-danger';
        case 'error':
            return 'badge badge-danger';
        case 'warning':
            return 'badge badge-warning';
        case 'info':
        default:
            return 'badge badge-info';
    }
}

function formatDuration(seconds: number | null): string {
    if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) {
        return '—';
    }
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

function formatTs(value: string | null): string {
    if (!value) return '—';
    try {
        const d = new Date(value);
        if (Number.isNaN(d.getTime())) return value;
        return d.toLocaleString();
    } catch {
        return value;
    }
}

interface RowProps {
    node: TimelineNode;
    depth: number;
    expanded: Set<string>;
    toggleExpand: (runId: string) => void;
    selectedRunId: string | null;
    onSelectRun: (runId: string) => void;
}

function TimelineRow({
    node,
    depth,
    expanded,
    toggleExpand,
    selectedRunId,
    onSelectRun,
}: RowProps) {
    const hasChildren = node.children.length > 0;
    const isOpen = expanded.has(node.run_id);
    const isSelected = node.run_id === selectedRunId;

    return (
        <li
            className={`timeline-row ${isSelected ? 'is-selected' : ''}`}
            data-run-id={node.run_id}
        >
            <div
                className="timeline-row-body"
                style={{ paddingLeft: `${depth * 16}px` }}
            >
                <button
                    type="button"
                    className="timeline-toggle"
                    aria-label={
                        hasChildren
                            ? isOpen
                                ? 'Collapse children'
                                : 'Expand children'
                            : 'No children'
                    }
                    onClick={() => hasChildren && toggleExpand(node.run_id)}
                    disabled={!hasChildren}
                >
                    {hasChildren ? (isOpen ? '▾' : '▸') : '·'}
                </button>
                <span className={severityBadgeClass(String(node.highest_severity))}>
                    {String(node.highest_severity)}
                </span>
                <span className="timeline-stage">{node.stage}</span>
                <button
                    type="button"
                    className="timeline-run-id"
                    onClick={() => onSelectRun(node.run_id)}
                    aria-label={`Open events for ${node.run_id}`}
                >
                    {node.run_id}
                </button>
                {node.is_orphan && (
                    <span className="badge badge-warning" title="Parent not in window">
                        orphan
                    </span>
                )}
                <span className="timeline-summary">
                    {node.summary || '(no summary)'}
                </span>
                <span className="timeline-meta dim">
                    {node.event_count} ev · {formatDuration(node.duration_seconds)} · {formatTs(node.last_ts)}
                </span>
                {node.latest_reason_code && (
                    <code className="timeline-reason-code">
                        {node.latest_reason_code}
                    </code>
                )}
            </div>
            {hasChildren && isOpen && (
                <ul className="timeline-children">
                    {node.children.map((child) => (
                        <TimelineRow
                            key={child.run_id}
                            node={child}
                            depth={depth + 1}
                            expanded={expanded}
                            toggleExpand={toggleExpand}
                            selectedRunId={selectedRunId}
                            onSelectRun={onSelectRun}
                        />
                    ))}
                </ul>
            )}
        </li>
    );
}

function collectAllRunIds(nodes: TimelineNode[]): string[] {
    const out: string[] = [];
    const walk = (n: TimelineNode) => {
        out.push(n.run_id);
        n.children.forEach(walk);
    };
    nodes.forEach(walk);
    return out;
}

function collectRoots(nodes: TimelineNode[]): string[] {
    return nodes.map((n) => n.run_id);
}

export default function TimelineTree({
    tree,
    selectedRunId,
    onSelectRun,
    defaultAllExpanded = false,
}: Props) {
    const initialExpanded = useMemo(() => {
        return new Set<string>(
            defaultAllExpanded ? collectAllRunIds(tree) : collectRoots(tree),
        );
    }, [tree, defaultAllExpanded]);

    const [expanded, setExpanded] = useState<Set<string>>(initialExpanded);

    const toggleExpand = useCallback((runId: string) => {
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(runId)) {
                next.delete(runId);
            } else {
                next.add(runId);
            }
            return next;
        });
    }, []);

    if (!tree.length) {
        return (
            <div className="timeline-empty" role="status">
                No timeline events for the current filter set.
            </div>
        );
    }

    return (
        <ul className="timeline-tree" aria-label="Run timeline tree">
            {tree.map((root) => (
                <TimelineRow
                    key={root.run_id}
                    node={root}
                    depth={0}
                    expanded={expanded}
                    toggleExpand={toggleExpand}
                    selectedRunId={selectedRunId}
                    onSelectRun={onSelectRun}
                />
            ))}
        </ul>
    );
}
