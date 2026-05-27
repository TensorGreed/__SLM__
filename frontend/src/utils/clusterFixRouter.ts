/**
 * clusterFixRouter — encodes a failure cluster into a "Fix in gold
 * set" deep link (E1, post-eval remediation arc).
 *
 * Pairs with ``forecastActionRouter`` from T4 — same pattern, different
 * domain: this one routes a cluster card click into the gold-set
 * page with prefill params so the LLM-gen panel lands focused on the
 * cluster's failure pattern with ``distribution.hallucination_traps``
 * defaulted to 5.
 *
 * URL contract:
 *   /project/{id}/pipeline/goldset
 *     ?focus_cluster_id=<id>
 *     &focus_hint=<one-line summary>
 *     &trap_count=<int>
 *
 * The destination ``LlmGoldGeneratePanel`` reads these and:
 *   1. Prefills its ``focusHint`` textarea (the field the LLM sees).
 *   2. Enables ``customizeMix`` + sets ``mix.hallucination_traps`` to
 *      ``trap_count``.
 *   3. Renders a dismissible banner so the user sees what triggered
 *      the prefill.
 */


/** Minimal shape we need from the FailureCluster — kept loose so the
 *  caller doesn't have to import the full FailureCluster type. */
export interface ClusterFixInput {
    cluster_id: string;
    reason_code?: string;
    output_pattern?: string;
    classifier_reason?: string;
    failure_count?: number;
    exemplars?: Array<{ prompt?: string; reference?: string; prediction?: string }>;
}


export interface ClusterFixRoute {
    path: string;
    search: string;
}


/** Default trap count when the caller doesn't specify. 5 is the
 *  prompt-spec default and matches what failure-cluster remediation
 *  typically prescribes — enough to anchor the pattern, not enough to
 *  swamp the gold set. */
const DEFAULT_TRAP_COUNT = 5;
const MAX_HINT_CHARS = 240;


/** Build a one-line focus_hint from a cluster. Prefers the classifier
 *  reason (LLM-generated explanation) → output_pattern → first
 *  exemplar prompt. Truncated to ``MAX_HINT_CHARS`` so the URL stays
 *  inside reasonable limits even on chunky exemplars.
 *
 *  Exported so the FailureClustersPanel can render the same hint
 *  preview next to the Fix button — keeps the click target honest
 *  about what the destination will see. */
export function buildClusterFocusHint(cluster: ClusterFixInput): string {
    const parts: string[] = [];
    if (cluster.reason_code) {
        parts.push(`[${cluster.reason_code}]`);
    }
    const body =
        (cluster.classifier_reason && cluster.classifier_reason.trim())
        || (cluster.output_pattern && cluster.output_pattern.trim())
        || cluster.exemplars?.[0]?.prompt?.trim()
        || '';
    if (body) parts.push(body);

    const hint = parts.join(' ').replace(/\s+/g, ' ').trim();
    if (hint.length <= MAX_HINT_CHARS) return hint;
    return hint.slice(0, MAX_HINT_CHARS - 1).trimEnd() + '…';
}


export function routeClusterFix(
    projectId: number,
    cluster: ClusterFixInput,
    options: { trapCount?: number } = {},
): ClusterFixRoute {
    const search = new URLSearchParams();
    search.set('focus_cluster_id', cluster.cluster_id);
    const hint = buildClusterFocusHint(cluster);
    if (hint) {
        search.set('focus_hint', hint);
    }
    const trapCount = Math.max(
        1,
        Math.min(20, Math.round(options.trapCount ?? DEFAULT_TRAP_COUNT)),
    );
    search.set('trap_count', String(trapCount));
    return {
        path: `/project/${projectId}/pipeline/goldset`,
        search: search.toString(),
    };
}
