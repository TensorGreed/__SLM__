/**
 * TokenLengthDistributionPanel — V3 of the ML-native visualisations arc.
 *
 * Overlays the token-length histograms of train / validation / test on
 * the same bucket axis so the user can see at a glance whether the
 * three splits have similar length distributions. A train set prepped
 * for short contexts that meets test rows twice as long is silently
 * losing every long row to truncation; reading that from p50/p95/p99
 * columns in three separate tables is muscle the user shouldn't need.
 *
 * Surfaces:
 *
 *  - **Grouped histogram** — one bar group per bucket
 *    (0-256 / 256-512 / 512-1024 / 1024-2048 / 2048+), three bars per
 *    group coloured per split. Y axis is sample count.
 *  - **Per-split percentile table** — p50/p95/p99 + truncation count
 *    + sample total, with the splits sorted train → val → test so
 *    the eye reads them left-to-right.
 *  - **Distribution-shift note** — when train's p95 is meaningfully
 *    smaller than test's p95 (more than 30% gap), a one-line honest
 *    beat: "test rows are longer than train — model trained for X
 *    tokens will silently truncate at eval". Below the gap threshold
 *    no note fires; spurious warnings are worse than no warning.
 *
 *  - **Missing splits chip** — if dataset prep hasn't finished, the
 *    panel renders the splits it has and surfaces "validation + test
 *    pending" instead of failing.
 *
 * Sources data from POST /api/projects/{id}/tokenization/analyze-splits,
 * which orchestrates the existing single-split analyze under the hood.
 */

import { useState } from 'react';
import { BarChart, Bar, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import api from '../../api/client';

interface SplitTokenStats {
    total_samples: number;
    p50_tokens: number;
    p95_tokens: number;
    p99_tokens: number;
    max_tokens: number;
    exceeding_max: number;
    max_seq_length: number;
    histogram: { bucket: string; count: number }[];
}

interface AnalyzeSplitsResponse {
    model_name: string;
    max_seq_length: number;
    splits: Record<string, SplitTokenStats>;
    missing_splits: string[];
    errors: Record<string, string>;
}

interface TokenLengthDistributionPanelProps {
    projectId: number;
    defaultModelName?: string;
    defaultMaxSeqLength?: number;
}

const SPLIT_ORDER = ['train', 'validation', 'test'] as const;
const SPLIT_COLOR: Record<(typeof SPLIT_ORDER)[number], string> = {
    train: '#2563eb',
    validation: '#16a34a',
    test: '#ea580c',
};

// Distribution-shift threshold. When test's p95 is more than this much
// larger than train's p95 (in proportional terms), surface the honest
// beat. 0.30 is empirically the smallest gap that's reliably above the
// step-to-step noise on real datasets and large enough to actually
// matter at training time — smaller gaps are within normal sampling
// variance and warning on them would cry wolf.
const DISTRIBUTION_SHIFT_THRESHOLD = 0.30;

export default function TokenLengthDistributionPanel({
    projectId,
    defaultModelName = 'HuggingFaceTB/SmolLM2-135M-Instruct',
    defaultMaxSeqLength = 2048,
}: TokenLengthDistributionPanelProps) {
    const [modelName, setModelName] = useState(defaultModelName);
    const [maxSeqLength, setMaxSeqLength] = useState(defaultMaxSeqLength);
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<AnalyzeSplitsResponse | null>(null);
    const [error, setError] = useState('');

    const run = async () => {
        setLoading(true);
        setError('');
        try {
            const res = await api.post<AnalyzeSplitsResponse>(
                `/projects/${projectId}/tokenization/analyze-splits`,
                { model_name: modelName, max_seq_length: maxSeqLength },
            );
            setData(res.data);
        } catch (err) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Failed to analyze splits.');
            setData(null);
        } finally {
            setLoading(false);
        }
    };

    // Combine the per-split histograms into one chart-friendly array
    // shape: [{bucket, train: n, validation: m, test: k}, ...]
    const chartData = (() => {
        if (!data) return [];
        const buckets = new Set<string>();
        for (const split of SPLIT_ORDER) {
            const hist = data.splits[split]?.histogram || [];
            for (const b of hist) buckets.add(b.bucket);
        }
        // Preserve the backend's bucket order: 0-256 → 256-512 → 512-1024 → 1024-2048 → 2048+
        const orderedBuckets = ['0-256', '256-512', '512-1024', '1024-2048', '2048+']
            .filter((b) => buckets.has(b));
        return orderedBuckets.map((bucket) => {
            const row: Record<string, string | number> = { bucket };
            for (const split of SPLIT_ORDER) {
                const hist = data.splits[split]?.histogram || [];
                const hit = hist.find((b) => b.bucket === bucket);
                row[split] = hit ? hit.count : 0;
            }
            return row;
        });
    })();

    // Detect distribution shift between train and test. We compare
    // p95s — they're the most stable summary above the long tail and
    // they're what max_seq_length is usually set against.
    const distributionShift = (() => {
        if (!data) return null;
        const train = data.splits.train;
        const test = data.splits.test;
        if (!train || !test || train.p95_tokens === 0) return null;
        const ratio = test.p95_tokens / train.p95_tokens;
        if (ratio < 1 + DISTRIBUTION_SHIFT_THRESHOLD) return null;
        return {
            trainP95: train.p95_tokens,
            testP95: test.p95_tokens,
            ratio,
        };
    })();

    const availableSplits = data
        ? SPLIT_ORDER.filter((s) => data.splits[s] !== undefined)
        : [];

    return (
        <section className="token-dist" data-testid="token-dist">
            <header className="token-dist__head">
                <h3 className="token-dist__title">Token-length distribution across splits</h3>
                <span className="token-dist__hint">
                    Overlays train / validation / test histograms on the same buckets so distribution skew across splits — the kind that silently loses long rows to truncation — lands in one glance.
                </span>
            </header>

            <div className="token-dist__controls">
                <label className="token-dist__field">
                    Model
                    <input
                        className="input"
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        placeholder="org/model-name"
                    />
                </label>
                <label className="token-dist__field">
                    max_seq_length
                    <input
                        className="input"
                        type="number"
                        min={128}
                        max={32768}
                        value={maxSeqLength}
                        onChange={(e) => setMaxSeqLength(Math.max(128, Math.min(32768, Number(e.target.value) || 2048)))}
                    />
                </label>
                <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => void run()}
                    disabled={loading || !modelName.trim()}
                    data-testid="token-dist-analyze"
                >
                    {loading ? 'Analyzing…' : 'Analyze splits'}
                </button>
            </div>

            {error && <div className="token-dist__error" data-testid="token-dist-error">{error}</div>}

            {data && (
                <>
                    {data.missing_splits.length > 0 && (
                        <div className="token-dist__missing" data-testid="token-dist-missing">
                            Splits not yet prepared:{' '}
                            <strong>{data.missing_splits.join(', ')}</strong>. Dataset prep
                            materialises train first; the overlay fills in as the rest land.
                        </div>
                    )}

                    {Object.keys(data.errors).length > 0 && (
                        <div className="token-dist__missing" data-testid="token-dist-errors">
                            Split analysis errored: {Object.entries(data.errors).map(([s, e]) => `${s} (${e})`).join(', ')}
                        </div>
                    )}

                    {chartData.length > 0 && (
                        <div className="token-dist__chart-wrap">
                            <ResponsiveContainer width="100%" height={260}>
                                <BarChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.18)" />
                                    <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                                    <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                                    <Tooltip />
                                    <Legend wrapperStyle={{ fontSize: 11 }} />
                                    {availableSplits.map((split) => (
                                        <Bar
                                            key={split}
                                            dataKey={split}
                                            fill={SPLIT_COLOR[split]}
                                            name={split}
                                        />
                                    ))}
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    <table className="token-dist__stats" data-testid="token-dist-stats">
                        <thead>
                            <tr>
                                <th>split</th>
                                <th>samples</th>
                                <th>p50</th>
                                <th>p95</th>
                                <th>p99</th>
                                <th>max</th>
                                <th>truncated</th>
                            </tr>
                        </thead>
                        <tbody>
                            {availableSplits.map((split) => {
                                const s = data.splits[split];
                                if (!s) return null;
                                const truncFrac = s.total_samples > 0 ? s.exceeding_max / s.total_samples : 0;
                                return (
                                    <tr key={split} data-testid={`token-dist-row-${split}`}>
                                        <td>
                                            <span
                                                className="token-dist__chip"
                                                style={{ background: SPLIT_COLOR[split] }}
                                            />
                                            {split}
                                        </td>
                                        <td>{s.total_samples}</td>
                                        <td>{s.p50_tokens}</td>
                                        <td>{s.p95_tokens}</td>
                                        <td>{s.p99_tokens}</td>
                                        <td>{s.max_tokens}</td>
                                        <td>
                                            {s.exceeding_max > 0 ? (
                                                <span className="token-dist__trunc">
                                                    {s.exceeding_max}{' '}
                                                    <span className="token-dist__trunc-frac">
                                                        ({(truncFrac * 100).toFixed(0)}%)
                                                    </span>
                                                </span>
                                            ) : (
                                                '0'
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>

                    {distributionShift && (
                        <div
                            className="token-dist__shift"
                            data-testid="token-dist-shift"
                        >
                            <strong>Distribution shift:</strong> test p95 = {distributionShift.testP95}{' '}
                            tokens vs train p95 = {distributionShift.trainP95} tokens
                            ({(distributionShift.ratio * 100).toFixed(0)}% of train). A model
                            trained with <code>max_seq_length={maxSeqLength}</code> will
                            silently truncate longer test rows at eval — either bump
                            max_seq_length, drop the long tail from test, or accept the
                            truncation as a known eval-time honest gap.
                        </div>
                    )}
                </>
            )}
        </section>
    );
}
