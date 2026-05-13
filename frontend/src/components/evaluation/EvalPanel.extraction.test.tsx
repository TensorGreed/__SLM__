/**
 * Phase 5.3.4 — Sample Predictions card with structured-extraction enrichment.
 *
 * StructuredExtractionHandler writes is_valid_json / missing_required_fields /
 * row_field_results / parsed_prediction / parsed_reference onto each
 * prediction. The UI surfaces those as:
 *
 *   - a "JSON: valid" / "JSON: malformed" badge inline below the prediction
 *   - an "X/Y fields" inline count when the JSON parsed
 *   - a red "missing: foo, bar" note when required fields are missing
 *   - a "Show field-by-field comparison" disclosure with a per-field table
 *
 * Other handlers don't write these fields, so the structured-extraction
 * surface is conditionally hidden for them (no regression for QA /
 * classification / seq2seq runs).
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    },
}));

vi.mock('../../api/client', () => ({
    default: apiMock,
}));

import EvalPanel from './EvalPanel';


function mockApi(predictionsPreview: unknown[]) {
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.includes('/training/experiments')) {
            return { data: [{ id: 21, name: 'exp-21' }] };
        }
        if (url.includes('/evaluation/results/21')) {
            return {
                data: [
                    {
                        id: 1001,
                        dataset_name: 'gold_dev',
                        eval_type: 'f1',
                        pass_rate: 0.75,
                        metrics: {},
                        details: {
                            predictions_preview: predictionsPreview,
                            handler_id: 'structured_extraction',
                            task_profile_resolved: 'structured_extraction',
                        },
                    },
                ],
            };
        }
        if (url.includes('/evaluation/safety-scorecard/21')) {
            return { data: { overall_risk: 'low', red_flags: [] } };
        }
        if (url.includes('/evaluation/scorecard/21')) {
            return {
                data: {
                    experiment_id: 21,
                    is_ship: true,
                    decision: 'SHIP',
                    reasons: [],
                    failed_gates: [],
                    missing_metrics: [],
                    gate_report: {
                        passed: true,
                        checks: [],
                        missing_required_metrics: [],
                        failed_gate_ids: [],
                    },
                },
            };
        }
        if (url.includes('/evaluation/packs')) {
            return { data: { packs: [] } };
        }
        if (url.includes('/evaluation/pack-preference')) {
            return {
                data: {
                    preferred_pack_id: null,
                    active_pack_id: null,
                    active_pack_source: 'default',
                    active_pack: { display_name: 'Auto' },
                },
            };
        }
        if (url.includes('/evaluation/gates/21')) {
            return {
                data: {
                    captured_at: '2026-03-26T00:00:00Z',
                    passed: true,
                    failed_gate_ids: [],
                    missing_required_metrics: [],
                    checks: [],
                },
            };
        }
        if (url.includes('failure-clusters')) {
            return {
                data: {
                    eval_result_id: 1001,
                    experiment_id: 21,
                    dataset_name: 'gold_dev',
                    eval_type: 'f1',
                    total_failures_analyzed: 0,
                    reason_code_totals: {},
                    dominant_reason_code: null,
                    clusters: [],
                    remediation_plans: [],
                },
            };
        }
        if (url.includes('remediation-plans')) {
            return { data: { project_id: 4, count: 0, plans: [] } };
        }
        return { data: {} };
    });
}


describe('EvalPanel — Sample Predictions with StructuredExtractionHandler enrichment', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('renders "JSON: valid" badge + field count for a clean extraction', async () => {
        mockApi([
            {
                prompt: 'Extract invoice fields from: Receipt $50 INV-001',
                reference: '{"invoice_no": "INV-001", "total": "50"}',
                prediction: '{"invoice_no": "INV-001", "total": "50"}',
                is_valid_json: true,
                parsed_prediction: { invoice_no: 'INV-001', total: '50' },
                parsed_reference: { invoice_no: 'INV-001', total: '50' },
                missing_required_fields: [],
                row_field_results: {
                    invoice_no: { em: 1.0, f1: 1.0 },
                    total: { em: 1.0, f1: 1.0 },
                },
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/JSON: valid/i)).toBeInTheDocument();
        expect(screen.getByText(/2\/2 fields/i)).toBeInTheDocument();
    });

    it('renders "JSON: malformed" badge for unparseable predictions', async () => {
        mockApi([
            {
                prompt: 'Extract: something',
                reference: '{"a": 1}',
                prediction: 'not json at all',
                is_valid_json: false,
                parsed_prediction: null,
                parsed_reference: { a: 1 },
                missing_required_fields: [],
                row_field_results: {},
                row_exact_match: 0.0,
                row_f1: 0.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/JSON: malformed/i)).toBeInTheDocument();
    });

    it('flags missing required fields inline in red', async () => {
        mockApi([
            {
                prompt: 'Extract invoice',
                reference: '{"invoice_no": "X", "total": "1"}',
                prediction: '{"invoice_no": "X"}',
                is_valid_json: true,
                parsed_prediction: { invoice_no: 'X' },
                parsed_reference: { invoice_no: 'X', total: '1' },
                missing_required_fields: ['total'],
                row_field_results: {
                    invoice_no: { em: 1.0, f1: 1.0 },
                },
                row_exact_match: 0.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/missing:\s*total/i)).toBeInTheDocument();
    });

    it('renders the field-by-field disclosure with per-field EM markers', async () => {
        mockApi([
            {
                prompt: 'Extract invoice',
                reference:
                    '{"invoice_no": "INV-001", "total": "50", "date": "2026-01-01"}',
                prediction:
                    '{"invoice_no": "INV-001", "total": "WRONG", "date": "2026-01-01"}',
                is_valid_json: true,
                parsed_prediction: {
                    invoice_no: 'INV-001',
                    total: 'WRONG',
                    date: '2026-01-01',
                },
                parsed_reference: {
                    invoice_no: 'INV-001',
                    total: '50',
                    date: '2026-01-01',
                },
                missing_required_fields: [],
                row_field_results: {
                    invoice_no: { em: 1.0, f1: 1.0 },
                    total: { em: 0.0, f1: 0.0 },
                    date: { em: 1.0, f1: 1.0 },
                },
                row_exact_match: 0.0,
                row_f1: 0.6667,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // The disclosure is collapsed by default; its summary is visible.
        expect(
            await screen.findByText(/Show field-by-field comparison/i),
        ).toBeInTheDocument();
        // The table body renders with field names + EM markers.
        // (details/summary keeps children mounted, just visually hidden.)
        expect(screen.getByText('invoice_no')).toBeInTheDocument();
        expect(screen.getByText('total')).toBeInTheDocument();
        expect(screen.getByText('date')).toBeInTheDocument();
    });

    it('omits the structured surface entirely for non-extraction runs', async () => {
        // No is_valid_json on the row → handler didn't enrich → don't
        // render the structured-extraction UI elements (preserves the
        // QA / classification / generic layout).
        mockApi([
            {
                prompt: 'What is 2+2?',
                reference: '4',
                prediction: '4',
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // Wait for the predictions card to land.
        expect(await screen.findByText(/What is 2\+2/i)).toBeInTheDocument();
        // No JSON-validity badge for non-extraction runs.
        expect(screen.queryByText(/JSON: valid/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/JSON: malformed/i)).not.toBeInTheDocument();
        expect(
            screen.queryByText(/Show field-by-field comparison/i),
        ).not.toBeInTheDocument();
    });
});
