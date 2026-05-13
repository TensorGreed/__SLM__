/**
 * Phase 5.3.4b — Sample Predictions card in span_set scoring mode.
 *
 * When the prediction row carries `scoring_mode: "span_set"`, the
 * structured-extraction surface swaps the per-field comparison for an
 * entity-by-entity view (matched / missed / hallucinated). This is
 * the load-bearing UI for PII / NER / span-extraction tasks where
 * per-class recall is what compliance teams actually evaluate.
 *
 * Tests confirm:
 *   - The inline counts switch to "X matched · Y missed · Z hallucinated".
 *   - The disclosure renders an entity table with type / text / offset
 *     for every matched, missed, and hallucinated entity.
 *   - field_match mode still renders the per-field table (regression
 *     guard — adding span_set must not break invoice-style extraction).
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
                        pass_rate: 0.5,
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
                    is_ship: false,
                    decision: 'NO-SHIP',
                    reasons: [],
                    failed_gates: [],
                    missing_metrics: [],
                    gate_report: {
                        passed: false,
                        checks: [],
                        missing_required_metrics: [],
                        failed_gate_ids: [],
                    },
                },
            };
        }
        if (url.includes('/evaluation/packs')) return { data: { packs: [] } };
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
                    passed: false,
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


describe('EvalPanel — Sample Predictions in span_set scoring mode', () => {
    beforeEach(() => {
        apiMock.get.mockReset();
        apiMock.post.mockReset();
    });

    it('shows matched / missed / hallucinated entity counts inline', async () => {
        mockApi([
            {
                prompt: 'Hi, my name is Jane Doe at jane@example.com',
                reference:
                    '{"entities": [{"type":"person_name","start":15,"end":23,"text":"Jane Doe"},{"type":"email","start":27,"end":43,"text":"jane@example.com"}]}',
                prediction:
                    '{"entities": [{"type":"person_name","start":15,"end":23,"text":"Jane Doe"},{"type":"ssn","start":99,"end":110,"text":"fake"}]}',
                is_valid_json: true,
                scoring_mode: 'span_set',
                row_matched_entities: [
                    { type: 'person_name', start: 15, end: 23, text: 'Jane Doe' },
                ],
                row_missed_entities: [
                    { type: 'email', start: 27, end: 43, text: 'jane@example.com' },
                ],
                row_hallucinated_entities: [
                    { type: 'ssn', start: 99, end: 110, text: 'fake' },
                ],
                row_precision: 0.5,
                row_recall: 0.5,
                row_f1: 0.5,
                row_exact_match: 0.0,
                missing_required_fields: [],
                parsed_prediction: {},
                parsed_reference: {},
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // Inline counts pick up the entity tallies.
        expect(await screen.findByText(/1 matched/i)).toBeInTheDocument();
        expect(screen.getByText(/1 missed/i)).toBeInTheDocument();
        expect(screen.getByText(/1 hallucinated/i)).toBeInTheDocument();
        // Per-row P/R inline so the engineer can see the row's score.
        expect(screen.getByText(/P 0.50/)).toBeInTheDocument();
        expect(screen.getByText(/R 0.50/)).toBeInTheDocument();
    });

    it('renders the entity-by-entity disclosure with type / text / offsets', async () => {
        mockApi([
            {
                prompt: 'Contact info',
                reference: '{"entities": [{"type":"email","start":10,"end":25,"text":"a@example.com"}]}',
                prediction: '{"entities": [{"type":"email","start":10,"end":25,"text":"a@example.com"}]}',
                is_valid_json: true,
                scoring_mode: 'span_set',
                row_matched_entities: [
                    { type: 'email', start: 10, end: 25, text: 'a@example.com' },
                ],
                row_missed_entities: [],
                row_hallucinated_entities: [],
                row_precision: 1.0,
                row_recall: 1.0,
                row_f1: 1.0,
                row_exact_match: 1.0,
                missing_required_fields: [],
                parsed_prediction: {},
                parsed_reference: {},
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // The disclosure summary is present.
        expect(
            await screen.findByText(/Show entity-by-entity breakdown/i),
        ).toBeInTheDocument();
        // The matched entity shows up in the table (details > table is
        // kept mounted even when collapsed).
        expect(screen.getByText('email')).toBeInTheDocument();
        expect(screen.getByText('a@example.com')).toBeInTheDocument();
        expect(screen.getByText('10–25')).toBeInTheDocument();
        expect(screen.getByText(/✓ matched/i)).toBeInTheDocument();
    });

    it('shows missed + hallucinated rows in the breakdown', async () => {
        mockApi([
            {
                prompt: 'Mixed',
                reference: '{"entities": [{"type":"phone","start":0,"end":8,"text":"555-0100"}]}',
                prediction: '{"entities": [{"type":"ssn","start":0,"end":11,"text":"000-12-3456"}]}',
                is_valid_json: true,
                scoring_mode: 'span_set',
                row_matched_entities: [],
                row_missed_entities: [
                    { type: 'phone', start: 0, end: 8, text: '555-0100' },
                ],
                row_hallucinated_entities: [
                    { type: 'ssn', start: 0, end: 11, text: '000-12-3456' },
                ],
                row_precision: 0.0,
                row_recall: 0.0,
                row_f1: 0.0,
                row_exact_match: 0.0,
                missing_required_fields: [],
                parsed_prediction: {},
                parsed_reference: {},
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        expect(await screen.findByText(/✗ missed/i)).toBeInTheDocument();
        expect(screen.getByText(/✗ hallucinated/i)).toBeInTheDocument();
        // The hallucinated entity is in the breakdown.
        expect(screen.getByText('000-12-3456')).toBeInTheDocument();
        // The missed entity is in the breakdown.
        expect(screen.getByText('555-0100')).toBeInTheDocument();
    });

    it('falls back to the field-by-field view for field_match mode (no regression)', async () => {
        mockApi([
            {
                prompt: 'Invoice scan',
                reference: '{"invoice_no": "INV-001", "total": "50"}',
                prediction: '{"invoice_no": "INV-001", "total": "50"}',
                is_valid_json: true,
                // No scoring_mode → field_match path. The previous
                // structured-extraction UI must keep rendering.
                row_field_results: {
                    invoice_no: { em: 1.0, f1: 1.0 },
                    total: { em: 1.0, f1: 1.0 },
                },
                missing_required_fields: [],
                parsed_prediction: { invoice_no: 'INV-001', total: '50' },
                parsed_reference: { invoice_no: 'INV-001', total: '50' },
                row_exact_match: 1.0,
                row_f1: 1.0,
            },
        ]);

        const user = userEvent.setup();
        render(<EvalPanel projectId={4} />);
        await user.click(await screen.findByRole('button', { name: 'exp-21' }));

        // The per-field count + disclosure render, NOT the entity-by-entity view.
        expect(await screen.findByText(/2\/2 fields/i)).toBeInTheDocument();
        expect(
            screen.queryByText(/Show entity-by-entity breakdown/i),
        ).not.toBeInTheDocument();
        expect(
            screen.getByText(/Show field-by-field comparison/i),
        ).toBeInTheDocument();
    });
});
