import { describe, expect, it } from 'vitest';

import { wrapTermsInBody } from './CoachSuggestion';

// These are unit tests for the term-wrap helper only — they don't
// need to render the full <CoachSuggestion> component or its toast
// store dependency. The component-level click → API behavior is
// covered in CoachStrip.test.tsx.

describe('wrapTermsInBody', () => {
    it('returns the raw text when no known terms appear', () => {
        const nodes = wrapTermsInBody('Your gold set has 50 rows.');
        // Single plain-string node.
        expect(nodes.length).toBe(1);
        expect(nodes[0]).toBe('Your gold set has 50 rows.');
    });

    it('wraps a single known term in a Term popover', () => {
        const nodes = wrapTermsInBody('Eval F1 was below target.');
        // Three nodes: "Eval " | <Term> | " was below target."
        expect(nodes.length).toBe(3);
        expect(nodes[0]).toBe('Eval ');
        // Middle node is a Term ReactElement — not a string.
        expect(typeof nodes[1]).toBe('object');
        const termNode = nodes[1] as { props: { id: string; label: string } };
        expect(termNode.props.id).toBe('f1');
        // Label preserves the original casing from the body so the
        // surrounding sentence reads naturally.
        expect(termNode.props.label).toBe('F1');
        expect(nodes[2]).toBe(' was below target.');
    });

    it('wraps multi-word terms before their single-word substrings', () => {
        // "Pass rate" must win over the bare "rate" or "pass" search.
        const nodes = wrapTermsInBody('Pass rate is 75% today.');
        const termNode = nodes.find(
            (n) => typeof n === 'object' && n !== null && 'props' in (n as object),
        ) as { props: { id: string; label: string } } | undefined;
        expect(termNode).toBeTruthy();
        expect(termNode!.props.id).toBe('pass_rate');
        expect(termNode!.props.label).toBe('Pass rate');
    });

    it('prefers the longest multi-word match when phrases overlap', () => {
        // "Predicted pass probability" must beat the embedded
        // "pass rate" sub-string.
        const nodes = wrapTermsInBody(
            'The predicted pass probability dropped this week.',
        );
        const termIds = nodes
            .filter((n) => typeof n === 'object' && n !== null && 'props' in (n as object))
            .map((n) => (n as { props: { id: string } }).props.id);
        // Exactly one match — the multi-word phrase — not a stray
        // "pass_rate" hit on the substring.
        expect(termIds).toEqual(['predicted_f1_confidence']);
    });

    it('wraps multiple distinct terms in the same body', () => {
        const nodes = wrapTermsInBody(
            'Class imbalance dropped F1 — Shannon entropy is now 0.42.',
        );
        const termIds = nodes
            .filter((n) => typeof n === 'object' && n !== null && 'props' in (n as object))
            .map((n) => (n as { props: { id: string } }).props.id);
        expect(termIds).toContain('class_imbalance');
        expect(termIds).toContain('f1');
        expect(termIds).toContain('shannon_entropy');
        // No stray "entropy" hit after the multi-word phrase has been
        // consumed.
        expect(termIds.filter((id) => id === 'shannon_entropy').length).toBe(1);
    });

    it('respects word boundaries so substrings do not trigger', () => {
        // "F12345" is a token / product code — not a metric reference;
        // and "passrate" without a separator shouldn't match either.
        const nodes = wrapTermsInBody('Run F12345 reported passrate metrics.');
        const termNodes = nodes.filter(
            (n) => typeof n === 'object' && n !== null && 'props' in (n as object),
        );
        expect(termNodes.length).toBe(0);
    });

    it('handles empty body gracefully', () => {
        expect(wrapTermsInBody('')).toEqual(['']);
    });
});
