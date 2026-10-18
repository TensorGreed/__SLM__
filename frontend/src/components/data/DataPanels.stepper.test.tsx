/**
 * Epic U phase U4: each data panel in ``layout="stepper"`` keeps its short
 * form on top and moves the rest into the Details drawer (own sections first,
 * the page's extra sections after).
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
    apiMock: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));
vi.mock('../../api/client', () => ({ default: apiMock }));
vi.mock('../../utils/workflowGraphPrefill', () => ({ loadWorkflowStagePrefill: async () => null, readWorkflowPrefill: () => null, consumeWorkflowPrefill: () => null }));
vi.mock('../coach/CoachStrip', () => ({ default: () => null }));
vi.mock('./LlmGoldGeneratePanel', () => ({ default: () => <div data-testid="llm-gold" /> }));
vi.mock('./GoldSetDiagnosticsPanel', () => ({ default: () => null }));
vi.mock('./SeedDocumentsCard', () => ({ default: () => <div data-testid="seed-documents" /> }));
vi.mock('./SavedMappingsPanel', () => ({ default: () => <div data-testid="saved-mappings" /> }));
vi.mock('./EDADashboard', () => ({ default: () => <div data-testid="eda" /> }));
vi.mock('../shared/ReadinessPanel', () => ({ ReadinessPanel: () => <div data-testid="readiness" /> }));
vi.mock('./LabelNoiseReviewPanel', () => ({ default: () => <div data-testid="label-noise" /> }));
vi.mock('./PlaybookPickerPanel', () => ({ default: () => <div data-testid="playbooks" /> }));
vi.mock('./SynthReviewQueue', () => ({ default: () => <div data-testid="review-queue" /> }));
vi.mock('./LlmSpendCard', () => ({ default: () => <div data-testid="spend" /> }));
vi.mock('./DataHealthReportPanel', () => ({ default: () => <div data-testid="health" /> }));
vi.mock('./TaskShapeConfirmCard', () => ({ default: () => <div data-testid="task-shape" /> }));

import IngestionPanel from './IngestionPanel';
import CleaningPanel from './CleaningPanel';
import GoldSetPanel from './GoldSetPanel';
import SyntheticPanel from './SyntheticPanel';
import DatasetPrepPanel from './DatasetPrepPanel';

const EXTRA = [{ id: 'extra', title: 'Extra section', render: () => <div data-testid="extra-body" /> }];

beforeEach(() => {
    apiMock.get.mockReset();
    apiMock.get.mockImplementation(async (url: string) => {
        if (url.endsWith('/projects/1')) return { data: { id: 1, name: 'P', selected_recipe: { recipe_id: 'qa-sft', scoring_mode: 'field_match' }, runtime_config: {} } };
        if (url.includes('/ingestion/documents')) return { data: [] };
        if (url.includes('/gold/entries')) return { data: { entries: [] } };
        if (url.includes('/synthetic/review-queue')) return { data: { total: 0, by_source: [], items: [] } };
        if (url.includes('/synthetic/playbooks')) return { data: { playbooks: [] } };
        if (url.includes('/synthetic/weak-types')) return { data: { types: [], all_types: [], suggested_rows: 0 } };
        if (url.includes('cleaning/chunks')) return { data: { chunks: [], total: 0 } };
        if (url.includes('/dataset/adapter-catalog')) return { data: { adapters: [], default_adapter: 'default-canonical' } };
        if (url.includes('/data-health')) return { data: { groups: [], overall: 'ok', severity_summary: { ok: 0, warn: 0, block: 0 }, total_signals: 0 } };
        if (url.includes('/task-shape')) return { data: { detection: null, confirmed: null, catalog: [] } };
        if (url.includes('/dataset/prepared-manifest') || url.includes('prepared-manifest')) return { data: null };
        return { data: {} };
    });
    apiMock.post.mockResolvedValue({ data: {} });
});

function sectionIds() {
    return Array.from(document.querySelectorAll('[data-testid^="data-details-section-"]')).map((el) => el.getAttribute('data-testid')!.replace('data-details-section-', ''));
}

async function openDrawer() {
    const user = userEvent.setup();
    await user.click(screen.getByTestId('data-details-toggle'));
    return user;
}

describe('Data panels — stepper layout', () => {
    it('Sources: seed / import / upload stay on top; documents, mappings, readiness, EDA and the extras go to Details', async () => {
        render(<MemoryRouter><IngestionPanel projectId={1} extraDetails={EXTRA} /></MemoryRouter>);
        expect(await screen.findByTestId('seed-documents')).toBeInTheDocument();
        expect(screen.getByTestId('open-import-wizard-btn')).toBeInTheDocument();
        expect(screen.queryByTestId('readiness')).toBeNull();
        expect(screen.queryByTestId('saved-mappings')).toBeNull();
        expect(screen.queryByText('Ingested Documents')).toBeNull();
        const user = await openDrawer();
        expect(sectionIds()).toEqual(['documents', 'mappings', 'readiness', 'eda', 'extra']);
        await user.click(screen.getByTestId('data-details-section-documents'));
        expect(screen.getByText('Ingested Documents')).toBeInTheDocument();
    });

    it('Prepare: the cleaning form stays; results, errors, label noise and extras go to Details', async () => {
        render(<MemoryRouter><CleaningPanel projectId={1} extraDetails={EXTRA} /></MemoryRouter>);
        expect(screen.getByText('Cleaning Configuration')).toBeInTheDocument();
        expect(screen.queryByTestId('label-noise')).toBeNull();
        const user = await openDrawer();
        expect(sectionIds()).toEqual(['results', 'errors', 'label-noise', 'extra']);
        await user.click(screen.getByTestId('data-details-section-label-noise'));
        expect(screen.getByTestId('label-noise')).toBeInTheDocument();
    });

    it('Labels: the answer-key card stays; the entries list goes to Details', async () => {
        render(<MemoryRouter><GoldSetPanel projectId={1} extraDetails={EXTRA} /></MemoryRouter>);
        expect(await screen.findByText('Gold Evaluation Dataset')).toBeInTheDocument();
        expect(screen.queryByTestId('gold-entries-header')).toBeNull();
        const user = await openDrawer();
        expect(sectionIds()).toEqual(['entries', 'extra']);
        await user.click(screen.getByTestId('data-details-section-entries'));
        expect(screen.getByTestId('gold-entries-header')).toBeInTheDocument();
    });

    it('Generate: the generation form stays; playbooks, review queue and spend go to Details', async () => {
        apiMock.get.mockImplementation(async (url: string) => {
            if (url.includes('/synthetic/review-queue')) return { data: { total: 0, by_source: [], items: [] } };
            if (url.includes('/synthetic/playbooks')) return { data: { playbooks: [{ playbook_id: 'p', display_name: 'P' }] } };
            if (url.includes('/synthetic/weak-types')) return { data: { types: [], all_types: [], suggested_rows: 0 } };
            if (url.includes('cleaning/chunks')) return { data: { chunks: [], total: 0 } };
            if (url.endsWith('/projects/1')) return { data: { id: 1, selected_recipe: { recipe_id: 'qa-sft', scoring_mode: 'field_match' } } };
            return { data: {} };
        });
        render(<MemoryRouter><SyntheticPanel projectId={1} extraDetails={EXTRA} /></MemoryRouter>);
        expect(await screen.findByText(/Synthetic Data Generation/)).toBeInTheDocument();
        expect(screen.queryByTestId('playbooks')).toBeNull();
        expect(screen.queryByTestId('spend')).toBeNull();
        const user = await openDrawer();
        expect(sectionIds()).toEqual(['playbooks', 'review-queue', 'spend', 'extra']);
        await user.click(screen.getByTestId('data-details-section-spend'));
        expect(screen.getByTestId('spend')).toBeInTheDocument();
    });

    it('Split: the split form stays; health, preview, adapters go to Details; no view tabs, no task-shape card', async () => {
        render(<MemoryRouter><DatasetPrepPanel projectId={1} onNextStep={() => undefined} extraDetails={EXTRA} /></MemoryRouter>);
        expect(await screen.findByText(/Train \/ Val \/ Test Split/)).toBeInTheDocument();
        expect(screen.queryByText('Adapter Lab')).toBeNull();
        expect(screen.queryByTestId('health')).toBeNull();
        expect(screen.queryByTestId('task-shape')).toBeNull();
        const user = await openDrawer();
        expect(sectionIds()).toEqual(['health', 'preview', 'adapters', 'extra']);
        await user.click(screen.getByTestId('data-details-section-health'));
        expect(screen.getByTestId('health')).toBeInTheDocument();
    });
});
