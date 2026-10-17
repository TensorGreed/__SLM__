import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import TrainForm from './TrainForm';
import { readinessText, type TrainReadiness } from './trainFormText';

const READY: TrainReadiness = { state: 'ready', errors: [], warnings: [] };

function renderForm(overrides: Partial<React.ComponentProps<typeof TrainForm>> = {}) {
    const props: React.ComponentProps<typeof TrainForm> = {
        runName: '',
        onRunNameChange: vi.fn(),
        baseModel: 'microsoft/deberta-v3-base',
        onBaseModelChange: vi.fn(),
        onBrowseModels: vi.fn(),
        taskShapeLabel: 'Span extraction (span-extraction) · token_classification',
        onChangeTaskShape: vi.fn(),
        numSeeds: 1,
        onNumSeedsChange: vi.fn(),
        readiness: READY,
        onCheckAgain: vi.fn(),
        onFix: vi.fn(),
        onTrain: vi.fn(),
        sections: [
            { id: 'lora', title: 'LoRA', render: () => <div data-testid="section-lora-body">lora body</div> },
            { id: 'models', title: 'Base model registry', render: () => <div data-testid="section-models-body">registry</div> },
        ],
        ...overrides,
    };
    return { props, ...render(<TrainForm {...props} />) };
}

describe('TrainForm', () => {
    it('page one shows base model, task, seeds, readiness and one Train button; More options stays closed', () => {
        renderForm();
        expect(screen.getByTestId('train-form-base-model')).toHaveValue('microsoft/deberta-v3-base');
        expect(screen.getByTestId('train-form-task')).toHaveTextContent('Span extraction (span-extraction)');
        expect(screen.getByTestId('train-form-seeds-1')).toBeChecked();
        expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Ready to train.');
        expect(screen.getByTestId('train-form-train')).toHaveTextContent('Train');
        expect(screen.queryByTestId('section-lora-body')).toBeNull();
        expect(screen.queryByTestId('train-section-lora')).toBeNull();
    });

    it('three seeds flips the count and the button label', async () => {
        const user = userEvent.setup();
        const { props } = renderForm();
        await user.click(screen.getByTestId('train-form-seeds-3'));
        expect(props.onNumSeedsChange).toHaveBeenCalledWith(3);
        renderForm({ numSeeds: 3 });
        expect(screen.getAllByTestId('train-form-train').at(-1)).toHaveTextContent('Train 3 seeds');
    });

    it('a blocked readiness lists the issues and offers Fix', async () => {
        const user = userEvent.setup();
        const { props } = renderForm({ readiness: { state: 'blocked', errors: ['No split yet', 'Model too big'], warnings: [] } });
        expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('2 blocking issues.');
        expect(screen.getByTestId('train-form-issues')).toHaveTextContent('No split yet');
        await user.click(screen.getByTestId('train-form-fix'));
        expect(props.onFix).toHaveBeenCalled();
    });

    it('Train is disabled while checking', () => {
        renderForm({ readiness: { state: 'checking', errors: [], warnings: [] }, trainDisabled: true });
        expect(screen.getByTestId('train-form-train')).toBeDisabled();
        expect(screen.getByTestId('train-form-readiness')).toHaveTextContent('Checking');
    });

    it('More options opens sections on request and renders a section body only while open', async () => {
        const user = userEvent.setup();
        renderForm();
        await user.click(screen.getByTestId('train-form-more'));
        expect(screen.getByTestId('train-section-lora')).toBeInTheDocument();
        expect(screen.queryByTestId('section-lora-body')).toBeNull();
        await user.click(screen.getByTestId('train-section-lora'));
        expect(screen.getByTestId('section-lora-body')).toBeInTheDocument();
        await user.click(screen.getByTestId('train-section-lora'));
        expect(screen.queryByTestId('section-lora-body')).toBeNull();
    });

    it('openSection opens More options and that section on mount', () => {
        renderForm({ openSection: 'models' });
        expect(screen.getByTestId('section-models-body')).toBeInTheDocument();
        expect(screen.queryByTestId('section-lora-body')).toBeNull();
    });

    it('Browse and Change call back to the owner', async () => {
        const user = userEvent.setup();
        const { props } = renderForm();
        await user.click(screen.getByTestId('train-form-browse'));
        await user.click(screen.getByTestId('train-form-change-task'));
        expect(props.onBrowseModels).toHaveBeenCalled();
        expect(props.onChangeTaskShape).toHaveBeenCalled();
    });

    it('readinessText covers every state', () => {
        expect(readinessText({ state: 'idle', errors: [], warnings: [] })).toBe('Not checked yet.');
        expect(readinessText({ state: 'ready', errors: [], warnings: ['w'] })).toBe('Ready to train, 1 warning.');
        expect(readinessText({ state: 'blocked', errors: ['e'], warnings: [] })).toBe('1 blocking issue.');
        expect(readinessText({ state: 'error', errors: [], warnings: [], message: 'boom' })).toBe('boom');
    });
});
