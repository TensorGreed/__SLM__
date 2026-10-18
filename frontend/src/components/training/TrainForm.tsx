/**
 * The Train form (Epic U, phase U2) — "an insurance form": page one asks
 * only what a run needs (base model, task shape, seeds) and shows one
 * readiness line and one Train button; everything else sits behind
 * "More options" and opens on request. Presentational: TrainingPanel owns
 * the state, the preflight and the launch.
 */

import { useEffect, useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import type { TrainFormSection } from './trainingPanelTypes';
import { readinessText, type TrainReadiness } from './trainFormText';
import './TrainForm.css';

export type { TrainReadiness } from './trainFormText';

interface TrainFormProps {
    runName: string;
    onRunNameChange: (value: string) => void;
    baseModel: string;
    onBaseModelChange: (value: string) => void;
    onBrowseModels: () => void;
    /** "Span extraction · token classification" — read-only, prefilled. */
    taskShapeLabel: string;
    onChangeTaskShape: () => void;
    numSeeds: number;
    onNumSeedsChange: (value: number) => void;
    readiness: TrainReadiness;
    onCheckAgain: () => void;
    onFix: () => void;
    onTrain: () => void;
    trainDisabled?: boolean;
    trainBusy?: boolean;
    sections: TrainFormSection[];
    /** Section to open (``?open=`` on mount; Browse / Change / Fix later). A
     *  new ``openNonce`` re-opens the same section. */
    openSection?: string | null;
    openNonce?: number;
    /** Rendered under the form: errors, warnings, resolved defaults. */
    notices?: ReactNode;
}

export default function TrainForm({
    runName,
    onRunNameChange,
    baseModel,
    onBaseModelChange,
    onBrowseModels,
    taskShapeLabel,
    onChangeTaskShape,
    numSeeds,
    onNumSeedsChange,
    readiness,
    onCheckAgain,
    onFix,
    onTrain,
    trainDisabled = false,
    trainBusy = false,
    sections,
    openSection = null,
    openNonce = 0,
    notices,
}: TrainFormProps) {
    const [open, setOpen] = useState<Set<string>>(() => new Set(openSection ? [openSection] : []));
    const [moreOpen, setMoreOpen] = useState<boolean>(Boolean(openSection));
    // A new open request (section or nonce) opens that section — derived
    // during render, the React way for props-driven state.
    const [seenRequest, setSeenRequest] = useState(`${openSection ?? ''}#${openNonce}`);
    const request = `${openSection ?? ''}#${openNonce}`;
    if (request !== seenRequest) {
        setSeenRequest(request);
        if (openSection) {
            setOpen((prev) => (prev.has(openSection) ? prev : new Set([...prev, openSection])));
            setMoreOpen(true);
        }
    }

    useEffect(() => {
        if (!openSection) return;
        const timer = window.setTimeout(() => {
            document.getElementById(`train-section-${openSection}`)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
        }, 50);
        return () => window.clearTimeout(timer);
    }, [openSection, openNonce]);

    const toggleSection = (id: string) => {
        setOpen((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    return (
        <div className="train-form" data-testid="train-form">
            <section className="card train-form__page">
                <div className="train-form__row">
                    <label className="train-form__label" htmlFor="train-form-base-model">Base model</label>
                    <div className="train-form__control">
                        <input
                            id="train-form-base-model"
                            className="input"
                            value={baseModel}
                            onChange={(e) => onBaseModelChange(e.target.value)}
                            placeholder="owner/model-name"
                            data-testid="train-form-base-model"
                        />
                        <button type="button" className="btn btn-secondary btn-sm" onClick={onBrowseModels} data-testid="train-form-browse">
                            Browse
                        </button>
                    </div>
                </div>

                <div className="train-form__row">
                    <span className="train-form__label">Task</span>
                    <div className="train-form__control train-form__control--static">
                        <span data-testid="train-form-task">{taskShapeLabel}</span>
                        <button type="button" className="btn btn-ghost btn-sm" onClick={onChangeTaskShape} data-testid="train-form-change-task">
                            Change
                        </button>
                    </div>
                </div>

                <div className="train-form__row">
                    <span className="train-form__label">Seeds</span>
                    <div className="train-form__control train-form__control--static" role="radiogroup" aria-label="Seeds">
                        <label className="train-form__radio">
                            <input type="radio" name="train-form-seeds" checked={numSeeds <= 1} onChange={() => onNumSeedsChange(1)} data-testid="train-form-seeds-1" />
                            One run
                        </label>
                        <label className="train-form__radio">
                            <input type="radio" name="train-form-seeds" checked={numSeeds === 3} onChange={() => onNumSeedsChange(3)} data-testid="train-form-seeds-3" />
                            Three seeds
                            <small>pick the best by validation</small>
                        </label>
                        {numSeeds > 1 && numSeeds !== 3 && (
                            <span className="train-form__muted" data-testid="train-form-seeds-custom">{numSeeds} seeds (set under More options)</span>
                        )}
                    </div>
                </div>

                <div className="train-form__row">
                    <label className="train-form__label" htmlFor="train-form-run-name">Run name</label>
                    <div className="train-form__control">
                        <input
                            id="train-form-run-name"
                            className="input"
                            value={runName}
                            onChange={(e) => onRunNameChange(e.target.value)}
                            placeholder="optional, named after the date otherwise"
                            data-testid="train-form-run-name"
                        />
                    </div>
                </div>

                <div className={`train-form__readiness train-form__readiness--${readiness.state}`} data-testid="train-form-readiness">
                    <span className="train-form__readiness-dot" aria-hidden="true" />
                    <span className="train-form__readiness-text">{readinessText(readiness)}</span>
                    {readiness.state === 'blocked' && (
                        <button type="button" className="btn btn-ghost btn-sm" onClick={onFix} data-testid="train-form-fix">
                            Fix
                        </button>
                    )}
                    {readiness.state !== 'checking' && (
                        <button type="button" className="btn btn-ghost btn-sm" onClick={onCheckAgain} data-testid="train-form-check">
                            Check again
                        </button>
                    )}
                </div>
                {readiness.state === 'blocked' && readiness.errors.length > 0 && (
                    <ul className="train-form__issues" data-testid="train-form-issues">
                        {readiness.errors.slice(0, 4).map((issue) => (
                            <li key={issue}>{issue}</li>
                        ))}
                    </ul>
                )}

                <div className="train-form__actions">
                    <button
                        type="button"
                        className="btn btn-primary btn-lg"
                        onClick={onTrain}
                        disabled={trainDisabled || trainBusy}
                        data-testid="train-form-train"
                    >
                        {trainBusy ? 'Creating the run…' : numSeeds > 1 ? `Train ${numSeeds} seeds` : 'Train'}
                    </button>
                </div>

                {notices}
            </section>

            <section className="card train-form__more">
                <button
                    type="button"
                    className="train-form__more-toggle"
                    onClick={() => setMoreOpen((v) => !v)}
                    aria-expanded={moreOpen ? 'true' : 'false'}
                    data-testid="train-form-more"
                >
                    {moreOpen ? <ChevronDown className="u-icon" aria-hidden="true" /> : <ChevronRight className="u-icon" aria-hidden="true" />} More options
                    <small>LoRA, epochs, cloud burst, OCR copies, presets, checks, the model registry…</small>
                </button>
                {moreOpen && (
                    <div className="train-form__sections">
                        {sections.map((section) => {
                            const isOpen = open.has(section.id);
                            return (
                                <div key={section.id} className={`train-form__section ${isOpen ? 'is-open' : ''}`} id={`train-section-${section.id}`}>
                                    <button
                                        type="button"
                                        className="train-form__section-toggle"
                                        onClick={() => toggleSection(section.id)}
                                        aria-expanded={isOpen ? 'true' : 'false'}
                                        data-testid={`train-section-${section.id}`}
                                    >
                                        <span>{section.title}</span>
                                        {section.hint && <small>{section.hint}</small>}
                                    </button>
                                    {isOpen && <div className="train-form__section-body">{section.render()}</div>}
                                </div>
                            );
                        })}
                    </div>
                )}
            </section>
        </div>
    );
}
