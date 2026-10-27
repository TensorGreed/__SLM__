/**
 * MultiTurnParamsForm — the knobs of the `documents_multi_turn` synth
 * playbook: the teacher writes whole dialogues over your document passages,
 * with follow-ups that depend on earlier turns and a cited passage in every
 * answer. Emits the `params` dict the run-playbook endpoint takes.
 */

import { useEffect, useState } from 'react';

import type { PlaybookParams } from '../../api/synthPlaybook';

interface Props {
    onChange: (params: PlaybookParams | null) => void;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.trunc(v) || lo));

export default function MultiTurnParamsForm({ onChange }: Props) {
    const [turnsMin, setTurnsMin] = useState(2);
    const [turnsMax, setTurnsMax] = useState(5);
    const [passages, setPassages] = useState(2);
    const [perGroup, setPerGroup] = useState(1);

    useEffect(() => {
        onChange({
            turns_min: turnsMin,
            turns_max: Math.max(turnsMin, turnsMax),
            passages_per_dialogue: passages,
            dialogues_per_passage_group: perGroup,
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [turnsMin, turnsMax, passages, perGroup]);

    return (
        <div className="code-findings-form" data-testid="multi-turn-form">
            <p className="u-hint u-mt0">
                The teacher writes a dialogue over your document passages: follow-up questions lean on earlier turns, and
                every answer cites a passage as [n]. Dialogues whose answers are not supported by the cited passage are
                rejected, grouped by reason.
            </p>
            <div className="playbook-picker__count">
                <label htmlFor="mt-turns-min">Min exchanges</label>
                <input id="mt-turns-min" type="number" min={2} max={8} value={turnsMin} data-testid="mt-turns-min"
                    onChange={(e) => setTurnsMin(clamp(Number(e.target.value), 2, 8))} />
                <label htmlFor="mt-turns-max">Max exchanges</label>
                <input id="mt-turns-max" type="number" min={2} max={8} value={turnsMax} data-testid="mt-turns-max"
                    onChange={(e) => setTurnsMax(clamp(Number(e.target.value), 2, 8))} />
            </div>
            <div className="playbook-picker__count">
                <label htmlFor="mt-passages">Passages per dialogue</label>
                <input id="mt-passages" type="number" min={1} max={3} value={passages} data-testid="mt-passages"
                    onChange={(e) => setPassages(clamp(Number(e.target.value), 1, 3))} />
                <label htmlFor="mt-per-group">Dialogues per passage group</label>
                <input id="mt-per-group" type="number" min={1} max={5} value={perGroup} data-testid="mt-per-group"
                    onChange={(e) => setPerGroup(clamp(Number(e.target.value), 1, 5))} />
            </div>
        </div>
    );
}
