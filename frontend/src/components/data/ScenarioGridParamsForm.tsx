/**
 * ScenarioGridParamsForm — the knobs of the `structured-plan` `scenario_grid`
 * synth playbook.
 *
 * A matrix file (a markdown pipe table or CSV) is read in the browser: every
 * column is a dimension (header = name, cells below = its values) and the grid
 * is the cartesian product. The teacher writes scenarios per grid cell, then a
 * plan for each, and only plans that validate against the saved schema are
 * kept. The preview mirrors the server's parse rule; the server has the final
 * say. Emits the `params` dict the run-playbook endpoint takes.
 */

import { useEffect, useMemo, useState } from 'react';

import type { PlaybookParams } from '../../api/synthPlaybook';

export interface MatrixDimension {
    name: string;
    values: string[];
}

export interface ParsedMatrix {
    dimensions: MatrixDimension[];
    cells: number;
    error: string | null;
}

const MAX_DIMENSIONS = 10;
const MAX_VALUES = 100;
const MAX_TEXT = 200 * 1024;

function splitCsvLine(line: string): string[] {
    const out: string[] = [];
    let cur = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (quoted) {
            if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
            else if (ch === '"') quoted = false;
            else cur += ch;
        } else if (ch === '"') quoted = true;
        else if (ch === ',') { out.push(cur.trim()); cur = ''; }
        else cur += ch;
    }
    out.push(cur.trim());
    return out;
}

function splitPipeLine(line: string): string[] {
    let t = line.trim();
    if (t.startsWith('|')) t = t.slice(1);
    if (t.endsWith('|')) t = t.slice(0, -1);
    return t.split('|').map((c) => c.trim());
}

const SEPARATOR = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

/** Client-side mirror of the backend's matrix rule; each column is a dimension. */
export function parseMatrix(text: string, format: 'auto' | 'markdown' | 'csv' = 'auto'): ParsedMatrix {
    const fail = (error: string): ParsedMatrix => ({ dimensions: [], cells: 0, error });
    if (text.length > MAX_TEXT) return fail('The matrix file is larger than 200 KB.');
    const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
    if (lines.length === 0) return fail('The matrix has no header row.');
    const fmt = format === 'auto' ? (lines[0].includes('|') ? 'markdown' : 'csv') : format;
    const rows = lines
        .filter((l) => !(fmt === 'markdown' && SEPARATOR.test(l)))
        .map((l) => (fmt === 'markdown' ? splitPipeLine(l) : splitCsvLine(l)));
    const header = rows[0].filter((h) => h !== '');
    if (header.length < 1) return fail('The matrix has no header row.');
    if (header.length > MAX_DIMENSIONS) return fail(`At most ${MAX_DIMENSIONS} dimensions are allowed.`);
    const dimensions: MatrixDimension[] = rows[0].map((name) => ({ name, values: [] as string[] })).filter((d) => d.name !== '');
    const indexes = rows[0].map((n, i) => (n !== '' ? i : -1)).filter((i) => i >= 0);
    for (const row of rows.slice(1)) {
        indexes.forEach((col, d) => {
            const v = (row[col] ?? '').trim();
            if (v && !dimensions[d].values.includes(v)) dimensions[d].values.push(v);
        });
    }
    for (const d of dimensions) {
        if (d.values.length === 0) return fail(`Dimension "${d.name}" has no values.`);
        if (d.values.length > MAX_VALUES) return fail(`Dimension "${d.name}" has more than ${MAX_VALUES} values.`);
    }
    return { dimensions, cells: dimensions.reduce((n, d) => n * d.values.length, 1), error: null };
}

interface Props {
    onChange: (params: PlaybookParams | null) => void;
}

function readText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ''));
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
}

export default function ScenarioGridParamsForm({ onChange }: Props) {
    const [text, setText] = useState('');
    const [fileName, setFileName] = useState<string | null>(null);
    const [readError, setReadError] = useState<string | null>(null);
    const [maxCells, setMaxCells] = useState(20);
    const [perCell, setPerCell] = useState(1);
    const [seed, setSeed] = useState(0);
    const [extra, setExtra] = useState('');

    const parsed = useMemo(() => (text ? parseMatrix(text) : null), [text]);
    const valid = parsed != null && parsed.error == null;

    useEffect(() => {
        if (!valid) {
            onChange(null);
            return;
        }
        const params: PlaybookParams = {
            matrix_text: text,
            matrix_format: 'auto',
            max_cells: maxCells,
            scenarios_per_cell: perCell,
            seed,
        };
        if (extra.trim()) params.extra_instructions = extra.trim();
        onChange(params);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [valid, text, maxCells, perCell, seed, extra]);

    const onFile = async (list: FileList | null) => {
        const file = list?.[0];
        if (!file) return;
        try {
            setText(await readText(file));
            setFileName(file.name);
            setReadError(null);
        } catch {
            setReadError(`${file.name} could not be read.`);
        }
    };

    const used = parsed && valid ? Math.min(parsed.cells, maxCells) : 0;
    return (
        <div className="code-findings-form" data-testid="scenario-grid-form">
            <p className="u-hint u-mt0">
                Upload a matrix: a markdown table or CSV where each column is a dimension and the cells below it are its
                values. The teacher writes scenarios for each combination, then a plan for each; only plans that validate
                against your saved schema are kept. The test answer key is never used.
            </p>
            <div className="playbook-picker__count">
                <label htmlFor="sg-file">Matrix file</label>
                <input id="sg-file" type="file" accept=".md,.csv,.txt,text/csv,text/markdown,text/plain" data-testid="sg-file"
                    onChange={(e) => { void onFile(e.target.files); e.target.value = ''; }} />
            </div>
            {readError && <p className="u-hint" data-testid="sg-read-error">{readError}</p>}
            {parsed?.error && <p className="u-hint" data-testid="sg-parse-error">{parsed.error}</p>}
            {parsed && valid && (
                <div data-testid="sg-preview">
                    <p className="u-hint" data-testid="sg-cells">
                        {fileName ? `${fileName}: ` : ''}{parsed.dimensions.length} dimension{parsed.dimensions.length === 1 ? '' : 's'},{' '}
                        {parsed.cells} cell{parsed.cells === 1 ? '' : 's'}
                        {parsed.cells > maxCells ? `; ${used} will be sampled` : ''}
                    </p>
                    <ul className="u-hint" data-testid="sg-dimensions">
                        {parsed.dimensions.map((d) => (
                            <li key={d.name} data-testid={`sg-dim-${d.name}`}>
                                <code>{d.name}</code> ({d.values.length}): {d.values.join(', ')}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
            <div className="playbook-picker__count">
                <label htmlFor="sg-max-cells">Cells to use</label>
                <input id="sg-max-cells" type="number" min={1} max={200} value={maxCells}
                    onChange={(e) => setMaxCells(Math.max(1, Math.min(200, Number(e.target.value) || 1)))} data-testid="sg-max-cells" />
                <label htmlFor="sg-per-cell">Scenarios per cell</label>
                <input id="sg-per-cell" type="number" min={1} max={5} value={perCell}
                    onChange={(e) => setPerCell(Math.max(1, Math.min(5, Number(e.target.value) || 1)))} data-testid="sg-per-cell" />
                <label htmlFor="sg-seed">Seed</label>
                <input id="sg-seed" type="number" value={seed}
                    onChange={(e) => setSeed(Math.trunc(Number(e.target.value) || 0))} data-testid="sg-seed" />
            </div>
            <div className="playbook-picker__count">
                <label htmlFor="sg-extra">Extra instructions</label>
                <textarea id="sg-extra" maxLength={8000} value={extra} onChange={(e) => setExtra(e.target.value)} data-testid="sg-extra" />
            </div>
        </div>
    );
}
