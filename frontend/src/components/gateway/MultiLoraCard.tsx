/**
 * MultiLoraCard — Ship > Deploy: serve several LoRA runs that share a base
 * model from one vLLM server, then point the gateway at it.
 */

import { useEffect, useState } from 'react';

import './gateway.css';
import {
    buildMultiLoraPlan, getGatewaySettings, parseGatewayError, putGatewaySettings, startMultiLoraRun,
    type MultiLoraPlan, type ServableRun,
} from '../../api/gateway';
import { toast } from '../../stores/toastStore';
import { copyText, runLabel } from './GatewayCard';

export default function MultiLoraCard({ projectId }: { projectId: number }) {
    const [runs, setRuns] = useState<ServableRun[]>([]);
    const [picked, setPicked] = useState<number[]>([]);
    const [plan, setPlan] = useState<MultiLoraPlan | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [status, setStatus] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        getGatewaySettings(projectId)
            .then((d) => { if (!cancelled) setRuns(d.servable_runs.filter((r) => r.adapter)); })
            .catch(() => { if (!cancelled) setRuns([]); });
        return () => { cancelled = true; };
    }, [projectId]);

    const toggle = (id: number) => {
        setPlan(null);
        setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
    };

    const build = async () => {
        setError(null);
        setStatus(null);
        try {
            setPlan(await buildMultiLoraPlan(projectId, { experiment_ids: picked }));
        } catch (err) {
            setPlan(null);
            setError(parseGatewayError(err).message);
        }
    };
    const start = async () => {
        setError(null);
        try {
            const res = await startMultiLoraRun(projectId, { experiment_ids: picked });
            setStatus(String(res.status ?? res.message ?? 'started'));
        } catch (err) {
            setError(parseGatewayError(err).message);
        }
    };
    const route = async () => {
        if (!plan) return;
        setError(null);
        try {
            await putGatewaySettings(projectId, plan.gateway_settings_hint);
            toast.success('Gateway now routes to this server');
        } catch (err) {
            setError(parseGatewayError(err).message);
        }
    };

    return (
        <section className="card" data-testid="multi-lora-card">
            <h3 className="u-card-title u-mb-sm">Serve several runs on one base (vLLM multi-LoRA)</h3>
            <p className="u-hint u-mt0 u-mb-md">
                Pick LoRA runs trained on the same base model. One server holds the base once and swaps adapters per request.
            </p>
            {runs.length === 0 ? (
                <p className="u-hint" data-testid="multi-lora-empty">No completed LoRA adapter runs in this project yet.</p>
            ) : (
                <ul className="gateway__list">
                    {runs.map((r) => (
                        <li key={r.id}>
                            <label>
                                <input type="checkbox" checked={picked.includes(r.id)} onChange={() => toggle(r.id)}
                                    data-testid={`multi-lora-run-${r.id}`} />
                                {' '}{runLabel(r)}
                            </label>
                        </li>
                    ))}
                </ul>
            )}
            <div className="gateway__row">
                <button type="button" className="btn btn-primary" disabled={picked.length === 0} data-testid="multi-lora-build"
                    onClick={() => void build()}>Build plan</button>
            </div>
            {error && <p className="gateway__error" role="alert" data-testid="multi-lora-error">{error}</p>}
            {plan && (
                <div data-testid="multi-lora-plan">
                    <p>Base model: <code>{plan.base_model}</code></p>
                    <table className="gateway__table">
                        <thead><tr><th>Run</th><th>Name</th><th>Module name</th><th>Rank</th><th>Adapter path</th></tr></thead>
                        <tbody>
                            {plan.adapters.map((a) => (
                                <tr key={a.run_id}>
                                    <td>#{a.run_id}</td><td>{a.name}</td><td><code>{a.module_name}</code></td>
                                    <td>{a.rank ?? '-'}</td><td><code>{a.path}</code></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {plan.template.command && (
                        <>
                            <div className="gateway__row">
                                <strong>Command</strong>
                                <button type="button" className="btn btn-sm btn-secondary" data-testid="multi-lora-copy"
                                    onClick={() => void copyText(String(plan.template.command), 'the command')}>Copy</button>
                            </div>
                            <pre className="gateway__code" data-testid="multi-lora-command">{plan.template.command}</pre>
                        </>
                    )}
                    <div className="gateway__row">
                        <button type="button" className="btn btn-secondary" data-testid="multi-lora-start" onClick={() => void start()}>Start</button>
                        <button type="button" className="btn btn-secondary" data-testid="multi-lora-route" onClick={() => void route()}>
                            Route the gateway to this server
                        </button>
                    </div>
                    {status && <p data-testid="multi-lora-status">Server status: {status}</p>}
                </div>
            )}
        </section>
    );
}
