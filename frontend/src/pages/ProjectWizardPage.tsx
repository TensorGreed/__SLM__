import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import api from '../api/client';
import EDADashboard from '../components/data/EDADashboard';
import type { ProjectWorkspaceContextValue } from './ProjectWorkspaceContext';
import './ProjectWizardPage.css';

interface BenchmarkRow {
    rank: number;
    model_id: string;
    estimated_accuracy_percent: number;
    estimated_latency_ms: number;
}

export default function ProjectWizardPage() {
    const { projectId } = useOutletContext<ProjectWorkspaceContextValue>();
    const navigate = useNavigate();

    const [currentStep, setCurrentStep] = useState(1);

    const [uploading, setUploading] = useState(false);
    const [docsCount, setDocsCount] = useState(0);
    const [runName, setRunName] = useState('Discovery-Triggered Run');

    const [hardware, setHardware] = useState('rtx-3090');
    const [trainingStrategy, setTrainingStrategy] = useState<'classification' | 'classification_entitlement'>('classification');
    const [benchmarkLoading, setBenchmarkLoading] = useState(false);
    const [benchmarkRows, setBenchmarkRows] = useState<BenchmarkRow[]>([]);
    const [benchmarkError, setBenchmarkError] = useState('');
    const [extraFilters, setExtraFilters] = useState<string[]>([]);

    const [trainingProgress, setTrainingProgress] = useState(0);

    useEffect(() => {
        api.get<any[]>(`/projects/${projectId}/ingestion/documents`)
            .then((res) => {
                const count = Array.isArray(res.data) ? res.data.length : 0;
                setDocsCount(count);
                if (count > 0 && currentStep === 1) {
                    setCurrentStep(2);
                }
            })
            .catch(() => { });
    }, [projectId, currentStep]);

    const stepItems = useMemo(() => ([
        { num: 1, label: 'Upload Data' },
        { num: 2, label: 'Review Data' },
        { num: 3, label: 'Select Hardware' },
        { num: 4, label: 'Auto-Train' },
        { num: 5, label: 'Chat with Model' },
    ]), []);

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.length) return;
        setUploading(true);
        const formData = new FormData();
        Array.from(e.target.files).forEach((f) => formData.append('files', f));
        try {
            await api.post(`/projects/${projectId}/ingestion/upload-batch`, formData);
            setDocsCount((prev) => prev + e.target.files!.length);
            setCurrentStep(2);
        } catch {
            alert('Upload failed');
        } finally {
            setUploading(false);
        }
    };

    const hardwareToTargetDevice = (value: string): 'mobile' | 'laptop' | 'server' => {
        if (value === 'server') return 'server';
        if (value === 'macbook') return 'mobile';
        return 'laptop';
    };

    const hardwareToVram = (value: string): number => {
        if (value === 'server') return 80;
        if (value === 'rtx-3090') return 24;
        return 8;
    };

    const runQuickBenchmark = async () => {
        setBenchmarkLoading(true);
        setBenchmarkError('');
        try {
            const res = await api.post(`/projects/${projectId}/training/model-selection/benchmark-sweep`, {
                target_device: hardwareToTargetDevice(hardware),
                primary_language: 'english',
                available_vram_gb: hardwareToVram(hardware),
                task_profile: trainingStrategy === 'classification_entitlement' ? 'classification' : undefined,
                max_models: 3,
            });
            setBenchmarkRows(Array.isArray(res.data?.matrix) ? res.data.matrix : []);
        } catch (err: any) {
            setBenchmarkRows([]);
            setBenchmarkError(err.response?.data?.detail || err.message || 'Benchmark failed');
        } finally {
            setBenchmarkLoading(false);
        }
    };

    const handleStartTraining = async () => {
        setCurrentStep(4);
        try {
            await api.post(`/projects/${projectId}/pipeline/advance`);
            await api.post(`/projects/${projectId}/pipeline/advance`);
            await api.post(`/projects/${projectId}/pipeline/advance`);
            for (let i = 0; i <= 100; i += 10) {
                setTrainingProgress(i);
                await new Promise((r) => setTimeout(r, 800));
            }
            setCurrentStep(5);
        } catch {
            alert('Training failed to start');
        }
    };

    const addFilter = () => {
        const token = `Filter ${extraFilters.length + 1}`;
        setExtraFilters((prev) => [...prev, token]);
    };

    return (
        <div className="wizard-page animate-fade-in">
            <div className="wizard-shell card">
                <div className="wizard-header">
                    <div>
                        <h2>Guided Setup</h2>
                        <p>Configure data discovery and model training with a simplified flow.</p>
                    </div>
                    <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/pipeline`)}>
                        Advanced Mode
                    </button>
                </div>

                <div className="wizard-stepper">
                    {stepItems.map((item) => (
                        <div key={item.num} className={`wizard-step-indicator ${currentStep >= item.num ? 'active' : ''}`}>
                            <div className={`step-circle ${currentStep > item.num ? 'complete' : ''}`}>
                                {currentStep > item.num ? '✓' : item.num}
                            </div>
                            <div className="step-label">{item.label}</div>
                        </div>
                    ))}
                </div>

                {currentStep === 1 && (
                    <section className="wizard-section">
                        <h3>General Info</h3>
                        <label className="form-label">Run name</label>
                        <input
                            className="input"
                            value={runName}
                            onChange={(e) => setRunName(e.target.value)}
                            placeholder="Discovery-Triggered Run"
                        />
                        <div className="wizard-upload-box">
                            <input type="file" multiple id="wizard-upload" onChange={handleUpload} />
                            <label htmlFor="wizard-upload" className="btn btn-primary">
                                {uploading ? 'Uploading...' : 'Select Files'}
                            </label>
                            <span>PDF, TXT, CSV and JSON supported.</span>
                        </div>
                    </section>
                )}

                {currentStep === 2 && (
                    <section className="wizard-section">
                        <h3>Review Data</h3>
                        <p className="wizard-muted">Detected {docsCount} documents. Review quality before continuing.</p>
                        <div className="wizard-panel">
                            <EDADashboard projectId={projectId} />
                        </div>
                        <div className="wizard-actions">
                            <button className="btn btn-secondary" onClick={() => setCurrentStep(1)}>Back</button>
                            <button className="btn btn-primary" onClick={() => setCurrentStep(3)}>Continue</button>
                        </div>
                    </section>
                )}

                {currentStep === 3 && (
                    <section className="wizard-section">
                        <h3>Select Scan Type</h3>
                        <div className="wizard-option-grid">
                            <button
                                className={`wizard-option-card ${trainingStrategy === 'classification' ? 'selected' : ''}`}
                                onClick={() => setTrainingStrategy('classification')}
                            >
                                <div className="wizard-option-title">Classification</div>
                                <div className="wizard-option-tags">
                                    <span className="option-tag">Classification</span>
                                </div>
                                <p>Scans your environment to locate and classify sensitive data for compliance flows.</p>
                            </button>
                            <button
                                className={`wizard-option-card ${trainingStrategy === 'classification_entitlement' ? 'selected' : ''}`}
                                onClick={() => setTrainingStrategy('classification_entitlement')}
                            >
                                <div className="wizard-option-title">Classification and Entitlement</div>
                                <div className="wizard-option-tags">
                                    <span className="option-tag">Classification</span>
                                    <span className="option-tag">Entitlement</span>
                                </div>
                                <p>Combines classification with access-oriented checks for entitlement analysis.</p>
                            </button>
                        </div>

                        <h3 className="wizard-subhead">Classification Search Parameters</h3>
                        <p className="wizard-muted">Define profiles and filters that training should focus on.</p>
                        <div className="wizard-param-row">
                            <label className="form-label">Target hardware</label>
                            <select
                                className="input"
                                value={hardware}
                                onChange={(e) => setHardware(e.target.value)}
                            >
                                <option value="macbook">MacBook / Laptop</option>
                                <option value="rtx-3090">RTX 3090 / 4090</option>
                                <option value="server">A100 / Server</option>
                            </select>
                        </div>

                        <div className="wizard-param-row">
                            <button className="btn btn-secondary" onClick={() => void runQuickBenchmark()} disabled={benchmarkLoading}>
                                {benchmarkLoading ? 'Benchmarking...' : 'Run 5-Minute Model Benchmark'}
                            </button>
                            {benchmarkError && <span className="wizard-error">{benchmarkError}</span>}
                        </div>

                        {benchmarkRows.length > 0 && (
                            <div className="wizard-table-wrap">
                                <table className="wizard-table">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Accuracy</th>
                                            <th>Latency</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {benchmarkRows.map((item) => (
                                            <tr key={item.model_id}>
                                                <td>{item.model_id}</td>
                                                <td>{Number(item.estimated_accuracy_percent || 0).toFixed(1)}%</td>
                                                <td>{Number(item.estimated_latency_ms || 0).toFixed(1)} ms</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        <div className="wizard-filters">
                            <button className="btn btn-ghost" onClick={addFilter}>+ Add filter</button>
                            {extraFilters.length > 0 && (
                                <div className="wizard-filter-list">
                                    {extraFilters.map((filter) => (
                                        <span key={filter} className="wizard-filter-chip">{filter}</span>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="wizard-actions wizard-actions-bottom">
                            <button className="btn btn-secondary" onClick={() => setCurrentStep(2)}>Cancel</button>
                            <button className="btn btn-primary" onClick={handleStartTraining}>Save</button>
                        </div>
                    </section>
                )}

                {currentStep === 4 && (
                    <section className="wizard-section">
                        <h3>Auto-Train</h3>
                        <p className="wizard-muted">Training is running with auto-selected parameters.</p>
                        <div className="wizard-progress">
                            <div className="wizard-progress-fill" style={{ width: `${trainingProgress}%` }} />
                        </div>
                        <div className="wizard-progress-label">{trainingProgress}% complete</div>
                    </section>
                )}

                {currentStep === 5 && (
                    <section className="wizard-section">
                        <h3>Training Complete</h3>
                        <p className="wizard-muted">Your model is ready for prompt testing and export.</p>
                        <div className="wizard-actions">
                            <button className="btn btn-secondary" onClick={() => navigate(`/project/${projectId}/pipeline/export`)}>
                                Export Model
                            </button>
                            <button className="btn btn-primary" onClick={() => navigate(`/project/${projectId}/playground`)}>
                                Open Playground
                            </button>
                        </div>
                    </section>
                )}
            </div>
        </div>
    );
}
