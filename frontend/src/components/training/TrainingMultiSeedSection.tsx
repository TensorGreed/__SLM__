/**
 * Multi-seed variance reporting (Power): base seed, seed count, explicit
 * seed list and parallel dispatch.
 * Reads/writes the shared config form; TrainingPanel owns loading + payload.
 */

import type {
} from './trainingPanelTypes';
import type { TrainingConfigForm } from './useTrainingConfigForm';

interface TrainingMultiSeedSectionProps {
  form: TrainingConfigForm;
}

export default function TrainingMultiSeedSection({
  form,
}: TrainingMultiSeedSectionProps) {
  const {
    seed,
    setSeed,
    numSeeds,
    setNumSeeds,
    seedsExplicit,
    setSeedsExplicit,
    parallelSeeds,
    setParallelSeeds,
    multiSeedExpanded,
    setMultiSeedExpanded,
    setTouchedConfig,
  } = form;

  return (
        <div
          id="multi-seed"
          className="training-multi-seed"
          data-testid="training-multi-seed-section"
        >
          <h4
            className="training-config-section-title"
            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
          >
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setMultiSeedExpanded((v) => !v)}
              aria-label={multiSeedExpanded ? 'Collapse multi-seed section' : 'Expand multi-seed section'}
              data-testid="training-multi-seed-toggle"
            >
              {multiSeedExpanded ? '▼' : '▶'}
            </button>
            Multi-seed variance (Quality-Lift phase 1)
            {numSeeds > 1 && (
              <span
                className="badge badge-info"
                data-testid="training-multi-seed-active-badge"
                style={{ marginLeft: 8 }}
              >
                {numSeeds} seeds
              </span>
            )}
          </h4>
          {multiSeedExpanded && (
            <div id="multi-seed-body" data-testid="training-multi-seed-body">
              <p className="form-hint" style={{ marginTop: 0 }}>
                Run N independent trainings with different
                seeds, then judge pass/fail rules by mean − std (no
                vanity metrics). Default 1 keeps single-run
                behavior.
              </p>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Base seed</label>
                  <input
                    className="input"
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(e) => {
                      setSeed(Math.max(0, Math.trunc(Number(e.target.value) || 0)));
                      setTouchedConfig((prev) => ({ ...prev, seed: true }));
                    }}
                    data-testid="training-multi-seed-base"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Number of seeds (1–8)</label>
                  <input
                    className="input"
                    type="number"
                    min={1}
                    max={8}
                    value={numSeeds}
                    onChange={(e) => {
                      const v = Math.max(1, Math.min(8, Math.trunc(Number(e.target.value) || 1)));
                      setNumSeeds(v);
                      setTouchedConfig((prev) => ({ ...prev, num_seeds: true }));
                    }}
                    data-testid="training-multi-seed-count"
                  />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">
                  Explicit seeds (comma-separated, overrides count)
                </label>
                <input
                  className="input"
                  value={seedsExplicit}
                  onChange={(e) => {
                    setSeedsExplicit(e.target.value);
                    setTouchedConfig((prev) => ({ ...prev, seeds: true }));
                  }}
                  placeholder="42, 1337, 7"
                  data-testid="training-multi-seed-explicit"
                />
              </div>
              <div className="form-group training-toggle-row">
                <input
                  type="checkbox"
                  checked={parallelSeeds}
                  onChange={(e) => {
                    setParallelSeeds(e.target.checked);
                    setTouchedConfig((prev) => ({ ...prev, parallel_seeds: true }));
                  }}
                  data-testid="training-multi-seed-parallel"
                />
                <label className="form-label form-label-inline-tight">
                  Run children in parallel (multi-GPU only)
                </label>
              </div>
              {(numSeeds > 1
                || seedsExplicit
                  .split(',')
                  .map((s) => s.trim())
                  .filter(Boolean).length > 1) && (
                <div
                  className="callout callout-info"
                  data-testid="training-multi-seed-variance-preview"
                  style={{ marginTop: 8 }}
                >
                  Will run{' '}
                  {seedsExplicit
                    .split(',')
                    .map((s) => s.trim())
                    .filter(Boolean).length > 1
                    ? seedsExplicit
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean).length
                    : numSeeds}{' '}
                  independent trainings under one{' '}
                  <code>seed_group_id</code>; pass/fail rules will
                  judge the run by <code>mean − std</code>{' '}
                  (the lower bound) so a vanity-good seed
                  can't paper over a flaky run.
                </div>
              )}
            </div>
          )}
        </div>
  );
}
