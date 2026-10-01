/**
 * Advanced & PEFT column (Power): LoRA, precision, OOM retry, alignment
 * dataset controls and observability telemetry.
 * Reads/writes the shared config form; TrainingPanel owns loading + payload.
 */

import type { TrainingConfigForm } from './useTrainingConfigForm';

interface TrainingAdvancedPeftSectionProps {
  form: TrainingConfigForm;
  isAlignmentMode: boolean;
}

export default function TrainingAdvancedPeftSection({
  form,
  isAlignmentMode,
}: TrainingAdvancedPeftSectionProps) {
  const {
    sequencePacking,
    setSequencePacking,
    useLora,
    setUseLora,
    curriculum,
    setCurriculum,
    loraR,
    setLoraR,
    loraAlpha,
    setLoraAlpha,
    targetModules,
    setTargetModules,
    fp16,
    setFp16,
    bf16,
    setBf16,
    flashAttention,
    setFlashAttention,
    autoOomRetry,
    setAutoOomRetry,
    maxOomRetries,
    setMaxOomRetries,
    oomRetrySeqShrink,
    setOomRetrySeqShrink,
    gradientCheckpointing,
    setGradientCheckpointing,
    multimodalRequireMedia,
    setMultimodalRequireMedia,
    alignmentAutoFilter,
    setAlignmentAutoFilter,
    alignmentQualityThreshold,
    setAlignmentQualityThreshold,
    alignmentBeta,
    setAlignmentBeta,
    alignmentMaxPromptLength,
    setAlignmentMaxPromptLength,
    alignmentMaxLength,
    setAlignmentMaxLength,
    alignmentMinKeepRatio,
    setAlignmentMinKeepRatio,
    alignmentDatasetPath,
    setAlignmentDatasetPath,
    alignmentIncludePlaygroundFeedback,
    setAlignmentIncludePlaygroundFeedback,
    alignmentPlaygroundMaxPairs,
    setAlignmentPlaygroundMaxPairs,
    observabilityEnabled,
    setObservabilityEnabled,
    observabilityLogSteps,
    setObservabilityLogSteps,
    observabilityMaxLayers,
    setObservabilityMaxLayers,
    observabilityProbeAttention,
    setObservabilityProbeAttention,
    observabilityProbeTopK,
    setObservabilityProbeTopK,
    setTouchedConfig,
  } = form;

  return (
        <div>
          <h4 className="training-config-section-title">Advanced & PEFT</h4>
          <div
            className="form-group training-toggle-row"
            data-testid="training-config-curriculum-row"
          >
            <input
              type="checkbox"
              checked={curriculum}
              onChange={(e) => {
                setCurriculum(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, curriculum: true }));
              }}
              data-testid="training-config-curriculum-toggle"
            />
            <label className="form-label form-label-inline-tight">
              Curriculum learning
              <span style={{ marginLeft: 8, color: 'var(--text-tertiary)', fontWeight: 400, fontSize: '0.85em' }}>
                (recommended for thin classification: easy rows first, then harder ones; auto-on by default for classification projects with ≤ 200 train rows)
              </span>
            </label>
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={useLora}
              onChange={(e) => {
                setUseLora(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, use_lora: true }));
              }}
            />
            <label className="form-label form-label-inline-tight">Enable LoRA</label>
          </div>
          {useLora && (
            <div className="training-lora-box">
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Rank (r)</label>
                  <input
                    className="input"
                    type="number"
                    value={loraR}
                    onChange={(e) => {
                      setLoraR(Number(e.target.value) || 1);
                      setTouchedConfig((prev) => ({ ...prev, lora_r: true }));
                    }}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Alpha</label>
                  <input
                    className="input"
                    type="number"
                    value={loraAlpha}
                    onChange={(e) => {
                      setLoraAlpha(Number(e.target.value) || 1);
                      setTouchedConfig((prev) => ({ ...prev, lora_alpha: true }));
                    }}
                  />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Target Modules (comma-separated)</label>
                <input
                  className="input"
                  placeholder="auto — every linear layer on small models (up to 2B), q_proj, v_proj on larger"
                  value={targetModules}
                  onChange={(e) => {
                    setTargetModules(e.target.value);
                    setTouchedConfig((prev) => ({ ...prev, target_modules: true }));
                  }}
                />
              </div>
            </div>
          )}
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={gradientCheckpointing}
              onChange={(e) => {
                setGradientCheckpointing(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, gradient_checkpointing: true }));
              }}
            />
            <label className="form-label form-label-inline-tight">Use Gradient Checkpointing</label>
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={sequencePacking}
              onChange={(e) => {
                setSequencePacking(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, sequence_packing: true }));
              }}
            />
            <label className="form-label form-label-inline-tight">Enable Sequence Packing</label>
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              aria-label="Require local media assets for multimodal batches"
              checked={multimodalRequireMedia}
              onChange={(e) => {
                setMultimodalRequireMedia(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, multimodal_require_media: true }));
              }}
            />
            <label className="form-label form-label-inline-tight">Require Local Media Assets (Strict Multimodal)</label>
          </div>
          <div className="form-hint">
            Blocks text-fallback for image/audio rows and fails on missing/remote media refs.
            Preflight Plan can auto-relax this flag when strict mode would block launch.
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={flashAttention}
              onChange={(e) => {
                setFlashAttention(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, flash_attention: true }));
              }}
            />
            <label className="form-label form-label-inline-tight">Enable Flash Attention</label>
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={bf16}
              onChange={(e) => {
                const checked = e.target.checked;
                setBf16(checked);
                setTouchedConfig((prev) => ({ ...prev, bf16: true }));
                if (checked && fp16) {
                  setFp16(false);
                  setTouchedConfig((prev) => ({ ...prev, fp16: true }));
                }
              }}
            />
            <label className="form-label form-label-inline-tight">Use BF16</label>
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={fp16}
              onChange={(e) => {
                const checked = e.target.checked;
                setFp16(checked);
                setTouchedConfig((prev) => ({ ...prev, fp16: true }));
                if (checked && bf16) {
                  setBf16(false);
                  setTouchedConfig((prev) => ({ ...prev, bf16: true }));
                }
              }}
            />
            <label className="form-label form-label-inline-tight">Use FP16</label>
          </div>
          <div className="form-group training-toggle-row">
            <input
              type="checkbox"
              checked={autoOomRetry}
              onChange={(e) => {
                setAutoOomRetry(e.target.checked);
                setTouchedConfig((prev) => ({ ...prev, auto_oom_retry: true }));
              }}
            />
            <label className="form-label form-label-inline-tight">Auto OOM Retry Planner</label>
          </div>
          <div className="training-grid-2">
            <div className="form-group">
              <label className="form-label">Max OOM Retries</label>
              <input
                className="input"
                type="number"
                min={0}
                max={5}
                value={maxOomRetries}
                onChange={(e) => {
                  const v = Math.min(5, Math.max(0, Number(e.target.value) || 0));
                  setMaxOomRetries(v);
                  setTouchedConfig((prev) => ({ ...prev, max_oom_retries: true }));
                }}
              />
            </div>
            <div className="form-group">
              <label className="form-label">OOM Seq Shrink</label>
              <input
                className="input"
                value={oomRetrySeqShrink}
                onChange={(e) => {
                  setOomRetrySeqShrink(e.target.value);
                  setTouchedConfig((prev) => ({ ...prev, oom_retry_seq_shrink: true }));
                }}
                placeholder="0.75"
              />
            </div>
          </div>
          {isAlignmentMode && (
            <div className="training-lora-box">
              <h5 className="training-config-section-title" style={{ marginTop: 0 }}>
                Alignment Dataset Controls
              </h5>
              <div className="form-group training-toggle-row">
                <input
                  type="checkbox"
                  checked={alignmentAutoFilter}
                  onChange={(e) => {
                    setAlignmentAutoFilter(e.target.checked);
                    setTouchedConfig((prev) => ({ ...prev, alignment_auto_filter: true }));
                  }}
                />
                <label className="form-label form-label-inline-tight">Auto filter preference pairs before run</label>
              </div>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Alignment Beta</label>
                  <input
                    className="input"
                    value={alignmentBeta}
                    onChange={(e) => {
                      setAlignmentBeta(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, alignment_beta: true }));
                    }}
                    placeholder="0.1"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Alignment Quality Threshold</label>
                  <input
                    className="input"
                    value={alignmentQualityThreshold}
                    onChange={(e) => {
                      setAlignmentQualityThreshold(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, alignment_quality_threshold: true }));
                    }}
                    placeholder="3.0"
                  />
                </div>
              </div>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Alignment Max Prompt Length</label>
                  <input
                    className="input"
                    value={alignmentMaxPromptLength}
                    onChange={(e) => {
                      setAlignmentMaxPromptLength(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, alignment_max_prompt_length: true }));
                    }}
                    placeholder="1024"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Alignment Max Length</label>
                  <input
                    className="input"
                    value={alignmentMaxLength}
                    onChange={(e) => {
                      setAlignmentMaxLength(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, alignment_max_length: true }));
                    }}
                    placeholder="2048"
                  />
                </div>
              </div>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Alignment Min Keep Ratio</label>
                  <input
                    className="input"
                    value={alignmentMinKeepRatio}
                    onChange={(e) => {
                      setAlignmentMinKeepRatio(e.target.value);
                      setTouchedConfig((prev) => ({ ...prev, alignment_min_keep_ratio: true }));
                    }}
                    placeholder="0.4"
                  />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Alignment Dataset Path (Optional)</label>
                <input
                  className="input"
                  value={alignmentDatasetPath}
                  onChange={(e) => {
                    setAlignmentDatasetPath(e.target.value);
                    setTouchedConfig((prev) => ({ ...prev, alignment_dataset_path: true }));
                  }}
                  placeholder="prepared/alignment/train.filtered.jsonl"
                />
                <div className="form-hint">
                  Leave empty to use prepared train split. Path is project-relative under data/projects/&lt;id&gt;.
                </div>
              </div>
              <div className="form-group training-toggle-row">
                <input
                  type="checkbox"
                  checked={alignmentIncludePlaygroundFeedback}
                  onChange={(e) => {
                    setAlignmentIncludePlaygroundFeedback(e.target.checked);
                    setTouchedConfig((prev) => ({
                      ...prev,
                      alignment_include_playground_feedback: true,
                    }));
                  }}
                />
                <label className="form-label form-label-inline-tight">
                  Merge playground downvote pairs into alignment train dataset
                </label>
              </div>
              <div className="training-grid-2">
                <div className="form-group">
                  <label className="form-label">Playground Feedback Max Pairs</label>
                  <input
                    className="input"
                    value={alignmentPlaygroundMaxPairs}
                    onChange={(e) => {
                      setAlignmentPlaygroundMaxPairs(e.target.value);
                      setTouchedConfig((prev) => ({
                        ...prev,
                        alignment_playground_max_pairs: true,
                      }));
                    }}
                    placeholder="5000"
                  />
                </div>
              </div>
            </div>
          )}
          <div className="training-lora-box">
            <h5 className="training-config-section-title" style={{ marginTop: 0 }}>
              Observability Telemetry
            </h5>
            <div className="form-group training-toggle-row">
              <input
                type="checkbox"
                checked={observabilityEnabled}
                onChange={(e) => {
                  setObservabilityEnabled(e.target.checked);
                  setTouchedConfig((prev) => ({ ...prev, observability_enabled: true }));
                }}
              />
              <label className="form-label form-label-inline-tight">Enable gradient/attention telemetry emission</label>
            </div>
            <div className="form-group training-toggle-row">
              <input
                type="checkbox"
                checked={observabilityProbeAttention}
                onChange={(e) => {
                  setObservabilityProbeAttention(e.target.checked);
                  setTouchedConfig((prev) => ({ ...prev, observability_probe_attention: true }));
                }}
              />
              <label className="form-label form-label-inline-tight">Run attention probe on logging steps</label>
            </div>
            <div className="training-grid-2">
              <div className="form-group">
                <label className="form-label">Observability Log Steps</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  value={observabilityLogSteps}
                  onChange={(e) => {
                    setObservabilityLogSteps(Math.max(1, Number(e.target.value) || 1));
                    setTouchedConfig((prev) => ({ ...prev, observability_log_steps: true }));
                  }}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Max Gradient Layers</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  value={observabilityMaxLayers}
                  onChange={(e) => {
                    setObservabilityMaxLayers(Math.max(1, Number(e.target.value) || 1));
                    setTouchedConfig((prev) => ({ ...prev, observability_max_layers: true }));
                  }}
                />
              </div>
            </div>
            <div className="training-grid-2">
              <div className="form-group">
                <label className="form-label">Attention Top-K Tokens</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  value={observabilityProbeTopK}
                  onChange={(e) => {
                    setObservabilityProbeTopK(Math.max(1, Number(e.target.value) || 1));
                    setTouchedConfig((prev) => ({ ...prev, observability_probe_top_k: true }));
                  }}
                />
              </div>
            </div>
          </div>
        </div>
  );
}
