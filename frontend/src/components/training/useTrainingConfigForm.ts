/**
 * Form state for a training run's config (model, hyperparameters, PEFT,
 * alignment, observability, multi-seed) plus which fields the user touched.
 * TrainingPanel owns loading defaults into it and building the payload; the
 * config sections read and write it through the returned object.
 */

import { useState } from 'react';

import { type ConfigFieldKey, untouchedConfig } from './trainingPanelUtils';

export function useTrainingConfigForm() {
  const [baseModel, setBaseModel] = useState('HuggingFaceTB/SmolLM2-135M-Instruct');
  const [trainingMode, setTrainingMode] = useState('sft');
  const [trainingRuntimeId, setTrainingRuntimeId] = useState('auto');
  const [taskType, setTaskType] = useState('causal_lm');
  const [trainerBackend, setTrainerBackend] = useState('auto');
  const [chatTemplate, setChatTemplate] = useState('llama3');
  const [lr, setLr] = useState('2e-4');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(4);
  const [maxSeqLength, setMaxSeqLength] = useState(2048);
  const [optimizer, setOptimizer] = useState('paged_adamw_8bit');
  const [saveSteps, setSaveSteps] = useState(100);
  const [evalSteps, setEvalSteps] = useState(100);
  const [sequencePacking, setSequencePacking] = useState(true);
  const [useLora, setUseLora] = useState(true);
  // Phase 6d — curriculum learning toggle. Default starts off; the
  // backend (training_service._decide_curriculum_default) auto-sets
  // it true at create_experiment time for thin classification
  // projects when the field is unset. We initialize the toggle from
  // the project's defaults effect like every other field; explicit
  // user touch (touchedConfig.curriculum) sends the value through
  // includeField so the user's choice always wins.
  const [curriculum, setCurriculum] = useState(false);
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [targetModules, setTargetModules] = useState('q_proj, v_proj');
  const [fp16, setFp16] = useState(false);
  const [bf16, setBf16] = useState(true);
  const [flashAttention, setFlashAttention] = useState(true);
  const [autoOomRetry, setAutoOomRetry] = useState(true);
  const [maxOomRetries, setMaxOomRetries] = useState(2);
  const [oomRetrySeqShrink, setOomRetrySeqShrink] = useState('0.75');
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);
  const [multimodalRequireMedia, setMultimodalRequireMedia] = useState(false);
  const [alignmentAutoFilter, setAlignmentAutoFilter] = useState(false);
  const [alignmentQualityThreshold, setAlignmentQualityThreshold] = useState('3.0');
  const [alignmentBeta, setAlignmentBeta] = useState('0.1');
  const [alignmentMaxPromptLength, setAlignmentMaxPromptLength] = useState('1024');
  const [alignmentMaxLength, setAlignmentMaxLength] = useState('2048');
  const [alignmentMinKeepRatio, setAlignmentMinKeepRatio] = useState('0.4');
  const [alignmentDatasetPath, setAlignmentDatasetPath] = useState('');
  const [alignmentIncludePlaygroundFeedback, setAlignmentIncludePlaygroundFeedback] = useState(true);
  const [alignmentPlaygroundMaxPairs, setAlignmentPlaygroundMaxPairs] = useState('5000');
  const [observabilityEnabled, setObservabilityEnabled] = useState(true);
  const [observabilityLogSteps, setObservabilityLogSteps] = useState(50);
  const [observabilityMaxLayers, setObservabilityMaxLayers] = useState(12);
  const [observabilityProbeAttention, setObservabilityProbeAttention] = useState(true);
  const [observabilityProbeTopK, setObservabilityProbeTopK] = useState(6);
  // Quality-Lift phase 7 slice 3 — multi-seed variance reporting.
  // ``seed`` is the base PRNG seed (also reused as the single-seed
  // value when ``numSeeds === 1``); ``numSeeds`` ≥ 2 fans the run out
  // into a seed-group whose EvalResults get rolled into one
  // mean±std aggregate that gates judge by mean−std (no vanity
  // metrics). ``seedsExplicit`` is a comma-separated override that
  // wins over the derived list when non-empty. ``parallelSeeds``
  // dispatches children concurrently — off by default since single-GPU
  // boxes just queue + risk OOM.
  const [seed, setSeed] = useState(42);
  const [numSeeds, setNumSeeds] = useState(1);
  const [seedsExplicit, setSeedsExplicit] = useState('');
  const [parallelSeeds, setParallelSeeds] = useState(false);
  // Section starts collapsed because the default (num_seeds=1) is the
  // single-run UX every existing user already knows. The coach nudge
  // and the URL query ``?expand_multi_seed=1`` both flip this true so
  // a deep-linked user lands with the section pre-opened.
  const [multiSeedExpanded, setMultiSeedExpanded] = useState(false);
  const [useProfileDefaults, setUseProfileDefaults] = useState(true);
  const [touchedConfig, setTouchedConfig] = useState<Record<ConfigFieldKey, boolean>>(untouchedConfig());

  return {
    baseModel,
    setBaseModel,
    trainingMode,
    setTrainingMode,
    trainingRuntimeId,
    setTrainingRuntimeId,
    taskType,
    setTaskType,
    trainerBackend,
    setTrainerBackend,
    chatTemplate,
    setChatTemplate,
    lr,
    setLr,
    epochs,
    setEpochs,
    batchSize,
    setBatchSize,
    gradientAccumulationSteps,
    setGradientAccumulationSteps,
    maxSeqLength,
    setMaxSeqLength,
    optimizer,
    setOptimizer,
    saveSteps,
    setSaveSteps,
    evalSteps,
    setEvalSteps,
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
    useProfileDefaults,
    setUseProfileDefaults,
    touchedConfig,
    setTouchedConfig,
  };
}

export type TrainingConfigForm = ReturnType<typeof useTrainingConfigForm>;
