/**
 * The one-line tip per step (``StepHelp`` shows it with the walkthrough
 * video; the Coach bar that used to carry it is gone — Epic U phase U6).
 */

import type { TabKey } from '../../types';

export const TAB_TIP: Record<TabKey, string> = {
    data: 'This is where your raw examples live. Import a CSV or launch a sample dataset to get started.',
    cleaning: 'Tidy the data — drop broken rows, dedupe, normalize. One-click autofixes handle the common cases.',
    goldset: 'Build a small, trusted answer key. Evaluation scores your model against it, so the quality here matters most.',
    synthetic: 'Short on data? Generate more examples from a playbook that matches your task shape.',
    dataprep: 'Split your data into train / validation / test sets the trainer can consume.',
    tokenization: 'See how your text becomes tokens — catch truncation and out-of-vocabulary surprises before you train (Train → More options → Tokenization).',
    training: 'Check the base model and task, pick one run or three seeds, and press Train. Everything else waits under More options; runs and live metrics stay below.',
    eval: 'Score the trained model against your answer key and the built-in checks — honest numbers, pass/fail rules that can fail.',
    compression: 'Shrink the model (quantize / distill) so it fits your deployment target.',
    export: 'Package the finished model for download or deployment.',
};

/**
 * Tips for the sub-steps of a destination that keeps one step key: the
 * Results tabs (``eval``) and the Ship stages (``export``). ``StepHelp``
 * prefers these over ``TAB_TIP`` when the page passes its ``sub``.
 */
export const SUB_STEP_TIP: Partial<Record<TabKey, Record<string, string>>> = {
    eval: {
        verdict: 'Score the trained model against your answer key and the built-in checks — honest numbers, pass/fail rules that can fail.',
        failures: 'See where the model goes wrong — failing rows, clusters, slices and the built-in checks — and what to fix next.',
        compare: 'Put runs side by side: external benchmarks, frontier models, retrieval against the fine-tune, student against teacher.',
        try: 'Chat with a trained run (or the base model with retrieval) to see how it answers before you ship it.',
    },
    export: {
        package: 'Package the finished model for download or deployment — pick the format, the best seed and any compression.',
        publish: 'Publish a completed Hugging Face export to the Hub with its model card and release notes.',
        deploy: 'Serve the packaged model or plan where it runs.',
        monitor: 'Watch the model after it ships — traces, latency and failures from real traffic.',
    },
};
