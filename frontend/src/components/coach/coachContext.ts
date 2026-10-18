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
    export: 'Package the finished model for download or deployment. You made it!',
};
