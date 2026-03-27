---
sidebar_position: 7
title: Evaluation and Remediation
---

# Evaluation and Remediation

Evaluation is not just a score. It is your decision engine for what to improve next.

## What to Measure

At minimum:

- task quality (accuracy/F1/pass rate)
- safety behavior
- hallucination tendency
- regression vs previous model

## Understand Failure Clusters

Group failures by pattern:

- coverage gap (missing examples)
- formatting mismatch
- unsupported reasoning pattern
- unsafe or policy-violating output

## Closed-Loop Remediation

Use remediation plans to convert failures into actions:

- **Data operations**: collect, augment, relabel, or filter.
- **Training changes**: adjust task profile, epochs, LR, or model family.
- **Expected impact**: apply high-confidence fixes first.

## Fast Improvement Cycle

1. Run eval
2. Pick top 1 to 2 failure clusters
3. Apply focused remediation
4. Retrain quickly
5. Re-evaluate

Repeat until improvements flatten.

## Promotion Gate Mindset

Use gate policies as deployment safety rails, not as bureaucracy.

A good gate policy answers:

- what must pass,
- what can degrade slightly,
- what is always unacceptable.
