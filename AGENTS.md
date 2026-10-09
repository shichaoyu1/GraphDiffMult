# PASA research workspace

## Context and communication

- Respond in Chinese unless requested otherwise. Start with the result and distinguish verified facts, inferences, and untested hypotheses.
- Read PROJECT_STATE.md when taking over research work. RESEARCH_WORKFLOW_PASA.md explains the research process; RESEARCH_PROTOCOL_TEMPLATE.md is for new experiments. PASA_EVALUATION_PROTOCOL.md is a proposed revision, not evidence that the revision was implemented.
- Follow the user's current authorization. Finish authorized local work; do not turn routine implementation choices into repeated approval requests.

## Scientific evidence

- Trace material claims and reported numbers to source code, data definitions, run configurations, and manuscript versions. Local intermediate manuscripts are not automatically the submitted paper.
- Patient metadata, imaging-derived proxy labels, region selection rules, and independently measured regional pathology are distinct evidence sources. Do not infer the last from the first three.
- Shared train/test label definitions are normal supervised learning. Establish leakage from actual information flow or overlapping cases, not merely a common label-construction function.
- Verify metric formulas. In the historical implementation recall@k means any-positive Hit@k, and anchor_consistency means a positive/negative cosine-similarity gap. Recheck after code changes.
- For ablations, compare the same patients, target definitions, candidate space, missing-label masks, input conditions, selection procedure, and budgets. Random untrained anchors are not automatically a meaningful control.
- Missing or unselected labels are not automatically true negatives. Report coverage, exclusions, fallbacks, and out-of-vocabulary labels.
- Keep exploration separate from frozen confirmation. Record protocol changes and distinguish split randomness, training randomness, and development rounds. Preserve negative results and adjust claims to evidence.

## Implementation and verification

- Define the expected behavior before scientific code changes. Use small synthetic cases for ranking, masks, splits, and edge cases before costly experiments.
- Keep training and evaluation protocol versions explicit. Save configuration, source version, data/split/label versions, checkpoint-selection rules, and patient-level outputs for consequential results.
- Do not silently reuse validation/test patients when a split is empty. Check actual run split records before claiming historical leakage.
- Preserve user changes and old evidence. A correction creates a traceable new result; do not overwrite inconvenient historical numbers.
- Do not invent expert validation, labels, completed experiments, significance tests, or biological interpretations.

## Model workflow

- Project default: GPT-6.1 Sol / low. GPT-6 Astra is the candidate for critical scientific review; use explicit model selection when requested or authorized. This file does not automatically switch models or authorize unrequested agents.
- Model agreement is not independent scientific evidence. Use source checks and executable validation; raise reasoning effort only when a task's observed failures justify another trial.
- Migration trial artifacts and limitations are in migration/gpt6_20261009/RESULTS.md. Configuration defaults do not prove the active model of an existing chat.
