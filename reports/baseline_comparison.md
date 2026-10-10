# Dermalens AI — Baseline Experiment Comparison

## Overview

This report summarizes five exploratory experiments for skin-lesion image classification using the MILK10k training dataset.

The experiments investigate image-only classification, training augmentation and class balancing, structured metadata fusion, a frozen-backbone approach, and label smoothing.

**Research status:** Experimental research only. These results do not establish clinical validity and must not be used to diagnose or treat skin conditions.

## Dataset and evaluation

* 5,240 lesions and 10,480 images in the local dataset.
* 11 diagnostic categories.
* 532 lesions in the test split.
* Lesion-level separation between training, validation, and test splits.
* Patient-level independence cannot be verified because explicit patient identifiers are unavailable in the supplied metadata.

The test results below have already been used to compare experiments. They should therefore be treated as exploratory results rather than an untouched final evaluation. Future model selection should prioritize validation data, followed by evaluation on a genuinely independent test set.

## Aggregate results

| Baseline | Approach                               | Accuracy | Balanced accuracy | Macro-F1 |
| -------- | -------------------------------------- | -------: | ----------------: | -------: |
| A        | Image-only ResNet-50                   |   62.59% |            31.86% |   33.16% |
| B        | Image augmentation and class balancing |   61.84% |            42.49% |   42.86% |
| C        | Image plus structured metadata         |   59.21% |            43.45% |   38.98% |
| D        | Frozen-backbone approach               |   59.40% |            35.83% |   35.24% |
| E        | Label smoothing                        |   60.53% |            43.54% |   44.69% |

*Values are copied from the existing experiment reports and have not been independently recalculated here.*

## Observations

### Baseline B: augmentation and class balancing

Compared with Baseline A, Baseline B improved test Macro-F1 from 33.16% to 42.86% and balanced accuracy from 31.86% to 42.49%, while overall accuracy decreased slightly.

### Baseline C: structured metadata

Baseline C achieved 43.45% balanced accuracy but a lower Macro-F1 of 38.98%. Adding metadata did not improve Macro-F1 over Baseline B in this experiment.

### Baseline D: frozen backbone

Baseline D achieved a test Macro-F1 of 35.24%, below the results reported for Baselines B and E. The result suggests that this particular configuration did not outperform the stronger baselines on the current test split.

### Baseline E: label smoothing

Baseline E achieved the highest reported Macro-F1 among these five experiments at 44.69%, with 60.53% accuracy and 43.54% balanced accuracy. This is a preliminary result and requires further validation.

## Limitations

* The dataset is class-imbalanced.
* Some categories have very few test examples; for example, MAL_OTH has only two.
* Patient-level independence cannot be confirmed from the available metadata.
* Performance on this dataset may not generalize to different clinical settings, devices, or skin tones.
* Test-set comparisons across multiple experiments can lead to optimistic model-selection estimates.
* Aggregate metrics alone do not establish clinical safety, diagnostic accuracy, or readiness for public use.

## Next steps

1. Preserve the current results as exploratory research records.
2. Use validation data for further model selection and tuning.
3. Seek an appropriately licensed, independent dataset for external evaluation.
4. Evaluate performance across classes, skin tones, and relevant image sources.
5. Investigate calibration and uncertainty estimation.
6. Keep any future user-facing prototype educational, privacy-conscious, and clearly non-diagnostic until appropriate clinical and regulatory evaluation has been completed.

## Current status

Five experimental baselines have been evaluated. Baseline E has the highest reported test Macro-F1 among them, but no model is currently established as suitable for clinical diagnosis.
