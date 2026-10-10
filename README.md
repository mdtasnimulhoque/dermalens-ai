# Dermalens AI

### Explainable AI Research for Skin-Lesion Image Analysis

**Building transparent, reproducible, and responsible AI for skin-lesion image research.**

Dermalens AI is an open-source research project investigating how deep learning and explainable artificial intelligence (XAI) can support skin-lesion image analysis. The project explores image classification, model evaluation, dataset integrity, and the limitations that must be addressed before AI systems can be considered for clinical use.

The goal is not to replace dermatologists or provide automated diagnoses. It is to study how machine-learning models can be developed, evaluated, and interpreted more responsibly.

<p align="center">
  <strong>Deep Learning · Computer Vision · Medical Imaging · Explainable AI · Responsible AI</strong>
</p>

<p align="center">
  <a href="https://github.com/mdtasnimulhoque/dermalens-ai/actions">Automated Checks</a> ·
  <a href="https://github.com/mdtasnimulhoque/dermalens-ai/issues">Issues</a> ·
  <a href="https://github.com/mdtasnimulhoque/dermalens-ai/blob/main/LICENSE">License</a>
</p>

---

## Table of Contents

* [Overview](#overview)
* [Research Status](#research-status)
* [Dataset](#dataset)
* [Technical Approach](#technical-approach)
* [Explainability and Future Work](#explainability-and-future-work)
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)
* [Responsible Use and Limitations](#responsible-use-and-limitations)
* [Contributing](#contributing)
* [Research Integrity](#research-integrity)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Overview

Skin-lesion image classification is a challenging machine-learning problem. Models must learn useful visual patterns across diverse categories while accounting for class imbalance, limited examples of uncommon categories, uncertainty, and potential biases in the data.

Dermalens AI approaches this as a research problem rather than treating overall accuracy as the sole measure of success.

The repository contains dataset-validation and preparation scripts, multiple ResNet-50 baseline experiments, model-training code, and evaluation utilities. Calibration and further explainability, uncertainty, and safety work are areas for continued development.

### Research objectives

* **Develop image-classification baselines:** Establish and compare deep-learning approaches for skin-lesion image analysis.
* **Evaluate beyond accuracy:** Examine balanced accuracy, Macro-F1, precision, recall, and class-specific performance.
* **Reduce data leakage:** Keep images associated with the same lesion within a single dataset split.
* **Investigate explainability:** Explore methods such as Grad-CAM to visualize image regions associated with model predictions.
* **Study uncertainty:** Investigate how models can communicate uncertainty and recognize unreliable predictions.
* **Promote responsible research:** Document limitations, avoid unsupported medical claims, and emphasize the importance of professional medical evaluation.

---

## Research Status

### Current best experiment

**Baseline E — ResNet-50 with data augmentation, class balancing, and label smoothing.**

The experiment was evaluated on a held-out test split from MILK10k containing 1,064 images from 532 lesions.

| Metric            | Test result |
| ----------------- | ----------: |
| Accuracy          |      60.53% |
| Balanced accuracy |      43.54% |
| Macro-F1          |      44.69% |
| Macro precision   |      49.09% |
| Macro recall      |      43.54% |

These results describe the current experimental setup. They are not evidence of clinical effectiveness, diagnostic accuracy in real-world practice, or suitability for patient care.

### Understanding the metrics

* **Accuracy:** The proportion of evaluated images classified correctly.
* **Balanced accuracy:** The average recall across classes, giving each class equal weight.
* **Macro-F1:** The unweighted average of the F1 scores across classes.
* **Macro precision:** The unweighted average of class-level precision.
* **Macro recall:** The unweighted average of class-level recall.

The difference between accuracy and balanced accuracy illustrates why overall accuracy alone can hide poor performance on underrepresented categories.

Metrics for classes with very few test examples can be unstable. Results should therefore be interpreted alongside class support, confusion patterns, uncertainty, and independent validation.

### Baseline comparison

| Experiment                                    | Test accuracy | Balanced accuracy | Macro-F1 |
| --------------------------------------------- | ------------: | ----------------: | -------: |
| Baseline A — Plain ResNet-50                  |        62.59% |            31.86% |   33.16% |
| Baseline B — Augmentation and class balancing |        61.84% |            42.49% |   42.86% |
| Baseline C — Image and metadata               |        59.21% |            43.45% |   38.98% |
| Baseline D — Multimodal with regularization   |        59.40% |            35.83% |   35.24% |
| Baseline E — Label smoothing                  |        60.53% |            43.54% |   44.69% |

Among these experiments, Baseline E achieved the highest Macro-F1 and balanced accuracy. Baseline A achieved the highest overall accuracy.

These differences illustrate why model performance should be evaluated across multiple metrics rather than optimized around a single number.

**Important evaluation limitation:** The held-out test split has already been used to compare these experiments. It should not be treated as an untouched test set for future model selection. Further experimentation should use validation data, with genuinely independent data reserved for final evaluation when available.

---

## Dataset

Dermalens AI uses **MILK10k** for its current research experiments.

The dataset contains 10,480 images representing 5,240 lesions, with two imaging modalities per lesion: a clinical close-up image and a dermoscopic image. The reported classification baselines use the clinical close-up modality.

### Target categories

The project uses the following 11 target labels. Their precise meanings should be checked against the original dataset documentation.

| Label     | Category description                                   |
| --------- | ------------------------------------------------------ |
| `AKIEC`   | Actinic keratosis / intraepithelial carcinoma category |
| `BCC`     | Basal cell carcinoma                                   |
| `BEN_OTH` | Other benign lesions                                   |
| `BKL`     | Benign keratosis-like lesions                          |
| `DF`      | Dermatofibroma                                         |
| `INF`     | Inflammatory category                                  |
| `MAL_OTH` | Other malignant lesions                                |
| `MEL`     | Melanoma                                               |
| `NV`      | Melanocytic nevi                                       |
| `SCCKA`   | Squamous cell carcinoma / keratoacanthoma category     |
| `VASC`    | Vascular lesions                                       |

The labels are dataset categories, not diagnoses produced by this project.

### Data validation and splitting

The repository includes scripts for dataset auditing, validation, indexing, splitting, and loading. The reported experiment setup uses lesion-level splitting.

| Split      |   Lesions |     Images |
| ---------- | --------: | ---------: |
| Training   |     4,188 |      8,376 |
| Validation |       520 |      1,040 |
| Test       |       532 |      1,064 |
| **Total**  | **5,240** | **10,480** |

Keeping images from the same lesion in one split prevents the two imaging views of that lesion from being distributed across training and evaluation sets.

Lesion-level separation reduces one important source of data leakage. It does not establish patient-level independence or demonstrate generalization to other datasets, populations, imaging devices, or clinical environments.

### Dataset licensing and access

MILK10k has its own **CC BY-NC license terms**. Consult the original dataset source, documentation, and license before obtaining or using the data.

* Dataset images are not included in this repository.
* Do not commit downloaded archives, original images, or local dataset files to GitHub.
* Check the license and attribution requirements before redistributing or modifying dataset materials.
* Verify that your intended use is permitted, particularly for commercial applications.
* Cite and acknowledge the original dataset creators as required.

The license for this project's source code is separate from the license for MILK10k and any other third-party resources.

---

## Technical Approach

### Technology stack

| Component                  | Technology                                |
| -------------------------- | ----------------------------------------- |
| Programming language       | Python                                    |
| Deep-learning framework    | PyTorch                                   |
| Image models and utilities | Torchvision                               |
| Data handling              | pandas, Pillow, NumPy                     |
| Evaluation                 | scikit-learn and project-specific scripts |
| Configuration              | YAML                                      |
| Automated checks           | GitHub Actions                            |

The project has been developed in a local Apple Silicon environment. Actual performance and compatibility can vary with the operating system, Python version, dependency versions, and available hardware.

### Experimental progression

1. **Baseline A — Plain image classifier:** Establishes a ResNet-50 reference experiment.
2. **Baseline B — Augmentation and class balancing:** Investigates training-time augmentation and class weighting.
3. **Baseline C — Image plus metadata:** Explores combining image features with available metadata.
4. **Baseline D — Regularized multimodal model:** Investigates a modified multimodal training approach with frozen early layers.
5. **Baseline E — Label smoothing:** Investigates label smoothing alongside augmentation and class balancing.

The results do not improve monotonically across every metric. Comparisons should use consistent evaluation procedures, documented configurations, and appropriate data-splitting practices.

### Automated checks

The repository includes a GitHub Actions workflow that checks Python syntax in the `src/` directory when code is pushed to `main` or a pull request targets `main`.

A successful syntax check helps catch certain coding errors. It does not establish that the models train correctly, that experiments are reproducible, or that predictions are medically valid.

---

## Explainability and Future Work

A predicted category does not explain whether a model is reliable or what visual evidence influenced its output.

The following areas are research priorities, not claims that a finished clinical system currently implements these capabilities.

* **Grad-CAM visualizations:** Investigate image regions associated with model predictions.
* **Confidence calibration:** Evaluate whether confidence scores correspond to observed correctness.
* **Uncertainty estimation:** Study how to identify unreliable predictions.
* **Abstention:** Explore when a model should withhold a prediction rather than return an uncertain result.
* **Error analysis:** Examine confusion between categories and performance on underrepresented classes.
* **External validation:** Assess generalization using independent datasets, subject to licensing and appropriate research safeguards.
* **Reproducibility:** Document configurations, preprocessing, random seeds, data splits, and evaluation procedures.
* **Testing and software quality:** Add automated tests and strengthen validation of the data and training pipelines.

Visual explanations are not proof that a prediction is correct. Confidence scores do not guarantee correctness either.

---

## Repository Structure

The following tree reflects the committed repository layout. The `explainability/`, `inference/`, and `safety/` directories are currently placeholders containing package initializers.

```text
dermalens-ai/
├── .github/
│   └── workflows/
│       └── main.yml
├── configs/
│   └── training.yaml
├── reports/
│   └── baseline_comparison.md
├── src/
│   ├── data/
│   │   ├── audit_milk10k.py
│   │   ├── build_milk10k_index.py
│   │   ├── create_milk10k_split.py
│   │   ├── image_integrity_checker.py
│   │   ├── milk10k_dataloader.py
│   │   ├── milk10k_dataset.py
│   │   ├── milk10k_multimodal_dataset.py
│   │   ├── milk10k_validator.py
│   │   └── paths.py
│   ├── explainability/
│   │   └── __init__.py
│   ├── inference/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet50_baseline.py
│   │   ├── resnet50_multimodal.py
│   │   └── resnet50_multimodal_regularized.py
│   ├── safety/
│   │   └── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline.py
│   │   ├── train_baseline_b_augmented.py
│   │   ├── train_baseline_c_multimodal.py
│   │   ├── train_baseline_d_frozen.py
│   │   └── train_baseline_e_label_smoothing.py
│   ├── __init__.py
│   ├── analyze_baseline_errors.py
│   ├── evaluate_baseline_c.py
│   ├── evaluate_baseline_d.py
│   ├── evaluate_baseline_e.py
│   ├── evaluate_calibration.py
│   ├── evaluation_baseline.py
│   └── temperature_scaling.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

**Repository status:** The committed files include data-handling utilities, model definitions, training and evaluation scripts, a training configuration, a baseline comparison report, and an automated syntax-check workflow. Local datasets, generated indexes and splits, trained model checkpoints, and experiment outputs are not included.

The presence of a module or directory does not imply that the associated feature is fully implemented.

---

## Getting Started

### Prerequisites

* Git
* Python compatible with the pinned dependencies in `requirements.txt`
* Sufficient storage for the dataset and any model checkpoints
* Authorized access to MILK10k under its applicable license

Dependency compatibility depends on your operating system, Python version, and hardware. The installation steps below have not been verified on every environment.

### 1. Clone the repository

```bash
git clone https://github.com/mdtasnimulhoque/dermalens-ai.git
cd dermalens-ai
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins specific package versions. If installation fails, check whether those versions are available for your Python version and platform before changing the pins.

### 4. Configure the dataset

Obtain MILK10k from its authorized source and review its licensing conditions. Configure the local dataset paths using the project's path utilities and `configs/training.yaml` as appropriate.

Dataset files and locally generated indexes or splits are intentionally excluded from version control.

The exact data preparation and training commands depend on the local dataset layout and the individual scripts. Review their configuration and argument handling before running them.

### 5. Explore the source code

Useful starting points include:

* `configs/training.yaml` — training configuration.
* `src/data/` — dataset auditing, validation, indexing, splitting, and loading.
* `src/models/` — model definitions.
* `src/training/` — baseline training scripts.
* `src/evaluate_baseline_e.py` — Baseline E evaluation script.
* `src/evaluate_calibration.py` — calibration evaluation utility.
* `reports/baseline_comparison.md` — baseline comparison report.
* `.github/workflows/main.yml` — automated Python syntax check.

**Reproducibility note:** No trained checkpoint is distributed with this repository. Reproducing the reported results requires authorized dataset access, compatible dependencies, the relevant scripts and configuration, and a suitable compute environment. Exact reproduction has not been established across different environments.

---

## Responsible Use and Limitations

**Dermalens AI is an experimental research project, not a validated medical device.**

The project has not been established as safe or effective for clinical diagnosis, screening, treatment selection, or ruling out disease.

The reported results come from a particular dataset and experimental setup. They should not be generalized to all populations, skin types, imaging devices, or clinical environments.

Important limitations include:

* Class imbalance and limited examples for some categories.
* Potential differences between the research dataset and real-world images.
* Use of clinical close-up images for the reported baselines rather than a validated general-purpose phone-camera workflow.
* The need for independent validation, calibration, uncertainty analysis, and evaluation across relevant populations.
* The possibility that visual explanations may be incomplete or misleading.
* The absence of evidence that these model predictions improve clinical outcomes.

**Do not use this project to self-diagnose, delay medical care, or make treatment decisions.** If you have a concerning or changing skin lesion, seek advice from a qualified healthcare professional.

The long-term goal is to contribute to research into transparent, reproducible, and carefully evaluated AI—not to replace professional medical judgment.

---

## Contributing

Contributions that improve scientific quality, reproducibility, accessibility, or responsible AI practices are welcome.

Potential areas for contribution include:

* Strengthening dataset validation and leakage checks.
* Adding unit tests and clearer configuration handling.
* Improving metric reporting and error analysis.
* Evaluating calibration and uncertainty.
* Investigating explainability methods and their limitations.
* Improving setup instructions and reproducibility.
* Reviewing responsible-use safeguards and documentation.

### Before contributing

1. Open an issue describing the problem or proposed improvement.
2. Keep changes focused and document experimental assumptions.
3. Never commit restricted datasets, private information, credentials, or large model checkpoints.
4. Document the evaluation methodology and limitations alongside performance claims.
5. Avoid presenting unvalidated results as clinical evidence.

For substantial changes, discuss the proposed approach in an issue before opening a pull request.

---

## Research Integrity

Meaningful comparisons require clear documentation of the dataset version, preprocessing, split definitions, training configuration, model-selection criteria, and evaluation metrics.

Class-level results should be reported with support counts, particularly for rare categories. Model selection should use validation data rather than repeatedly optimizing against the same held-out test set.

Future external evaluations should respect dataset licensing, appropriate data governance, and relevant research ethics requirements.

Reproducibility and transparent reporting are core goals of this project. Experimental results should be described accurately, including negative results, limitations, and unresolved questions.

---

## License

The project source code is governed by the license in the repository's [`LICENSE`](LICENSE) file. Review that file before reusing, modifying, or redistributing the code.

MILK10k and other third-party materials remain subject to their own licensing terms. The project code license does not grant additional rights to those materials.

---

## Acknowledgments

This project builds on the work of the researchers, dataset creators, and open-source contributors whose work makes machine-learning research possible.

Please consult the original MILK10k documentation for the appropriate dataset citation, attribution, and usage requirements.

---

<p align="center">
  <strong>Dermalens AI</strong><br>
  Researching skin-lesion image analysis with explainability, reproducibility, and responsibility.
</p>
