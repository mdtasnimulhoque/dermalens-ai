# Dermalens AI

### Explainable AI Research for Skin-Lesion Image Analysis

**Building transparent, reproducible, and responsible AI for skin-lesion image research.**

Dermalens AI is an open-source research project exploring how deep learning and explainable artificial intelligence (XAI) can support the analysis of skin-lesion images. The project focuses on image classification, careful evaluation, model interpretability, and the limitations that must be addressed before AI systems can be considered for clinical use.

The goal is not to replace dermatologists or make automated diagnoses. It is to investigate how AI models can be developed, evaluated, and explained more responsibly.

<p align="center">
  <strong>Deep Learning · Explainable AI · Medical Imaging · Responsible AI</strong>
</p>

---

## Overview

Skin-lesion image analysis is a challenging machine-learning problem. Models must learn meaningful visual patterns across diverse lesion categories while dealing with class imbalance, limited examples for uncommon categories, uncertainty, and the risk of learning misleading patterns from training data.

Dermalens AI approaches this as a research problem rather than treating model accuracy as the only measure of success.

The project currently includes a dataset validation workflow, lesion-level data splitting, multiple ResNet-50 baseline experiments, and evaluation of classification performance. Further work is planned around calibration, visual explanations, uncertainty estimation, and safer inference.

### Research objectives

* **Develop image-classification baselines:** Establish and compare deep-learning approaches for skin-lesion image analysis.
* **Evaluate beyond accuracy:** Measure balanced accuracy, Macro-F1, precision, recall, and class-specific performance.
* **Reduce evaluation leakage:** Keep images belonging to the same lesion within a single data split.
* **Investigate explainability:** Explore techniques such as Grad-CAM to visualize image regions that influence model predictions.
* **Study uncertainty:** Investigate when a model should express uncertainty instead of presenting an unreliable prediction confidently.
* **Promote responsible use:** Communicate limitations, avoid unsupported medical claims, and prioritize appropriate professional evaluation.

---

## Current research status

**Current best experiment: Baseline E — ResNet-50 with data augmentation, class balancing, and label smoothing.**

The model was evaluated on a held-out test split from the MILK10k dataset. The test split contains 1,064 images from 532 lesions.

| Metric            | Test result |
| ----------------- | ----------: |
| Accuracy          |      60.53% |
| Balanced accuracy |      43.54% |
| Macro-F1          |      44.69% |
| Macro precision   |      49.09% |
| Macro recall      |      43.54% |

These values are research results from the current experimental setup, not evidence of clinical effectiveness.

### What do these metrics mean?

* **Accuracy:** The proportion of test images classified correctly.
* **Balanced accuracy:** The average recall across classes, giving each class equal weight.
* **Macro-F1:** The unweighted average of each class's F1 score, helping reveal performance on less frequent classes.
* **Macro precision and recall:** Additional views of the model's prediction quality across classes.

The difference between accuracy and balanced accuracy highlights why aggregate accuracy alone can be misleading on an imbalanced dataset.

Some categories have very few examples in the test split. Their individual metrics can therefore be unstable. Performance must be interpreted alongside class support, confusion patterns, uncertainty, and independent validation.

### Baseline comparison

| Experiment                                    | Test accuracy | Balanced accuracy | Macro-F1 |
| --------------------------------------------- | ------------: | ----------------: | -------: |
| Baseline A — Plain ResNet-50                  |        62.59% |            31.86% |   33.16% |
| Baseline B — Augmentation and class balancing |        61.84% |            42.49% |   42.86% |
| Baseline C — Image and metadata               |        59.21% |            43.45% |   38.98% |
| Baseline D — Multimodal with regularization   |        59.40% |            35.83% |   35.24% |
| Baseline E — Label smoothing                  |        60.53% |            43.54% |   44.69% |

Baseline E currently has the highest Macro-F1 among these experiments. Baseline C has a slightly lower balanced accuracy than E but a substantially lower Macro-F1.

These experiments illustrate that a modification does not necessarily improve every metric. Future comparisons should use validation data for model selection and reserve independent data for final evaluation.

**Evaluation note:** The held-out test split has already been used to compare these baselines. It should not become a repeatedly optimized tuning set. Further model selection should rely on validation data, followed by evaluation on genuinely independent data when available.

---

## Dataset

Dermalens AI currently uses **MILK10k** for research and experimentation.

The dataset contains 10,480 images representing 5,240 lesions, with a clinical close-up image and a dermoscopic image for each lesion. The project currently trains and evaluates the reported image-classification baselines using the clinical close-up modality.

The 11 target categories are:

| Label   | Category                                               |
| ------- | ------------------------------------------------------ |
| AKIEC   | Actinic keratosis / intraepithelial carcinoma category |
| BCC     | Basal cell carcinoma                                   |
| BEN_OTH | Other benign lesions                                   |
| BKL     | Benign keratosis-like lesions                          |
| DF      | Dermatofibroma                                         |
| INF     | Inflammatory category                                  |
| MAL_OTH | Other malignant lesions                                |
| MEL     | Melanoma                                               |
| NV      | Melanocytic nevi                                       |
| SCCKA   | Squamous cell carcinoma / keratoacanthoma category     |
| VASC    | Vascular lesions                                       |

The exact category definitions should be interpreted according to the dataset's official documentation.

### Data validation and splitting

The local data-audit workflow checked dataset integrity, and the current split was constructed at the lesion level.

| Split      |   Lesions |     Images |
| ---------- | --------: | ---------: |
| Training   |     4,188 |      8,376 |
| Validation |       520 |      1,040 |
| Test       |       532 |      1,064 |
| **Total**  | **5,240** | **10,480** |

Images from the same lesion remain in the same split, preventing the clinical and dermoscopic views of one lesion from being divided across training and evaluation sets.

This reduces one important source of data leakage. It does not, by itself, establish patient-level independence or guarantee generalization to other datasets and clinical settings.

### Dataset licensing and access

MILK10k is subject to its own **CC BY-NC license terms**. Review the original dataset documentation and license before downloading, using, modifying, or redistributing it.

* The dataset is not included in this repository.
* Do not upload the original images, downloaded archives, or local dataset indexes to GitHub.
* Check whether your intended use is permitted by the dataset license, especially for commercial applications.
* Cite and acknowledge the original dataset and its creators according to their requirements.

Dataset licensing is separate from the license governing this project's source code.

---

## Technical approach

### Current implementation

* **Deep-learning architecture:** ResNet-50 image-classification baseline.
* **Framework:** PyTorch and Torchvision.
* **Data handling:** Python, pandas, and Pillow.
* **Evaluation:** scikit-learn and custom evaluation scripts.
* **Training environment:** Apple Silicon with PyTorch MPS support available.
* **Experiment tracking:** Configuration files, training histories, metrics, and confusion matrices.

### Experiments

The current research progression includes:

1. **Baseline A — Plain image classifier:** Establishes an initial ResNet-50 reference.
2. **Baseline B — Augmentation and class balancing:** Investigates whether training-time augmentation and class weighting improve minority-class performance.
3. **Baseline C — Image plus metadata:** Explores combining image features with available metadata.
4. **Baseline D — Regularized multimodal model:** Investigates a modified training strategy with frozen early layers.
5. **Baseline E — Label smoothing:** Adds label smoothing to the augmentation and class-balancing approach.

The experiments are not guaranteed to improve performance monotonically. Their results should be assessed using the same data-splitting policy and clearly documented evaluation criteria.

---

## Explainable AI and future direction

A model's predicted category alone does not explain whether its decision is reliable or what visual evidence influenced it.

The following capabilities are research priorities, not claims that a finished clinical system already provides them.

* **Grad-CAM visualizations:** Investigate which image regions contribute to model predictions.
* **Confidence calibration:** Examine whether confidence scores correspond to observed correctness.
* **Uncertainty estimation:** Identify predictions that warrant caution.
* **Abstention:** Explore whether the system should decline to provide a result when confidence or image quality is inadequate.
* **Error analysis:** Examine confusion between categories and performance on underrepresented classes.
* **External validation:** Evaluate generalization on independent datasets, subject to their licenses and appropriate research safeguards.
* **Reproducibility:** Document configurations, preprocessing, random seeds, splits, and evaluation procedures.

Visual explanations are not proof that a prediction is medically correct. Likewise, a high confidence score does not guarantee correctness.

---

## Repository structure

```text
dermalens-ai/
├── app/
│   ├── backend/
│   └── frontend/
├── configs/
│   └── training.yaml
├── data/
│   ├── raw/                 # Local datasets; not committed
│   ├── processed/           # Local processed data; not committed
│   └── metadata/            # Local indexes and splits; not committed
├── docs/
├── models/
│   └── checkpoints/         # Local model checkpoints; not committed
├── notebooks/
├── reports/
│   ├── baseline_comparison.md
│   ├── baseline_e_test_metrics.txt
│   └── ...
├── src/
│   ├── data/
│   ├── explainability/
│   ├── inference/
│   ├── models/
│   ├── safety/
│   └── training/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The structure represents the current project organization and planned application areas. Some directories, including the application, tests, and explainability modules, may not yet contain a complete implementation.

---

## Getting started

### Requirements

* Python 3
* Git
* A compatible PyTorch installation
* Sufficient storage for the dataset and model weights
* Access to MILK10k under its applicable license

The project has been developed and tested in a local Apple Silicon environment. Other operating systems and hardware configurations may require different dependency installation steps.

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

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Obtain and configure the dataset

Obtain MILK10k through its authorized source, review its license, and configure the local dataset paths according to the project scripts and `configs/training.yaml`.

The dataset, local indexes, and split files are intentionally excluded from version control. The exact dataset setup and training commands should be documented alongside the corresponding scripts as the workflow becomes more reproducible.

### 5. Explore the code

Start with:

* `configs/training.yaml` — training configuration.
* `src/data/` — dataset auditing, validation, indexing, splitting, and loading.
* `src/models/` — model architectures.
* `src/training/` — baseline training scripts.
* `src/evaluate_baseline_e.py` — evaluation for Baseline E.
* `reports/` — saved experiment summaries and metrics.

**Reproducibility note:** A trained checkpoint is not currently distributed with this repository. Reproducing the reported results requires the permitted dataset, compatible dependencies, the corresponding training/evaluation workflow, and appropriate configuration.

---

## Responsible use and limitations

Dermalens AI is an **experimental research project, not a medical device**.

It has not been established as safe or effective for clinical diagnosis, screening, treatment selection, or ruling out disease. The reported results come from a specific dataset and split and should not be generalized to all populations, skin types, imaging devices, or clinical environments.

Important limitations include:

* Imbalanced target categories and limited examples for some classes.
* Potential differences between the training dataset and real-world images.
* The current use of clinical close-up images for the reported baselines, rather than a validated general-purpose phone-camera workflow.
* The need for calibration, uncertainty analysis, independent validation, and evaluation across relevant populations.
* The possibility that visual explanations may be misleading or incomplete.
* The lack of evidence that model predictions improve clinical outcomes.

**Please do not use this project to self-diagnose, delay medical care, or make treatment decisions.** If you have a concerning or changing skin lesion, seek advice from a qualified healthcare professional.

The project's long-term aim is to contribute to research into more transparent and carefully evaluated AI—not to replace professional medical judgment.

---

## Contributing

Contributions that improve the scientific quality, reproducibility, accessibility, or safety of this project are welcome.

Potential areas include:

* Improving dataset validation and leakage checks.
* Adding unit tests and clearer configuration handling.
* Improving metric reporting and error analysis.
* Implementing calibration and uncertainty evaluation.
* Exploring explainability methods and their limitations.
* Improving documentation and reproducible setup instructions.
* Reviewing responsible-use safeguards and accessibility.

Before submitting a contribution:

1. Open an issue describing the problem or proposed improvement.
2. Keep changes focused and document any experimental assumptions.
3. Do not include restricted datasets, private information, credentials, or large model checkpoints.
4. Report evaluation methodology and limitations alongside performance claims.
5. Avoid presenting unvalidated results as clinical evidence.

---

## Research integrity

For meaningful comparisons, experiments should document the dataset version, preprocessing, split definitions, training configuration, model-selection criterion, and evaluation metrics.

Class-wise metrics should be reported with support counts, particularly for rare categories. Model selection should use validation data; the held-out test set should not be repeatedly used to optimize model choices.

Any future external evaluation should respect dataset licensing, appropriate data governance, and relevant research ethics requirements.

---

## License

The project source code is governed by the license in the repository's `LICENSE` file. Review that file before reusing or redistributing code.

The MILK10k dataset and other third-party assets retain their own licensing terms. The project code license does not grant additional rights to those materials.

---

## Acknowledgments

This project builds on the work of the researchers and contributors who created the datasets, open-source libraries, and machine-learning tools used in this research.

Please consult the original MILK10k documentation for the appropriate dataset citation and attribution requirements.

---

<p align="center">
  <strong>Dermalens AI</strong><br>
  Researching skin-lesion image analysis with explainability, reproducibility, and responsibility.
</p>
