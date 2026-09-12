# -Methodological-Robustness-in-Machine-Learning-Studies

# A Framework for Evaluating Methodological Robustness in Machine Learning Studies for Clinical Autism Research

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)

This repository contains the notebook and computational materials associated with the paper:

> **A Framework for Evaluating Methodological Robustness in Machine Learning Studies for Clinical Autism Research**

The project presents two empirical case studies that illustrate methodological risks affecting machine learning (ML) research using clinical Autism Spectrum Disorder (ASD) data.

The analyses focus on:

- deterministic label leakage;
- feature-selection leakage;
- instability of feature selection across subsamples;
- class imbalance and the interpretation of predictive metrics;
- multisite heterogeneity;
- measurement-instrument confounding;
- permutation-based inference;
- multiple-comparison correction; and
- reproducibility through transparent computational reporting.

> **Clinical-use disclaimer:** This repository does not provide a diagnostic tool, a clinical decision-support system, or a validated predictive model for ASD diagnosis. The datasets and analyses are used exclusively as empirical case studies for methodological evaluation.

---

## Citation

If you use this repository, please cite the associated manuscript:

```text
J. Mata de Figueiredo, “A Framework for Evaluating Methodological
Robustness in Machine Learning Studies for Clinical Autism Research,”
manuscript submitted for conference publication.
```

Repository: <a href="https://github.com/juanmata77/Methodological-Robustness-in-Machine-Learning-Studies" target="_blank" style="text-decoration: underline;">github.com/juanmata77/Methodological-Robustness-in-Machine-Learning-Studies</a>

---

## Repository Structure

```text
.
├── QCHATeABIDE.ipynb
├── README.md
├── LICENSE
└── data/
    ├── ABIDEII_Long_Composite_Phenotypic.csv
    └── Toddler Autism dataset July 2018.csv
```

### Main notebook

| File | Description |
|---|---|
| `QCHATeABIDE.ipynb` | Main Jupyter notebook containing exploratory data processing, statistical tests, and machine-learning analyses for the Q-CHAT-10 Toddler and ABIDE longitudinal case studies. |

---

## Case Studies

### Case Study A — ABIDE II Longitudinal

The ABIDE case study examines the robustness of analyses involving a small longitudinal and multisite sample.

The manuscript investigates:

- longitudinal data availability and structure;
- group differences in Performance IQ (PIQ), age, and sex;
- instability across validation schemes;
- Leave-Site-Out validation;
- IQ-test-type confounding;
- site-specific models;
- covariate inclusion;
- effect sizes and bootstrap confidence intervals; and
- correction for multiple comparisons.

The analytical sample reported in the manuscript contains:

- **38 participants**;
- **23 ASD participants**;
- **15 typically developing (TD) participants**; and
- baseline and follow-up observations used only where longitudinal comparisons are applicable.

### Case Study B — Q-CHAT-10 Toddler

The Q-CHAT-10 Toddler case study illustrates risks associated with label construction and feature selection.

The manuscript examines:

- deterministic leakage between `Qchat-10-Score` and `Class`;
- associations between individual questionnaire items and the class label;
- item--rest correlations;
- classification using questionnaire items that contribute to the construction of the class label;
- raw accuracy, balanced accuracy, sensitivity, and specificity;
- feature selection performed outside versus inside the cross-validation loop; and
- instability of selected features across LOOCV folds.

The analytical sample reported in the manuscript contains **1,054 observations**.

---

## Data Availability

The datasets are publicly available but are **not redistributed in this repository**. Users must obtain them directly from their original sources and comply with the applicable terms of use.

### 1. ABIDE II Longitudinal data

The ABIDE data are available through the International Neuroimaging Data-sharing Initiative (INDI) and associated data-sharing resources.

- <a href="https://fcon_1000.projects.nitrc.org/indi/abide/" target="_blank" style="text-decoration: underline;">ABIDE / INDI project page</a>
- <a href="https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html" target="_blank" style="text-decoration: underline;">ABIDE II information page</a>

Place the phenotype file in the `data/` directory using the following filename:

```text
ABIDEII_Long_Composite_Phenotypic.csv
```

### 2. Q-CHAT-10 Toddler dataset

The toddler screening dataset is associated with the dataset described by Thabtah and collaborators.

- <a href="https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+toddlers" target="_blank" style="text-decoration: underline;">UCI Machine Learning Repository: Autistic Spectrum Disorder Screening Data for Toddlers</a>

Place the dataset in the `data/` directory using the following filename:

```text
Toddler Autism dataset July 2018.csv
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/juanmata77/Methodological-Robustness-in-Machine-Learning-Studies.git
cd Methodological-Robustness-in-Machine-Learning-Studies
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scipy statsmodels scikit-learn jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
QCHATeABIDE.ipynb
```

---

## Required Python Packages

The notebook uses the following main packages:

| Package | Purpose |
|---|---|
| `pandas` | Dataset loading, cleaning, tabular data manipulation |
| `numpy` | Numerical operations and random permutation generation |
| `scipy` | Statistical tests, including Wilcoxon, chi-square, and t-tests |
| `statsmodels` | False Discovery Rate correction using Benjamini--Hochberg |
| `scikit-learn` | Classification models, cross-validation, preprocessing, and evaluation metrics |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical data visualization |

A suitable environment is Python **3.10 or later**.

---

## Running the Notebook

The current notebook was initially developed in Google Colab and loads datasets from `/content/`.

For local execution, update the first data-loading cell from:

```python
df_abide = pd.read_csv('/content/ABIDEII_Long_Composite_Phenotypic.csv')
df_toddler_autism = pd.read_csv('/content/Toddler Autism dataset July 2018.csv')
```

to:

```python
df_abide = pd.read_csv('data/ABIDEII_Long_Composite_Phenotypic.csv')
df_toddler_autism = pd.read_csv('data/Toddler Autism dataset July 2018.csv')
```

After the paths are updated and the required files are available in `data/`, run all notebook cells sequentially.

---

## Methodological Notes

### Deterministic label leakage

In the Q-CHAT-10 Toddler dataset:

- `Qchat-10-Score` is computed as the sum of items `A1` through `A10`;
- `Class` is derived by thresholding that score.

Consequently, using `Qchat-10-Score` to predict `Class` is circular: the predictor directly participates in the construction of the target. A high or perfect accuracy in this setting does **not** demonstrate independent diagnostic performance.

### Feature-selection leakage

Feature selection must be performed using only the training data of each cross-validation fold. Selecting variables from the complete dataset before cross-validation allows information from observations later used for testing to influence the predictor set.

The intended comparison is:

1. **Leaked procedure:** select features once using the full dataset, then evaluate the fixed subset with cross;
2. **Properly nested procedure:** repeat feature selection independently within each training fold before evaluating the held-out observation.

### Multisite validation

When data originate from multiple acquisition sites, random cross-validation alone may not answer the relevant generalization question. Leave-Site-Out validation provides a more appropriate assessment when the objective is to evaluate generalization across collection centers.

---

## Reproducibility Status

This repository is intended to support transparent inspection and reproduction of the analyses described in the associated manuscript.

Before using the repository as a fully reproducible archival version of the paper, the following consistency checks should be completed:

| Component | Status to verify |
|---|---|
| Dataset filenames and local paths | Update from Google Colab `/content/` paths to relative repository paths |
| Random seeds | Ensure that all stochastic procedures use documented fixed seeds |
| Deterministic leakage evaluation | Ensure that the implementation uses the same validation procedure reported in the manuscript |
| Item-level association analyses | Ensure that the implemented association statistic and correction procedures match the manuscript |
| Classifier comparison table | Ensure that accuracy, balanced accuracy, sensitivity, and specificity match the reported values |
| Nested feature selection | Ensure that ranking and selection occur independently in every training fold |
| ABIDE analyses | Ensure that all reported ABIDE analyses, including site-level validation and multiplicity correction, are implemented and executable |
| Environment specification | Add a `requirements.txt` or `environment.yml` file with tested package versions |

The public repository should be tagged with a release identifier corresponding to the manuscript version used for submission.

---

## Recommended Reproducible Environment File

For long-term reproducibility, create a `requirements.txt` file with the package versions used to run the final analyses.

Example:

```text
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

For a final archival release, replace version ranges with exact tested versions.

---

## Limitations

- The ABIDE case study uses a small sample, particularly for site-specific analyses.
- The Q-CHAT-10 case study concerns a screening-related dataset and a label derived from questionnaire-score rules; it must not be interpreted as external clinical diagnostic validation.
- The two case studies illustrate methodological risks but do not constitute a formal validation of the proposed framework as a universal audit instrument.
- Results may vary if datasets are updated, differently preprocessed, or accessed through modified versions.

---

## Ethical and Clinical Statement

This project uses publicly available, de-identified research datasets. No new participant data were collected.

The analyses are methodological demonstrations. They are not intended to diagnose ASD, replace clinical judgment, guide treatment, or support individual-level medical decisions.

---

## License

The code in this repository is distributed under the MIT License. See the `LICENSE` file for details.

The datasets are governed by their respective original licenses, access policies, and terms of use. Dataset rights remain with the corresponding data providers and original authors.

---

## References

1. C. Allison, B. Auyeung, and S. Baron-Cohen, “Toward brief ‘red flags’ for autism screening: the short autism spectrum quotient and the short quantitative checklist for autism in toddlers in 1,000 cases and 3,000 controls,” *Journal of the American Academy of Child & Adolescent Psychiatry*, vol. 51, no. 2, pp. 202--212, 2012.

2. J. A. Nielsen et al., “Multisite functional connectivity MRI classification of autism: ABIDE results,” *Frontiers in Human Neuroscience*, vol. 7, Art. 599, 2013. doi: 10.3389/fnhum.2013.00599.

3. M. Rosenblatt et al., “Data leakage inflates prediction performance in connectome-based machine learning models,” *Nature Communications*, vol. 15, Art. 1829, 2024. doi: 10.1038/s41467-024-46150-w.

4. G. Tartarisco et al., “Use of machine learning to investigate the Quantitative Checklist for Autism in Toddlers (Q-CHAT) towards early autism screening,” *Diagnostics*, vol. 11, no. 3, p. 574, 2021. doi: 10.3390/diagnostics11030574.

5. F. Thabtah, F. Kamalov, and K. Rajab, “A new computational intelligence approach to detect autistic features for autism screening,” *International Journal of Medical Informatics*, vol. 117, pp. 112--124, 2018.
