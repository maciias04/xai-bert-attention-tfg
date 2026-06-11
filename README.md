# Explainable Artificial Intelligence over BERT Attention Maps

## PCA, oMEDA, SHAP, PLS-DA and ASCA for Transformer Interpretability

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-MEDA%20Toolbox-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repository associated with the Final Degree Project:

> **Inteligencia Artificial Explicativa (XAI): Métodos y Prueba de Concepto**

**Author:** Adrián Macias Caballero
**Academic supervisor:** José Camacho Páez
**Degree:** B.Sc. in Telecommunication Technologies Engineering
**University:** Universidad de Granada
**School:** Escuela Técnica Superior de Ingenierías Informática y de Telecomunicación
**Date:** June 2026

---

## Table of contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Why this project matters](#why-this-project-matters)
4. [Main contribution: multivariate XAI over BERT attention maps](#main-contribution-multivariate-xai-over-bert-attention-maps)
5. [Research questions](#research-questions)
6. [Model used](#model-used)
7. [Experimental dataset](#experimental-dataset)
8. [Attention matrix representation](#attention-matrix-representation)
9. [Methodology](#methodology)
10. [Methods implemented](#methods-implemented)
11. [Repository structure](#repository-structure)
12. [Installation](#installation)
13. [How to reproduce the analysis](#how-to-reproduce-the-analysis)
14. [Outputs and results](#outputs-and-results)
15. [Main findings](#main-findings)
16. [Interpretation: SHAP vs multivariate methods](#interpretation-shap-vs-multivariate-methods)
17. [Limitations](#limitations)
18. [Future work](#future-work)
19. [Scientific and technical contribution](#scientific-and-technical-contribution)
20. [Citation](#citation)
21. [License](#license)
22. [Acknowledgements](#acknowledgements)
23. [Academic disclaimer](#academic-disclaimer)

---

## Overview

This repository contains the code, experimental data, MATLAB scripts, Python scripts, intermediate files and result figures developed as part of the Final Degree Project **“Inteligencia Artificial Explicativa (XAI): Métodos y Prueba de Concepto”**.

The project focuses on **Explainable Artificial Intelligence (XAI)** applied to a Transformer-based language model. More specifically, it studies the internal behaviour of a **BERT model fine-tuned for binary sentiment classification**.

The goal is not to train a new neural network. Instead, the project analyses an already trained BERT classifier by extracting its **attention maps** and transforming them into a structured multivariate dataset. This dataset is then analysed using several complementary techniques:

* **PCA** — Principal Component Analysis.
* **oMEDA** — observation-based Missing-data methods for Exploratory Data Analysis.
* **SHAP** — SHapley Additive exPlanations.
* **PLS-DA** — Partial Least Squares Discriminant Analysis.
* **ASCA** — ANOVA-Simultaneous Component Analysis.

The central idea is to treat BERT attention maps not only as visual objects, but also as **high-dimensional data structures** that can be studied using multivariate analysis.

---

## Motivation

Modern deep learning models have achieved strong performance in tasks such as natural language processing, computer vision, classification, recommendation systems and decision support. However, this predictive power often comes at the cost of interpretability.

Transformer-based models such as BERT are particularly difficult to interpret because their internal processing is distributed across:

* multiple Transformer layers,
* multiple attention heads per layer,
* contextual token embeddings,
* residual connections,
* layer normalisation blocks,
* nonlinear feed-forward networks,
* high-dimensional internal representations.

Although attention maps provide a useful visual entry point into the internal behaviour of Transformers, attention should not be automatically considered a complete explanation. Attention weights show how tokens attend to each other, but they do not necessarily measure the direct contribution of each token to the final prediction.

This project is motivated by the need to combine **different levels of explanation**:

1. **Token-level attribution**, using SHAP.
2. **Structural analysis of internal attention variables**, using PCA, oMEDA, PLS-DA and ASCA.

By combining these perspectives, the project provides a richer and more cautious interpretation of BERT’s behaviour in a sentiment classification task.

---

## Why this project matters

Many XAI analyses focus either on the input-output behaviour of a model or on isolated visualisations of internal activations. This project explores a complementary direction: using attention maps as a complete multivariate object.

This matters for three main reasons.

First, BERT attention maps contain information about how the model internally relates the tokens of a sentence. These relations are not isolated: they depend on the layer, the attention head, the source token and the target token.

Second, methods such as PCA, oMEDA, PLS-DA and ASCA can help identify which internal attention variables are most associated with the difference between positive and negative sentences. This makes it possible to move from individual attention heatmaps to a more systematic analysis of internal behaviour.

Third, SHAP and the multivariate methods do not explain the same thing. SHAP focuses on the contribution of input tokens to the final prediction. In contrast, oMEDA, PLS-DA and ASCA analyse internal attention relationships. Therefore, discrepancies between SHAP and the multivariate rankings should not be interpreted as errors by default, but as evidence that the methods operate at different explanatory levels.

---

## Main contribution: multivariate XAI over BERT attention maps

The main contribution of this project is the development of a proof of concept for applying multivariate analysis methods to BERT attention maps.

Within the scope of the thesis and the reviewed literature, this work is presented as a **pioneering exploratory application** of **oMEDA, PLS-DA and ASCA** to attention maps from a Transformer-based language model in the XAI domain.

A careful formulation is important: this repository does not claim an absolute universal priority without qualification. Instead, it presents the work as:

* one of the first exploratory applications of these multivariate methods to BERT attention maps,
* a proof of concept connecting multivariate data analysis and Transformer interpretability,
* a methodological bridge between attention-based analysis and XAI,
* a starting point for future research on multivariate interpretability of language models.

The key methodological step is the transformation:

```text
BERT attention maps
        ↓
Layer × Head × Source token × Target token
        ↓
High-dimensional attention matrix
        ↓
PCA / oMEDA / PLS-DA / ASCA / SHAP comparison
```

Instead of analysing a single attention image, this approach transforms all attention weights into a numerical matrix that can be explored statistically.

---

## Research questions

The project is guided by the following research questions:

1. Can BERT attention maps reveal internal differences between positive and negative sentences?
2. Can attention maps be transformed into a multivariate matrix suitable for statistical analysis?
3. Are the most relevant attention variables concentrated in specific layers or heads?
4. Do the final layers of BERT carry greater relevance for sentiment classification?
5. Can supervised multivariate methods such as PLS-DA discriminate between positive and negative sentences using only attention variables?
6. Can ASCA detect an effect of the sentiment class over the structure of the attention maps?
7. How do SHAP token attributions compare with the structural importance obtained from attention-based multivariate methods?
8. What are the limitations of interpreting attention as explanation?

---

## Model used

The model used in this project is:

```text
textattack/bert-base-uncased-SST-2
```

This is a BERT-base model fine-tuned for binary sentiment classification on SST-2.

| Feature                                    | Value                           |
| ------------------------------------------ | ------------------------------- |
| Base architecture                          | BERT-base                       |
| Transformer type                           | Encoder-only                    |
| Task                                       | Binary sentiment classification |
| Classes                                    | Positive / Negative             |
| Tokenisation                               | WordPiece                       |
| Case sensitivity                           | Uncased                         |
| Number of Transformer layers               | 12                              |
| Attention heads per layer                  | 12                              |
| Fixed sequence length used in this project | 8 tokens                        |

In this project, the model receives an English sentence and returns a sentiment prediction together with the associated class probabilities. The internal attention matrices are then extracted for further analysis.

---

## Experimental dataset

The experimental dataset consists of **40 controlled English sentences**:

* **20 positive sentences**
* **20 negative sentences**

The dataset was designed in a paired way. For each positive sentence, there is a structurally similar negative sentence, mainly changing the word that introduces the sentiment polarity.

Example:

```text
Positive: The movie was very good.
Negative: The movie was very bad.
```

Another example:

```text
Positive: I found the story charming.
Negative: I found the story boring.
```

This controlled design helps reduce linguistic variability and makes the comparison between positive and negative attention patterns more focused.

The sentence files are stored in:

```text
data/sentences/
├── positive_sentences.txt
└── negative_sentences.txt
```

---

## Attention matrix representation

For each input sentence, BERT produces attention matrices with the following structure:

```text
Layer × Head × Source token × Target token
```

In this project:

```text
12 layers × 12 heads × 8 source tokens × 8 target tokens = 9216 attention variables
```

Therefore, each sentence is represented by a vector of **9216 attention values**.

Since the experimental dataset contains 40 sentences, the final attention matrix has the following size:

```text
40 observations × 9216 variables
```

Each row corresponds to one sentence.
Each column corresponds to one specific attention relationship:

```text
layer_i / head_j / source_token_k / target_token_l
```

This representation is the basis for the multivariate analyses carried out in MATLAB and Python.

---

## Methodology

The complete workflow is divided into six main stages.

### 1. Sentence preparation

The positive and negative sentences are stored as plain text files. The dataset is balanced and controlled to facilitate comparison between sentiment classes.

### 2. BERT inference

The BERT model is loaded using Hugging Face Transformers and PyTorch. Each sentence is tokenised and passed through the model.

For each sentence, the following outputs are obtained:

* predicted class,
* positive class probability,
* negative class probability,
* tokenised input sequence,
* attention maps for all layers and heads.

### 3. Attention extraction

The attention maps are extracted from BERT and stored in a structured format. Each attention value is associated with:

* layer,
* head,
* source token,
* target token,
* sentence index,
* sentiment class.

### 4. Matrix construction

The attention maps are vectorised to construct the final matrix:

```text
X = 40 × 9216
```

This matrix is used as input for PCA, oMEDA, PLS-DA and ASCA.

### 5. Multivariate analysis

The matrix is analysed using MATLAB and the MEDA Toolbox. The analyses include:

* PCA score plots,
* explained variance,
* oMEDA variable contribution,
* PLS-DA latent variables,
* PLS-DA prediction and confusion matrix,
* ASCA class effect,
* importance by layer,
* importance by head,
* importance by layer-head pair,
* importance by token position,
* importance by real token.

### 6. SHAP comparison

SHAP is applied directly to the BERT classifier. Unlike the attention-based multivariate methods, SHAP assigns importance to the real input tokens according to their contribution to the final prediction.

---

## Methods implemented

### PCA

**Principal Component Analysis** is used as the first exploratory method.

In this project, PCA is applied to the `40 × 9216` attention matrix to study the global structure of the observations.

PCA is used to:

* reduce dimensionality,
* visualise the sentences in a latent space,
* inspect whether positive and negative sentences show natural separation,
* analyse explained variance,
* provide a basis for further oMEDA analysis.

The main MATLAB script associated with PCA is:

```text
src/matlab/pca_medatoolbox.m
```

The corresponding results are stored in:

```text
results/pca/
```

---

### oMEDA

**oMEDA** is used to interpret which original variables contribute to a difference observed in the multivariate space.

In this project, oMEDA is applied to attention variables. Each variable corresponds to a specific relationship between two token positions in a specific layer and head.

oMEDA is used to study:

* relevant attention variables,
* importance by layer,
* importance by head,
* importance by layer-head pair,
* relevant source and target token positions,
* token-level structural importance.

The main MATLAB script is:

```text
src/matlab/omeda_toolbox.m
```

The associated files and figures are stored in:

```text
results/omeda/
data/attention/
```

Examples of generated CSV files include:

```text
data/attention/omeda_meda_importancia_layer.csv
data/attention/omeda_meda_importancia_tokens.csv
data/attention/omeda_meda_top50_variables.csv
data/attention/omeda_meda_variables_relevantes.csv
data/attention/omeda_meda_vector_variables.csv
```

---

### SHAP

**SHAP** is used as a post-hoc attribution method based on Shapley values.

In this project, SHAP is applied directly to the input tokens of the BERT classifier. This makes it possible to analyse which words contribute most strongly to the final sentiment prediction.

SHAP is used to:

* explain individual predictions,
* identify the most influential real tokens,
* obtain global token importance rankings,
* compare lexical attribution with structural attention-based importance.

The main Python scripts are:

```text
src/python/shap_bert_analysis.py
src/python/shap_analysisespecialimportanciatokens.py
```

The corresponding results are stored in:

```text
results/shap/
```

---

### PLS-DA

**Partial Least Squares Discriminant Analysis** is used as a supervised multivariate method.

Unlike PCA, PLS-DA uses the sentiment labels during model fitting. It searches for latent variables that maximise the relationship between the attention matrix and the class label.

PLS-DA is used to:

* evaluate whether attention variables contain discriminant information,
* separate positive and negative sentences in a supervised latent space,
* obtain continuous class predictions,
* generate a confusion matrix,
* analyse model coefficients,
* compare variable importance with oMEDA.

The main MATLAB script is:

```text
src/matlab/plsda_toolbox.m
```

The results are stored in:

```text
results/pls-da/
```

Examples of generated CSV files include:

```text
data/attention/plsda_meda_importancia_layer.csv
data/attention/plsda_meda_importancia_tokens.csv
data/attention/plsda_meda_predicciones.csv
data/attention/plsda_meda_top50_variables_coeficientes.csv
data/attention/plsda_meda_validacion_cruzada.csv
data/attention/plsda_meda_varianza_capturada.csv
```

---

### ASCA

**ANOVA-Simultaneous Component Analysis** is used as an exploratory multivariate method based on experimental design.

In this project, ASCA is used to analyse whether the sentiment class factor produces an appreciable effect on the attention matrix.

The main factor considered is:

```text
Class = positive / negative
```

ASCA is used to study:

* the effect of sentiment class on attention maps,
* importance by layer,
* importance by attention head,
* importance by layer-head pair,
* importance by source token,
* importance by target token,
* importance by token pair,
* global token importance,
* comparison with PCA, oMEDA, SHAP and PLS-DA.

ASCA should not be interpreted as a classifier. Its purpose is not to predict whether a sentence is positive or negative, but to study whether the change in sentiment class introduces a systematic variation in the internal attention variables.

The main MATLAB script is:

```text
src/matlab/asca_meda_toolbox_attention_actualizado.m
```

The results are stored in:

```text
results/asca/
```

Examples of generated CSV files include:

```text
data/attention/asca_meda_importancia_head.csv
data/attention/asca_meda_importancia_layer.csv
data/attention/asca_meda_importancia_layer_head.csv
data/attention/asca_meda_importancia_tokens.csv
data/attention/asca_meda_top50_variables.csv
data/attention/asca_meda_variables_interpretadas.csv
```

---

## Repository structure

The current repository is organised as follows:

```text
.
├── data/
│   ├── attention/
│   │   ├── attention_matrix_40x9216.csv
│   │   ├── attention_tokens_40_sentences.csv
│   │   ├── attention_vector_9216.csv
│   │   ├── omeda_meda_*.csv
│   │   ├── plsda_meda_*.csv
│   │   ├── asca_meda_*.csv
│   │   └── global_top_tokens_tutor_simple.csv
│   │
│   ├── matlab/
│   │   └── MATLAB-compatible exported data files
│   │
│   └── sentences/
│       ├── positive_sentences.txt
│       └── negative_sentences.txt
│
├── docs/
│   ├── memoria/
│   ├── presentacion/
│   └── timeline/
│
├── results/
│   ├── asca/
│   ├── bert/
│   ├── comparison/
│   ├── frase_individual_analisis/
│   ├── omeda/
│   ├── pca/
│   ├── pls-da/
│   └── shap/
│
├── src/
│   ├── python/
│   │   ├── attention_matrix.py
│   │   ├── attention_matrix_export.py
│   │   ├── attention_vector.py
│   │   ├── shap_analysisespecialimportanciatokens.py
│   │   ├── shap_bert_analysis.py
│   │   └── si/
│   │
│   └── matlab/
│       ├── analisis_individualizado_tutor_simple.m
│       ├── asca_meda_toolbox_attention_actualizado.m
│       ├── omeda_toolbox.m
│       ├── pca_medatoolbox.m
│       ├── plsda_toolbox.m
│       ├── toolbox.zip
│       └── visualizacionbert.m
│
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/maciias04/xai-bert-attention-tfg.git
cd xai-bert-attention-tfg
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install torch transformers numpy pandas matplotlib shap scikit-learn scipy
```

A `requirements.txt` file can also be created with the following content:

```text
torch
transformers
numpy
pandas
matplotlib
shap
scikit-learn
scipy
```

Then install it using:

```bash
pip install -r requirements.txt
```

MATLAB and MEDA Toolbox are required separately for the multivariate analyses.

---

## Requirements

### Python

The Python part of the project requires:

* Python 3.x
* PyTorch
* Hugging Face Transformers
* NumPy
* Pandas
* Matplotlib
* SHAP
* scikit-learn
* SciPy

### MATLAB

The MATLAB part requires:

* MATLAB
* MEDA Toolbox
* scripts included in `src/matlab/`
* exported attention matrices from `data/attention/` or `data/matlab/`

The file `src/matlab/toolbox.zip` is included as part of the repository resources, but users should verify the correct installation and path configuration in MATLAB before running the analysis scripts.

---

## How to reproduce the analysis

### 1. Prepare the sentence files

Check the sentence files:

```text
data/sentences/positive_sentences.txt
data/sentences/negative_sentences.txt
```

These files contain the positive and negative sentences used in the experiment.

---

### 2. Extract attention maps with Python

Run the Python scripts from the repository root.

```bash
python src/python/attention_matrix.py
```

Depending on the exact script configuration, this step loads the BERT model, processes the input sentences and extracts the attention maps.

Then export the attention matrix:

```bash
python src/python/attention_matrix_export.py
```

Generate or inspect the vectorised attention representation:

```bash
python src/python/attention_vector.py
```

The expected core output is the attention matrix:

```text
data/attention/attention_matrix_40x9216.csv
```

---

### 3. Run SHAP analysis

Run:

```bash
python src/python/shap_bert_analysis.py
```

For the additional token importance analysis:

```bash
python src/python/shap_analysisespecialimportanciatokens.py
```

The SHAP outputs are stored in:

```text
results/shap/
```

---

### 4. Run PCA in MATLAB

Open MATLAB, set the repository root as the working directory and run:

```matlab
run('src/matlab/pca_medatoolbox.m')
```

Expected outputs include PCA score plots, explained variance plots and exported figures in:

```text
results/pca/
```

---

### 5. Run oMEDA in MATLAB

Run:

```matlab
run('src/matlab/omeda_toolbox.m')
```

Expected outputs include:

* oMEDA variable vector,
* top relevant attention variables,
* importance by layer,
* importance by token,
* figures and CSV summaries.

---

### 6. Run PLS-DA in MATLAB

Run:

```matlab
run('src/matlab/plsda_toolbox.m')
```

Expected outputs include:

* latent variable representation,
* continuous class predictions,
* confusion matrix,
* coefficient-based importance,
* importance by layer,
* importance by token.

---

### 7. Run ASCA in MATLAB

Run:

```matlab
run('src/matlab/asca_meda_toolbox_attention_actualizado.m')
```

Expected outputs include:

* class-effect scores,
* ASCA variable importance,
* importance by layer,
* importance by head,
* importance by layer-head pair,
* token-origin and token-destination importance,
* token-pair importance,
* global token ranking.

---

### 8. Run the individual phrase analysis

The individual analysis proposed during the project can be reproduced with:

```matlab
run('src/matlab/analisis_individualizado_tutor_simple.m')
```

This script follows a simplified tutor-style workflow for checking the individual and final global token-importance plots for oMEDA, PLS-DA and ASCA.

---

## Outputs and results

The main generated outputs are distributed across the following folders:

```text
results/bert/
```

BERT-related outputs, attention visualisations and model-level figures.

```text
results/pca/
```

PCA plots and explained variance results.

```text
results/omeda/
```

oMEDA importance plots and variable contribution figures.

```text
results/pls-da/
```

PLS-DA results, prediction plots, confusion matrix and coefficient-based importance.

```text
results/asca/
```

ASCA class-effect results and importance plots.

```text
results/shap/
```

SHAP explanations and token attribution outputs.

```text
results/comparison/
```

Joint comparison between oMEDA, PLS-DA, ASCA and SHAP.

```text
results/frase_individual_analisis/
```

Individual sentence analysis figures.

---

## Main findings

The main findings of the project can be summarised as follows.

### 1. Attention maps provide useful internal information

The attention maps allow visual inspection of how BERT distributes attention across tokens. They reveal that different layers and heads establish different token-token relationships.

However, attention maps should be interpreted carefully. They are useful for internal analysis, but they are not automatically equivalent to causal explanations.

---

### 2. PCA provides an exploratory view of the attention matrix

PCA makes it possible to project the `40 × 9216` matrix into a lower-dimensional space.

The PCA results suggest that the difference between positive and negative sentences is not necessarily captured by a simple two-dimensional linear separation. This does not mean that attention variables are irrelevant; rather, it suggests that the discriminant information is distributed across many internal variables.

---

### 3. oMEDA identifies relevant internal attention variables

oMEDA allows the analysis to move back from the latent PCA space to the original attention variables.

This makes it possible to identify which specific layer-head-token-token relationships contribute most strongly to the observed differences between the positive and negative sentence groups.

---

### 4. PLS-DA confirms discriminant information in attention variables

PLS-DA introduces the sentiment class labels during model fitting.

The results indicate that the attention matrix contains information related to the positive/negative distinction. This supports the idea that the internal attention structure of BERT changes depending on the sentiment polarity of the input sentence.

---

### 5. ASCA studies the effect of class over the full attention structure

ASCA provides a different perspective by analysing the effect of the experimental factor:

```text
Class = positive / negative
```

The method helps study whether the sentiment class produces a systematic effect on the attention variables. In this project, ASCA is especially relevant because it brings an experimental-design perspective into the analysis of BERT attention maps.

---

### 6. Final BERT layers tend to be more relevant

The analyses based on oMEDA, PLS-DA and ASCA tend to assign greater importance to the final layers of the model.

This is coherent with the interpretation that deeper layers in BERT contain more abstract and task-related representations, while earlier layers tend to capture more local or syntactic patterns.

---

### 7. SHAP highlights semantically meaningful sentiment tokens

SHAP tends to highlight real input tokens with a clearer sentiment meaning, such as:

```text
good, bad, awful, dreadful, painful, disliked, hated, dull, sharp, stale
```

This behaviour is expected because SHAP measures the contribution of input tokens to the final classifier output.

---

## Interpretation: SHAP vs multivariate methods

A central conclusion of this project is that SHAP and the multivariate attention-based methods should not be interpreted as equivalent.

### SHAP

SHAP works directly on the model input and estimates the contribution of real tokens to the final prediction.

It answers questions such as:

```text
Which words contributed most to the positive or negative classification?
```

This makes SHAP closer to a lexical explanation of the classifier output.

---

### oMEDA, PLS-DA and ASCA

oMEDA, PLS-DA and ASCA work on internal attention variables.

They answer questions such as:

```text
Which attention relationships are most relevant?
Which layers concentrate more importance?
Which heads show stronger class-related patterns?
Which token positions participate in relevant attention structures?
```

These methods do not directly explain the final prediction in terms of isolated words. Instead, they provide a structural interpretation of BERT’s internal attention behaviour.

---

### Why the difference matters

A token may be important for two different reasons:

1. It directly affects the final prediction.
2. It participates in an important internal attention pattern.

SHAP mainly captures the first case.
The multivariate methods mainly capture the second case.

Therefore, disagreement between SHAP and oMEDA, PLS-DA or ASCA is not necessarily a failure. It can indicate that different explanatory layers of the model are being analysed.

---

## Figures and expected visual outputs

The project generates several types of figures:

### BERT attention maps

Attention heatmaps showing token-token relationships for different layers and heads.

### PCA plots

* Explained variance by component.
* Accumulated explained variance.
* Score plots for PC1-PC2 and PC1-PC3.

### oMEDA plots

* oMEDA vector of attention variables.
* Distribution of variable importance.
* Importance by layer.
* Importance by layer-head pair.
* Importance by token position.
* Global token ranking.

### PLS-DA plots

* Latent-variable representation.
* Continuous class prediction.
* Confusion matrix.
* Variable importance based on coefficients.
* Importance by layer.
* Importance by layer-head pair.
* Global token ranking.

### ASCA plots

* Scores associated with class effect.
* ASCA variable importance.
* Importance by source token.
* Importance by target token.
* Importance by head.
* Importance by layer.
* Importance by layer-head pair.
* Token-pair importance.
* Global token ranking.

### SHAP plots

* Individual SHAP explanations.
* Global token importance ranking.

### Comparison plots

* Importance by layer comparison.
* Layer-head comparison between oMEDA, PLS-DA and ASCA.
* Global token comparison between SHAP and multivariate methods.

---

## Limitations

This project should be interpreted as an exploratory proof of concept. The main limitations are:

1. The dataset is small: only 40 sentences are used.
2. The sentences are controlled and do not represent the full complexity of natural language.
3. The model is not trained from scratch; the project analyses an already fine-tuned BERT model.
4. The study focuses on a single model: `textattack/bert-base-uncased-SST-2`.
5. The input length is fixed to 8 tokens to make the attention matrices homogeneous.
6. Attention weights should not be interpreted as direct causal explanations.
7. The projection from attention variables to real token importance requires caution.
8. The conclusions should not be generalised to all Transformer models.
9. PLS-DA results must be interpreted carefully because the number of variables is much larger than the number of observations.
10. ASCA is used as an exploratory effect-analysis tool, not as a classifier.

---

## Future work

Several future research directions are proposed.

### Larger datasets

The analysis could be extended to larger and more diverse sentiment datasets, including real SST-2 samples or other benchmark datasets.

### Other Transformer models

The same methodology could be applied to other language models, such as:

* RoBERTa,
* DistilBERT,
* DeBERTa,
* ALBERT,
* multilingual BERT,
* domain-specific BERT variants.

### More complex experimental designs

ASCA could be extended by including additional factors:

* sentence length,
* language,
* sentiment intensity,
* syntactic structure,
* presence of spelling errors,
* masking of specific tokens,
* controlled perturbations,
* negation,
* sarcasm,
* ambiguity.

### Original vs modified sentence pairs

Future work could compare original sentences with masked or perturbed versions, for example:

```text
Original: This movie was very good.
Modified: This movie was very [MASK].
```

This would make it possible to connect structural attention analysis with more direct token-level interventions.

### Comparison with gradient-based methods

The framework could be extended with:

* Integrated Gradients,
* Layer-wise Relevance Propagation,
* Gradient × Input,
* attention rollout,
* attention flow,
* LIME.

### Interactive visualisation tool

A future tool could integrate:

* model prediction,
* attention maps,
* PCA projections,
* oMEDA variables,
* PLS-DA coefficients,
* ASCA effects,
* SHAP explanations.

Such a tool could be useful for research, education and model debugging.

### Quantitative comparison of explanations

Future work could define metrics to compare explanation methods more systematically, such as:

* ranking correlation,
* overlap between top-k relevant tokens,
* overlap between top-k layers or heads,
* consistency across sentence pairs,
* stability under perturbation.

---

## Scientific and technical contribution

This repository contributes to the XAI field in several ways:

1. It provides a complete proof of concept for analysing BERT attention maps through multivariate methods.
2. It transforms attention maps into a structured numerical matrix suitable for statistical analysis.
3. It applies PCA, oMEDA, PLS-DA and ASCA to internal Transformer attention variables.
4. It compares structural attention-based explanations with SHAP token-level attributions.
5. It highlights the importance of distinguishing between lexical and structural interpretability.
6. It provides scripts, data and results that can be reused or extended in future work.
7. It proposes a methodological bridge between multivariate data analysis and explainable NLP.

---

## Citation

If you use this repository or build upon this work, please cite:

```bibtex
@thesis{macias2026xai,
  author = {Macias Caballero, Adrian},
  title = {Inteligencia Artificial Explicativa (XAI): Métodos y Prueba de Concepto},
  school = {Universidad de Granada},
  year = {2026},
  type = {Trabajo Fin de Grado}
}
```

---

## License

This repository is released under the MIT License.

See:

```text
LICENSE
```

for details.

---

## Acknowledgements

This project was developed as part of a Final Degree Project at the Universidad de Granada.

Special thanks to the academic supervisor, José Camacho Páez, for his guidance, support and methodological orientation throughout the development of the work.

---

## Academic disclaimer

This repository is associated with an academic proof of concept. The results should be interpreted as exploratory and should not be understood as a definitive or universal characterisation of BERT, Transformer interpretability or sentiment classification.

The analyses presented here are intended to support research and learning in Explainable Artificial Intelligence, multivariate analysis and natural language processing.
