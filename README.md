# Smart MCQ Solver Challenge

### BSDA2001P – Introduction to Deep Learning and Generative AI Project

## Student Information

* Name: Kunal
* Roll Number: 23F2001870
* Course: BSDA2001P – Introduction to Deep Learning and Generative AI

---

## Project Title

Smart MCQ Solver using Transformer-based Answer Ranking and Large Language Models

---

## Problem Statement

The objective of this project is to build an intelligent Multiple Choice Question Answering (MCQA) system capable of predicting the top three most probable answers for a given question.

Each question consists of:

* A question prompt
* Five answer options labeled:

  * A
  * B
  * C
  * D
  * E

The model must rank the three most likely answers in descending order of confidence.

Example:

| Question ID | Prediction |
| ----------- | ---------- |
| 1           | A B C      |
| 2           | C A D      |
| 3           | B D A      |

---

## Evaluation Metric

Submissions are evaluated using Mean Average Precision at 3 (MAP@3).

The metric rewards models that:

1. Predict the correct answer.
2. Rank the correct answer as high as possible.

Examples:

Correct Answer: A

| Prediction | Relative Score |
| ---------- | -------------- |
| A B C      | Highest        |
| B A C      | Lower          |
| C D A      | Lowest         |

---

## Objectives

* Understand and preprocess MCQ datasets
* Build transformer-based answer ranking systems
* Experiment with retrieval-augmented and prompt-based approaches
* Fine-tune language models for MCQ reasoning
* Evaluate models using MAP@3
* Develop efficient inference pipelines

---

## Repository Structure

smart-mcq-solver-map3/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── milestone-1.ipynb
│   ├── milestone-2.ipynb
│   ├── milestone-3.ipynb
│   └── final_notebook.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│
├── reports/
│   ├── milestone-1-report.pdf
│   ├── milestone-2-report.pdf
│   ├── milestone-3-report.pdf
│   └── final-report.pdf
│
├── scripts/
│
├── submissions/
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Proposed Methodology

Phase 1:

* Dataset exploration
* Exploratory Data Analysis
* Text preprocessing

Phase 2:

* Baseline TF-IDF + Classical ML models
* Transformer embeddings

Phase 3:

* Fine-tuning transformer models
* Answer ranking and probability estimation

Phase 4:

* Model ensembling
* MAP@3 optimization
* Final inference pipeline

---

## Branching Strategy

Main Branch:

* Latest stable code
* Final notebooks
* Training scripts
* Reports
* Documentation

Milestone Branches:

* milestone-1
* milestone-2
* milestone-3

All development will be performed inside milestone branches and merged into main after completion.

Milestone branches will be preserved for evaluation and progress tracking.

---

## Expected Deliverables

* Reproducible training pipeline
* Trained models
* Inference scripts
* Submission generation pipeline
* Project reports
* Documentation
