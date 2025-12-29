---
title: Igala Dataset Explorer
emoji: 📊
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.31.0"
app_file: app.py
pinned: false
---

# Igala Dataset Explorer

The **Igala Dataset Explorer** is an interactive web application designed to support
exploration, analysis, and preparation of text datasets for **low-resource languages**,
with an initial focus on the Igala language spoken in Nigeria.

The tool is intended for researchers, developers, and linguists working on
natural language processing (NLP) tasks where data scarcity, dataset quality,
and preprocessing are key challenges.

---

## Motivation

Many African and low-resource languages lack accessible tooling for corpus inspection
and dataset preparation. This makes it difficult to:
- Understand dataset structure and quality
- Perform reproducible preprocessing
- Prepare data for downstream NLP tasks

This project aims to lower that barrier by providing a reusable, language-agnostic
dataset exploration interface that works for Igala and can easily be extended
to other languages.

---

## Core Features

- **Dataset upload support** (CSV format)
- **Dataset preview and filtering**
- **Sentence length statistics**
- **Token-level statistics**
  - Average tokens per sentence
  - Minimum and maximum token length
  - Token length distribution
- **Character-level statistics**
  - Average characters per sentence
  - Average characters per token
- **Word frequency analysis**
- **Bigram extraction**
- **Dataset metadata annotation**
  - Language
  - Dialect
  - Source
- **Download filtered datasets** for reuse

---

## Research Use Cases

This tool can be used for:

- Exploratory data analysis of low-resource language corpora
- Dataset quality assessment before model training
- Preprocessing and filtering text for NLP pipelines
- Linguistic analysis of sentence and token structure
- Preparing datasets for downstream tasks such as:
  - Language modeling
  - Text classification
  - Machine translation

---

## Design Principles

- **Language-agnostic**: Works with any text dataset, not limited to Igala
- **Reproducible**: Encourages consistent preprocessing and analysis
- **Lightweight**: Minimal dependencies, Hugging Face–friendly deployment
- **Research-oriented**: Focused on dataset understanding rather than model training

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

---

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
