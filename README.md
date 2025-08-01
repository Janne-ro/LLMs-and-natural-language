# LLMs and Natural Language

This project explores modern techniques in speech and language processing using Python and state-of-the-art deep learning models. It is organized into three main notebooks:

- [Speech Feature Extraction.ipynb](LLMs-and-natural-language/Speech%20Feature%20Extraction.ipynb): Fundamentals of extracting features from raw audio.
- [Speech Recognition - CTC.ipynb](LLMs-and-natural-language/Speech%20Recognition%20-%20CTC.ipynb): End-to-end speech recognition using wav2vec and Connectionist Temporal Classification (CTC).
- [LLMs and Classifiers.ipynb](LLMs-and-natural-language/LLMs%20and%20Classifiers.ipynb): Large Language Models (T5) for text embeddings and sentiment classification.

Note that the notebooks are originally intended to be run in google colab. 


## Table of Contents

- [Overview](#overview)
- [1. Speech Feature Extraction](#1-speech-feature-extraction)
- [2. End-to-End Speech Recognition with CTC](#2-end-to-end-speech-recognition-with-ctc)
- [3. LLMs and Sentiment Classification](#3-llms-and-sentiment-classification)
- [Setup and Installation](#setup-and-installation)
- [References](#references)


## Overview

This repository demonstrates the pipeline from raw speech audio to text and sentiment understanding, using both classic and modern deep learning approaches. The notebooks are designed to be run independently, but together they provide a comprehensive introduction to:

- Audio feature extraction (waveforms, spectrograms, mel-spectrograms, MFCCs)
- End-to-end automatic spech recognition (ASR) with wav2vec and CTC
- Text representation and sentiment analysis with transformer-based LLMs (T5)
- Classic ML classifiers and fine-tuning of transformer models

## 1. Speech Feature Extraction

Notebook: [Speech Feature Extraction.ipynb](LLMs-and-natural-language/Speech%20Feature%20Extraction.ipynb)

- Loads and visualizes speech audio from the Google Speech Commands dataset.
- Demonstrates extraction of:
  - Raw waveforms
  - Spectrograms (STFT)
  - Mel-spectrograms
  - MFCCs (Mel Frequency Cepstral Coefficients)
- Discusses the trade-offs between different audio representations for downstream tasks.


## 2. End-to-End Speech Recognition with CTC

Notebook: [Speech Recognition - CTC.ipynb](LLMs-and-natural-language/Speech%20Recognition%20-%20CTC.ipynb)

- Introduces Connectionist Temporal Classification (CTC) for aligning variable-length audio to text.
- Uses HuggingFace's wav2vec2 model for ASR on the LibriSpeech dataset.
- Steps include:
  1. Downloading and exploring LibriSpeech test data.
  2. Loading a pretrained wav2vec2 model and tokenizer.
  3. Transcribing audio and evaluating with Word Error Rate (WER).
  4. Testing robustness by adding noise at different SNR levels.
  5. Experimenting with your own voice and cross-lingual models (e.g., German).


## 3. LLMs and Sentiment Classification

Notebook: [LLMs and Classifiers.ipynb](LLMs-and-natural-language/LLMs%20and%20Classifiers.ipynb)

- Explores text embeddings and sentiment classification using the T5 transformer model.
- Covers:
  - Extracting and visualizing T5 embeddings for words and sentences.
  - Training classic ML classifiers (KNN, MLP, SVM) on T5 embeddings.
  - Evaluating the effect of PCA dimensionality reduction.
  - Using T5 for zero-shot and fine-tuned sentiment classification on the IMDB dataset.
  - Comparing classic ML and transformer-based approaches.

---

## Setup and Installation

To run these notebooks, you will need Python (>=3.7), Jupyter Notebook, and the following libraries:

```bash
pip install torch torchaudio librosa matplotlib tqdm scikit-learn transformers datasets pytorch-lightning jiwer pydub
```

Some notebooks may require additional system packages (e.g., `libsndfile1` for audio).

---

## References

- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [LibriSpeech Dataset](https://www.openslr.org/12)
- [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209)
- [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

---

Each notebook is self-contained and includes detailed explanations and code comments. 