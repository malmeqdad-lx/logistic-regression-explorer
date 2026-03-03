# 📘 Multinomial Logistic Regression Explorer

An interactive Streamlit tool for visualizing and experimenting with logistic regression for NLP intent classification — a hands-on companion to **Jurafsky & Martin, *Speech and Language Processing*, Chapter 4** (Sections 4.1–4.8).

All algorithms implemented **from scratch in NumPy** — no sklearn, no pytorch.

## What You Can Do

| Tab | Covers | J&M Sections |
|-----|--------|-------------|
| **① Sigmoid & Intuition** | Interactive sigmoid explorer, binary classification worked example from Fig. 4.2 | §4.1–4.3 |
| **② Setup Intents** | Define K intent classes with training samples (defaults to 5 Vanguard intents) | §4.1 |
| **③ Feature Matrix** | Bag-of-words feature extraction, per-class vocabulary analysis | §4.1 |
| **④ Softmax Forward Pass** | Step-by-step z = Wx + b → softmax → cross-entropy loss with full computation trace | §4.7–4.8 |
| **⑤ Train (SGD)** | Run SGD with gradient anatomy: see the error vector (ŷ − y), per-weight updates, loss curve, W heatmap, L2 regularization | §4.6–4.8, 4.10 |
| **⑥ Predict & Evaluate** | Classify new sentences, confusion matrix, per-class Precision/Recall/F1, macro vs micro F1 | §4.7.2, 4.9 |

## Key Algorithms (from scratch)

- **Bag-of-words** feature extraction
- **Sigmoid**: σ(z) = 1/(1+exp(−z)) with numerical stability
- **Softmax**: exp(z)/Σexp(z) with max-shift stability
- **Cross-entropy loss**: −log P(true class | x)
- **SGD update**: W_k ← W_k − η(ŷ_k − y_k)x, b_k ← b_k − η(ŷ_k − y_k)
- **L2 regularization**: loss + λ‖W‖²

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dependencies

- streamlit >= 1.54.0
- numpy
- plotly

## Architecture

- `app.py` — Single-file Streamlit app (all logic from scratch in NumPy)
- `.streamlit/config.toml` — Server config (port 5000, headless)
- `requirements.txt` — Python dependencies

## Reference

Jurafsky, D. & Martin, J.H. (2024). *Speech and Language Processing* (3rd ed. draft), Chapter 4: Logistic Regression and Text Classification.
