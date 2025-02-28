# Skip-gram Model Implementation in Python

This repository contains a **Skip-gram model** implemented **from scratch** using **NumPy** for numerical computations and **PyMuPDF** for extracting text from a PDF. The Skip-gram model is a neural network-based approach to learn **word embeddings**, where a **target word** is used to predict its **context words**.

---

## 📌 Overview

The **Skip-gram model** learns word representations by predicting **context words** given a **single target word**. This project follows these steps:

1. **Text Extraction:** Reads text from `corpus.pdf` using **PyMuPDF**.
2. **Preprocessing:** Tokenizes the text, removes punctuation and stopwords, and converts words to lowercase.
3. **Skip-gram Pair Generation:** Creates (target, context) word pairs using a specified window size.
4. **One-Hot Encoding:** Converts words into numerical vectors using one-hot encoding.
5. **Training Data Preparation:** Represents each word as a one-hot vector for input to a **two-layer neural network**.
6. **Neural Network Training:** Trains the model using **gradient descent** and **cross-entropy loss**.
7. **Prediction:** Given a target word, predicts **likely context words**.

---

## 🛠️ Requirements

- Python 3.x
- [PyMuPDF](https://pypi.org/project/PyMuPDF/) (for PDF text extraction)
- [NumPy](https://numpy.org/) (for numerical operations)

### 📥 Installation

To install the required libraries, run:

```bash
pip install PyMuPDF numpy
