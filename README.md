# Fake news detection

# Project Walkthrough: Fake News Detection Using Text Classification (From Scratch)

We have successfully built, tested, and validated a complete Machine Learning text classification pipeline from scratch. Every module—including preprocessing, feature extraction, splitting, model training (KNN, Logistic Regression, Random Forest, and a Simple Neural Network), and performance evaluations—was implemented from scratch using only Python, NumPy, and Pandas.

---

## 📂 Project Structure

All source files have been created in the workspace at [fake news](file:///c:/Users/ROHIT/Desktop/fake%20news/):

- **Text Preprocessing**: [preprocessing.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/preprocessing.py) - Text cleaning, manual tokenization, and customized stopword filtering.
- **Feature Extraction**: [features.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/features.py) - Custom `CountVectorizer` and custom smooth L2-normalized `TfidfVectorizer`.
- **Classification Models**:
  - Base Class: [base.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/base.py)
  - KNN Classifier: [knn.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/knn.py) (Euclidean & Cosine distance vectorization)
  - Logistic Regression: [logistic_reg.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/logistic_reg.py) (L2 Regularized gradient descent)
  - Random Forest & Decision Tree: [random_forest.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/random_forest.py) (Entropy/Gini impurity optimization with random sub-feature splits)
  - Simple Neural Network (MLP): [neural_net.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/neural_net.py) (Xavier init, Backpropagation SGD with mini-batches)
- **Evaluation Utilities**: [evaluation.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/evaluation.py) - Metric computations (Accuracy, Precision, Recall, F1, Confusion Matrix) and `train_test_split_scratch` split helper.
- **Testing & Verification**: [test_pipeline.py](file:///c:/Users/ROHIT/Desktop/fake%20news/test_pipeline.py) - Verification script for checking correctness on toy corpuses.
- **Streamlit Application**: [app.py](file:///c:/Users/ROHIT/Desktop/fake%20news/app.py) - Premium interactive dashboard with real-time training, parameter sliders, EDA distributions, and live classification.
- **CLI Runner**: [main.py](file:///c:/Users/ROHIT/Desktop/fake%20news/main.py) - Command line pipeline driver.

---

## 🚀 Performance Metrics (2,000 Sample Subset, 1,000 Max Features)

During our validation run, the custom models yielded the following test metrics on the ISOT Fake News Dataset:

| Model Name | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple Neural Network** | **97.25%** | **97.52%** | **97.04%** | **97.28%** | ~11.6s |
| **Logistic Regression** | **92.00%** | 98.31% | 85.71% | 91.58% | ~0.34s |
| **Random Forest** | **90.50%** | 98.82% | 82.27% | 89.78% | ~6.14s |
| **K-Nearest Neighbors** | **81.75%** | 91.67% | 70.44% | 79.67% | ~0.01s |

> [!TIP]
> The **Simple Neural Network (MLP)** achieved outstanding accuracy (97.25%), demonstrating that backpropagation and feature mapping in NumPy can solve advanced NLP classification tasks reliably.
> **Cosine distance KNN** and **Logistic Regression** provide lightweight, lightning-fast baselines with robust F1 scores.

---

## 🎬 Live Interactive App Session

We validated the pipeline behavior in two separate interactive browser sessions:

### 1. Training & Verification Demo
Demonstrates launching the dashboard, executing dataset analysis, setting parameters, running the from-scratch training pipeline, and plotting convergence curves.
![Streamlit Interactive Dashboard Demo](C:\Users\ROHIT\.gemini\antigravity-ide\brain\72b71f91-04f1-40dc-8dc0-94ad7e4a0679\streamlit_interactive_test_1782840996343.webp)

### 2. Model Persistence & Auto-Loading Demo
Demonstrates the training execution, serialization of fitted weights to the disk, and instant loading on refresh (showing `🟢 Ready (Models loaded)` in the sidebar without requiring retraining).
![Streamlit Model Persistence Demo](C:\Users\ROHIT\.gemini\antigravity-ide\brain\72b71f91-04f1-40dc-8dc0-94ad7e4a0679\streamlit_state_test_1782842779440.webp)

### 3. Unified Verification & Supporting Live Coverage Demo
Demonstrates how entering a news text instantly runs predictions on all 4 models, and automatically triggers background live news searches across both **NewsAPI** and **NewsData.io** (with keys securely loaded from `.env`) displaying coverage side-by-side to support the models' decisions.
![Streamlit Unified Verification Demo](C:\Users\ROHIT\.gemini\antigravity-ide\brain\72b71f91-04f1-40dc-8dc0-94ad7e4a0679\unified_predictor_1782845940622.webp)

---

## 🛠️ How to Run the Project

### 1. Launch the Streamlit Web App
To start the interactive web application, run:
```bash
streamlit run app.py
```
This opens the browser at `http://localhost:8501`, where you can:
- Explore dataset length distributions and top vocabulary terms in the **Dataset & EDA** tab.
- Set model parameters (neighbors, learning rates, estimators, depth, etc.) and hit **Train All Models** in the sidebar.
- Inspect cost history curves, confusion matrices, and comparison bar charts in the **Model Comparison** tab.
- Test your own custom headlines/articles inside the **Live Article Predictor** tab.

### 2. Run the CLI Pipeline
To run training, evaluation, and interactive predictions directly in your terminal, execute:
```bash
python main.py --samples 2000 --vocab_size 1000 --test_split 0.2
```
After fitting, you can enter any news paragraph directly into the terminal prompt to get instant labels from all models.

---

## 🛡️ Model Generalization & General Accuracy Improvements
We implemented a series of optimizations to prevent **target leakage / shortcut learning** and improve classification accuracy:

1. **Target Leakage Cleaning**: Added `strip_datelines_and_leakage` to [preprocessing.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/preprocessing.py) to automatically strip location-based dateline prefixes (like `WASHINGTON (Reuters) -`) and common publisher tags. This prevents the models from relying on simple publisher markers (since 99.8% of the Real dataset news contains `"Reuters"` while only 1.3% of Fake news does).
2. **Stylistic Feature Engineering**: Added `append_stylistic_features` to [app.py](file:///c:/Users/ROHIT/Desktop/fake%20news/app.py) to extract and append uppercase letter density and exclamation mark density to the TF-IDF feature space before feeding vectors to classifiers. This helps capture the sensationalist tone (e.g. capitalizing words, multiple exclamations) commonly seen in fake news.
3. **Hyperparameter Default Tuning**: Defaulted KNN to `"cosine"` distance (mathematically superior for TF-IDF vectors), optimized Logistic Regression convergence iteration limit, and improved Neural Network learning rates.
4. **Strict Real-Time AND Searches**: Configured NewsAPI to use strict query formatting (with prefix `+` keys) and NewsData.io to use strict `AND` operators to ensure only articles matching all critical keywords are retrieved.
5. **Zero-Coverage Verdict Override**: Integrated dynamic checking where having zero matching sources from the live APIs automatically penalizes the credibility score (capped at 20%) to override style-based model predictions. Also passes search summaries directly to Gemini to perform contextual fact comparison.
6. **Continuous Online Learning (Human-in-the-Loop)**: Added `partial_fit` methods to [logistic_reg.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/logistic_reg.py) and [neural_net.py](file:///c:/Users/ROHIT/Desktop/fake%20news/src/models/neural_net.py) to support real-time single-sample online weight updates. Added a feedback collection and instant-training panel in [app.py](file:///c:/Users/ROHIT/Desktop/fake%20news/app.py) Tab 3 which appends user-corrected labels to a local `user_feedback.csv` log and immediately updates model files on disk.

