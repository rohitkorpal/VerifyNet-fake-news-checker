import os
import time
import argparse
import pandas as pd
import numpy as np

from src.preprocessing import preprocess_pipeline
from src.features import CustomTfidfVectorizer
from src.evaluation import train_test_split_scratch, classification_report_scratch
from src.models.knn import KNNClassifier
from src.models.logistic_reg import LogisticRegressionClassifier
from src.models.random_forest import RandomForestClassifier
from src.models.neural_net import SimpleNeuralNetwork

def load_data(sample_size=2000):
    """
    Loads True.csv and Fake.csv, labels them, merges them, and samples them.
    """
    print(f"Loading data...")
    true_path = os.path.join("dataset", "True.csv")
    fake_path = os.path.join("dataset", "Fake.csv")
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError("True.csv and Fake.csv must be in the 'dataset' directory.")
        
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    
    print(f"  Found {len(df_true)} Real and {len(df_fake)} Fake articles.")
    
    # Assign labels: 0 = Real, 1 = Fake
    df_true['label'] = 0
    df_fake['label'] = 1
    
    # Merge datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)
    
    # Sample if sample_size is smaller than total size
    if sample_size and sample_size < len(df):
        print(f"  Sampling {sample_size} articles (balanced split)...")
        # To keep it balanced:
        half_sample = sample_size // 2
        df_true_sampled = df_true.sample(half_sample, random_state=42)
        df_fake_sampled = df_fake.sample(half_sample, random_state=42)
        df_sampled = pd.concat([df_true_sampled, df_fake_sampled], ignore_index=True)
        # Shuffle the sampled dataset
        df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        return df_sampled
        
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Fake News Classification Pipeline from Scratch")
    parser.add_argument("--samples", type=int, default=2000, help="Number of total samples to use (default: 2000 for speed)")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size for TF-IDF (default: 1000)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train-test split ratio (default: 0.2)")
    
    args = parser.parse_args()
    
    print("=================================================================")
    print("   AI-POWERED FAKE NEWS CLASSIFICATION PIPELINE (FROM SCRATCH)   ")
    print("=================================================================\n")
    
    # 1. Load Data
    try:
        df = load_data(sample_size=args.samples)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    # We will use 'text' column for classification. Let's handle titles as well if needed, but text has more context.
    texts = df['text'].fillna("").values
    labels = df['label'].values
    
    # 2. Preprocess Data
    print("\nPreprocessing texts (cleaning, tokenization, stopword removal)...")
    start_time = time.time()
    clean_texts = [preprocess_pipeline(text) for text in texts]
    print(f"  Preprocessed {len(clean_texts)} texts in {time.time() - start_time:.2f} seconds.")
    
    # 3. Feature Extraction
    print(f"\nExtracting features using Custom TF-IDF Vectorizer (max_features={args.vocab_size})...")
    start_time = time.time()
    vectorizer = CustomTfidfVectorizer(max_features=args.vocab_size)
    X = vectorizer.fit_transform(clean_texts)
    print(f"  Extracted feature matrix of shape {X.shape} in {time.time() - start_time:.2f} seconds.")
    
    # 4. Train-Test Split
    print(f"\nSplitting dataset (test_size={args.test_split})...")
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, labels, test_size=args.test_split, random_state=42)
    print(f"  Training set size: {X_train.shape[0]}")
    print(f"  Testing set size : {X_test.shape[0]}")
    
    # 5. Initialize Models
    models = {
        "K-Nearest Neighbors (k=5, cosine)": KNNClassifier(k=5, metric='cosine'),
        "Logistic Regression (lr=0.1, epochs=150)": LogisticRegressionClassifier(lr=0.1, epochs=150, lambda_reg=0.01, verbose=False),
        "Random Forest (n=10, depth=8)": RandomForestClassifier(n_estimators=10, max_depth=8),
        "Simple Neural Network (hidden=64, epochs=100)": SimpleNeuralNetwork(hidden_dim=64, lr=0.05, epochs=100, batch_size=32)
    }
    
    # 6. Train and Evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  Trained in {train_time:.2f} seconds.")
        
        print("  Evaluating model...")
        preds = model.predict(X_test)
        
        # Calculate accuracy manually
        acc = np.mean(preds == y_test)
        results[name] = {
            'model': model,
            'preds': preds,
            'acc': acc,
            'train_time': train_time
        }
        print(f"  Test Accuracy: {acc:.4f}")
        
    # 7. Print Final Comparison Report
    print("\n========================================================")
    print("                FINAL COMPARISON REPORT                 ")
    print("========================================================")
    for name, res in results.items():
        print(f"\nModel: {name}")
        print(f"Training Time: {res['train_time']:.2f}s | Test Accuracy: {res['acc']:.4%}")
        report = classification_report_scratch(y_test, res['preds'])
        print(report)
        print("-" * 56)
        
    # 8. Interactive Testing
    print("\n========================================================")
    print("                INTERACTIVE TEST MODE                   ")
    print("========================================================")
    print("Type a news headline or short article below to get predictions from all models.")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nEnter news text: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.strip().lower() == 'exit':
            break
            
        if not user_input.strip():
            continue
            
        # Clean user input
        clean_input = preprocess_pipeline(user_input)
        # Transform using fitted TF-IDF
        x_input = vectorizer.transform([clean_input])
        
        print("\nPredictions:")
        for name, res in results.items():
            pred = res['model'].predict(x_input)[0]
            pred_label = "FAKE NEWS" if pred == 1 else "REAL NEWS"
            print(f"  {name:45s} -> {pred_label}")

if __name__ == '__main__':
    main()
