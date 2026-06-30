import numpy as np
from src.preprocessing import preprocess_pipeline
from src.features import CustomTfidfVectorizer
from src.evaluation import train_test_split_scratch, classification_report_scratch
from src.models.knn import KNNClassifier
from src.models.logistic_reg import LogisticRegressionClassifier
from src.models.random_forest import RandomForestClassifier
from src.models.neural_net import SimpleNeuralNetwork

def main():
    print("=== Starting Fake News Pipeline Verification ===")
    
    # 1. Create a toy dataset
    raw_texts = [
        "breaking news Donald Trump sends out an embarrassing and shocking tweet today",
        "Donald Trump bragged about Russian ties in a secret staffer meeting",
        "Hillary Clinton emails revealed shocking secrets about conspiracy theories",
        "Senate Republicans pass a massive tax reform bill in Congress today",
        "President Obama signed the new healthcare legislation into law",
        "The Supreme Court ruled on the controversial voting rights case today",
        "mass vote fraud uncovered in elections across the country, source says",
        "aliens land on the White House lawn to meet the President, breaking fake report"
    ]
    # Labels: 1 = Fake, 0 = Real
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    
    print("\n1. Original Texts:")
    for text, label in zip(raw_texts, labels):
        print(f"[{'Fake' if label == 1 else 'Real'}] - {text}")
        
    # 2. Text Preprocessing
    print("\n2. Running Preprocessing Pipeline...")
    clean_texts = [preprocess_pipeline(text) for text in raw_texts]
    for orig, clean in zip(raw_texts, clean_texts):
        print(f"  Orig: '{orig}'\n  Clean: '{clean}'\n")
        
    # 3. Feature Extraction (TF-IDF Vectorizer from scratch)
    print("3. Transforming text to TF-IDF features...")
    vectorizer = CustomTfidfVectorizer(max_features=15)
    X = vectorizer.fit_transform(clean_texts)
    print(f"  Feature dimensions: {X.shape}")
    print(f"  Vocabulary: {vectorizer.vocabulary_}")
    print("  Sample TF-IDF vector for document 0:")
    print(X[0])
    
    # 4. Train-Test Split (Scratch)
    print("\n4. Performing Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, labels, test_size=0.25, random_state=42)
    print(f"  Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # 5. Fit & Predict using all models
    models = {
        "KNN (k=3, cosine)": KNNClassifier(k=3, metric='cosine'),
        "Logistic Regression": LogisticRegressionClassifier(lr=0.5, epochs=50, lambda_reg=0.01, verbose=False),
        "Random Forest (n=5)": RandomForestClassifier(n_estimators=5, max_depth=4),
        "Simple Neural Network": SimpleNeuralNetwork(hidden_dim=8, lr=0.2, epochs=50, batch_size=4)
    }
    
    for name, model in models.items():
        print(f"\n--- Testing {name} ---")
        print(f"Fitting model...")
        model.fit(X_train, y_train)
        
        print("Predicting on training set...")
        train_preds = model.predict(X_train)
        print(f"  Train Accuracy: {np.mean(train_preds == y_train):.4f}")
        
        print("Predicting on test set...")
        test_preds = model.predict(X_test)
        print(f"  Test Predictions: {test_preds}")
        print(f"  Actual Labels:    {y_test}")
        
        # Calculate and print metrics
        report = classification_report_scratch(y_test, test_preds)
        print(report)

    print("=== Pipeline Verification Completed Successfully ===")

if __name__ == '__main__':
    main()
