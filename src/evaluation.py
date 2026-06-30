import numpy as np

def train_test_split_scratch(X, y, test_size=0.2, random_state=None):
    """
    Splits datasets into random train and test subsets.
    Implemented from scratch using NumPy.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(X)
    shuffled_indices = np.random.permutation(n_samples)
    
    test_set_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # Check if inputs are NumPy arrays or standard lists/DataFrames
    if hasattr(X, 'iloc'):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    elif isinstance(X, np.ndarray):
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        
    if hasattr(y, 'iloc'):
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    elif isinstance(y, np.ndarray):
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        
    return X_train, X_test, y_train, y_test


def confusion_matrix_scratch(y_true, y_pred):
    """
    Computes confusion matrix from scratch.
    Returns: TP, FP, TN, FN and visual list of lists [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Binary classification assumed: 1 = Fake/Positive, 0 = Real/Negative (or vice-versa)
    # Let's assume standard binary labels: 1 and 0.
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    matrix = [
        [TN, FP],
        [FN, TP]
    ]
    return TP, FP, TN, FN, matrix


def accuracy_score_scratch(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision_score_scratch(y_true, y_pred):
    TP, FP, TN, FN, _ = confusion_matrix_scratch(y_true, y_pred)
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall_score_scratch(y_true, y_pred):
    TP, FP, TN, FN, _ = confusion_matrix_scratch(y_true, y_pred)
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def f1_score_scratch(y_true, y_pred):
    precision = precision_score_scratch(y_true, y_pred)
    recall = recall_score_scratch(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def classification_report_scratch(y_true, y_pred):
    """
    Generates a text report showing classification metrics.
    """
    accuracy = accuracy_score_scratch(y_true, y_pred)
    precision = precision_score_scratch(y_true, y_pred)
    recall = recall_score_scratch(y_true, y_pred)
    f1 = f1_score_scratch(y_true, y_pred)
    TP, FP, TN, FN, _ = confusion_matrix_scratch(y_true, y_pred)
    
    report = (
        f"Classification Metrics Report:\n"
        f"-------------------------------\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1-Score : {f1:.4f}\n\n"
        f"Confusion Matrix Details:\n"
        f"-----------------------\n"
        f"True Positives (TP) : {TP}\n"
        f"False Positives (FP): {FP}\n"
        f"True Negatives (TN) : {TN}\n"
        f"False Negatives (FN): {FN}\n"
    )
    return report
