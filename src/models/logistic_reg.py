import numpy as np
from src.models.base import BaseModel

class LogisticRegressionClassifier(BaseModel):
    def __init__(self, lr=0.1, epochs=100, lambda_reg=0.01, batch_size=None, verbose=False):
        """
        Logistic Regression Classifier from scratch with L2 regularization.
        lr: learning rate
        epochs: number of training iterations
        lambda_reg: L2 regularization penalty parameter
        batch_size: mini-batch training (if None, standard batch gradient descent is used)
        """
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        # Clip z to avoid numerical overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Trains the model using Gradient Descent.
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Xavier-style initialization of weights
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []
        
        for epoch in range(self.epochs):
            if self.batch_size is not None and self.batch_size < n_samples:
                # Shuffle the dataset for mini-batch GD
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                epoch_loss = 0.0
                num_batches = int(np.ceil(n_samples / self.batch_size))
                
                for b in range(num_batches):
                    start_idx = b * self.batch_size
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    m_batch = X_batch.shape[0]
                    
                    # Forward pass
                    z = np.dot(X_batch, self.weights) + self.bias
                    a = self._sigmoid(z)
                    
                    # Compute batch loss
                    # Add small epsilon to avoid log(0)
                    eps = 1e-15
                    loss = -np.mean(y_batch * np.log(a + eps) + (1 - y_batch) * np.log(1 - a + eps))
                    reg_loss = (self.lambda_reg / (2 * m_batch)) * np.sum(self.weights ** 2)
                    epoch_loss += (loss + reg_loss) * (m_batch / n_samples)
                    
                    # Backpropagation (Gradients)
                    dz = a - y_batch
                    dw = (1 / m_batch) * np.dot(X_batch.T, dz) + (self.lambda_reg / m_batch) * self.weights
                    db = (1 / m_batch) * np.sum(dz)
                    
                    # Update weights
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db
                
                self.loss_history.append(epoch_loss)
            else:
                # Standard Batch Gradient Descent
                # Forward pass
                z = np.dot(X, self.weights) + self.bias
                a = self._sigmoid(z)
                
                # Compute Loss
                eps = 1e-15
                loss = -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))
                reg_loss = (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
                total_loss = loss + reg_loss
                self.loss_history.append(total_loss)
                
                # Backpropagation
                dz = a - y
                dw = (1 / n_samples) * np.dot(X.T, dz) + (self.lambda_reg / n_samples) * self.weights
                db = (1 / n_samples) * np.sum(dz)
                
                # Update weights
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d}/{self.epochs:4d} - Loss: {self.loss_history[-1]:.6f}")
                
        return self

    def predict_proba(self, X):
        """
        Predict probability estimates for samples in X.
        """
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z).flatten()

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1) for samples in X.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
