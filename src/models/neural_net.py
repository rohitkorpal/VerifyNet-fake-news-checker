import numpy as np
from src.models.base import BaseModel

class SimpleNeuralNetwork(BaseModel):
    def __init__(self, hidden_dim=64, lr=0.01, epochs=100, batch_size=64, verbose=False):
        """
        A simple Multi-Layer Perceptron (MLP) classifier from scratch.
        Architecture: Input -> Hidden (ReLU) -> Output (Sigmoid)
        hidden_dim: number of hidden units in the hidden layer
        lr: learning rate
        epochs: number of epochs to train
        batch_size: batch size for mini-batch gradient descent
        """
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Weights and Biases
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self.loss_history = []

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Trains the network using mini-batch gradient descent and backpropagation.
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Xavier (He) Initialization
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(n_features, self.hidden_dim) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, self.hidden_dim))
        
        self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, 1))
        
        self.loss_history = []
        
        for epoch in range(self.epochs):
            # Shuffle for mini-batch
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
                
                # --- FORWARD PASS ---
                # Hidden Layer
                Z1 = np.dot(X_batch, self.W1) + self.b1
                A1 = self._relu(Z1)
                
                # Output Layer
                Z2 = np.dot(A1, self.W2) + self.b2
                A2 = self._sigmoid(Z2)
                
                # Compute Loss (Binary Cross Entropy)
                eps = 1e-15
                loss = -np.mean(y_batch * np.log(A2 + eps) + (1 - y_batch) * np.log(1 - A2 + eps))
                epoch_loss += loss * (m_batch / n_samples)
                
                # --- BACKWARD PASS ---
                # Output Layer gradients
                dZ2 = A2 - y_batch
                dW2 = (1.0 / m_batch) * np.dot(A1.T, dZ2)
                db2 = (1.0 / m_batch) * np.sum(dZ2, axis=0, keepdims=True)
                
                # Hidden Layer gradients
                dZ1 = np.dot(dZ2, self.W2.T) * self._relu_derivative(Z1)
                dW1 = (1.0 / m_batch) * np.dot(X_batch.T, dZ1)
                db1 = (1.0 / m_batch) * np.sum(dZ1, axis=0, keepdims=True)
                
                # --- PARAMETER UPDATES ---
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                
            self.loss_history.append(epoch_loss)
            
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d}/{self.epochs:4d} - Loss: {epoch_loss:.6f}")
                
        return self

    def predict_proba(self, X):
        """
        Returns predictions as probabilities.
        """
        X = np.array(X)
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        return A2.flatten()

    def predict(self, X, threshold=0.5):
        """
        Returns class predictions (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def partial_fit(self, X, y, lr=None):
        """
        Performs a single incremental training step (online learning) on a new batch/sample.
        """
        if self.W1 is None:
            # Initialize if not already fitted
            n_features = X.shape[1]
            self.W1 = np.random.randn(n_features, self.hidden_dim) * np.sqrt(2.0 / n_features)
            self.b1 = np.zeros((1, self.hidden_dim))
            self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
            self.b2 = np.zeros((1, 1))
            
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        step_lr = lr if lr is not None else self.lr
        
        # --- FORWARD PASS ---
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)
        
        # --- BACKPROPAGATION ---
        m = X.shape[0]
        dZ2 = A2 - y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- PARAMETER UPDATES ---
        self.W1 -= step_lr * dW1
        self.b1 -= step_lr * db1
        self.W2 -= step_lr * dW2
        self.b2 -= step_lr * db2
        return self
