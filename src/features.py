import numpy as np
from collections import Counter
from src.preprocessing import preprocess_pipeline

class CustomCountVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.feature_names_ = []

    def fit(self, raw_documents):
        """
        Builds a vocabulary of up to max_features most frequent words from raw_documents.
        """
        word_counts = Counter()
        for doc in raw_documents:
            # If document is a raw string, preprocess it
            if isinstance(doc, str):
                tokens = doc.split()
            else:
                tokens = list(doc)
            word_counts.update(tokens)

        # Get top max_features words
        most_common = word_counts.most_common(self.max_features)
        
        # Build vocabulary dictionary mapping word -> index
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.feature_names_ = [word for word, _ in most_common]
        return self

    def transform(self, raw_documents):
        """
        Transforms raw_documents into a term-document count matrix.
        Returns a dense NumPy array.
        """
        num_docs = len(raw_documents)
        vocab_size = len(self.vocabulary_)
        X = np.zeros((num_docs, vocab_size), dtype=np.float32)

        for idx, doc in enumerate(raw_documents):
            if isinstance(doc, str):
                tokens = doc.split()
            else:
                tokens = list(doc)
            
            # Count words for this document
            doc_counts = Counter(tokens)
            for word, count in doc_counts.items():
                if word in self.vocabulary_:
                    X[idx, self.vocabulary_[word]] = count
        return X

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)


class CustomTfidfVectorizer:
    def __init__(self, max_features=1000, norm='l2'):
        self.max_features = max_features
        self.norm = norm
        self.count_vectorizer = CustomCountVectorizer(max_features=max_features)
        self.idf_ = None

    def fit(self, raw_documents):
        """
        Fits CountVectorizer and computes IDF weights.
        """
        # Fit count vectorizer and transform to term counts
        X_counts = self.count_vectorizer.fit(raw_documents).transform(raw_documents)
        n_samples = X_counts.shape[0]

        # Document frequency (DF): number of documents containing each word
        # We sum the boolean mask (X_counts > 0) along columns
        df = np.sum(X_counts > 0, axis=0)

        # Compute IDF using scikit-learn standard formula (with smoothing)
        # idf(t) = log((1 + n_samples) / (1 + df(t))) + 1
        self.idf_ = np.log((1 + n_samples) / (1 + df)) + 1
        
        # Save vocabulary references
        self.vocabulary_ = self.count_vectorizer.vocabulary_
        self.feature_names_ = self.count_vectorizer.feature_names_
        return self

    def transform(self, raw_documents):
        """
        Transforms raw_documents into TF-IDF representation.
        """
        X_counts = self.count_vectorizer.transform(raw_documents)
        
        # TF-IDF = TF * IDF (using broadcasting)
        X_tfidf = X_counts * self.idf_

        # Apply normalization if specified
        if self.norm == 'l2':
            # Row-wise L2 norm: sqrt(sum(val^2))
            norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0.0] = 1.0
            X_tfidf = X_tfidf / norms

        return X_tfidf

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
