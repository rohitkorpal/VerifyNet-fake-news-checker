import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import pickle
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import preprocess_pipeline
from src.features import CustomTfidfVectorizer
from src.evaluation import train_test_split_scratch, confusion_matrix_scratch, classification_report_scratch, accuracy_score_scratch, precision_score_scratch, recall_score_scratch, f1_score_scratch
from src.models.knn import KNNClassifier
from src.models.logistic_reg import LogisticRegressionClassifier
from src.models.random_forest import RandomForestClassifier
from src.models.neural_net import SimpleNeuralNetwork

# Setup page layout
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Dark Mode aesthetic customization */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Cards and boxes styling */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .model-card-real {
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0px;
    }
    .model-card-fake {
        background: linear-gradient(135deg, #7F1D1D 0%, #450A0A 100%);
        border: 1px solid #DC2626;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0px;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for data loading and model training
@st.cache_data(show_spinner=False)
def get_dataset_statistics():
    """
    Loads raw CSV files to get exact lengths and descriptions without caching the whole split.
    """
    if not os.path.exists("True.csv") or not os.path.exists("Fake.csv"):
        return 0, 0
    df_true = pd.read_csv("True.csv")
    df_fake = pd.read_csv("Fake.csv")
    return len(df_true), len(df_fake)

@st.cache_data(show_spinner=False)
def load_and_preprocess_subset(sample_size, vocab_size):
    """
    Loads, samples, and preprocesses a balanced subset of news.
    """
    df_true = pd.read_csv("True.csv")
    df_fake = pd.read_csv("Fake.csv")
    
    half_sample = sample_size // 2
    
    # Balanced sample
    df_t = df_true.sample(min(half_sample, len(df_true)), random_state=42)
    df_f = df_fake.sample(min(half_sample, len(df_fake)), random_state=42)
    
    df_t['label'] = 0
    df_f['label'] = 1
    
    df_all = pd.concat([df_t, df_f], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Preprocess
    texts = df_all['text'].fillna("").values
    labels = df_all['label'].values
    
    clean_texts = [preprocess_pipeline(t) for t in texts]
    
    # Custom Vectorizer
    vectorizer = CustomTfidfVectorizer(max_features=vocab_size)
    X = vectorizer.fit_transform(clean_texts)
    
    return X, labels, vectorizer, clean_texts, df_all


SAVED_MODELS_DIR = "saved_models"

def save_models_to_disk(vectorizer, knn, log_reg, rf, nn, metrics):
    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)
    
    with open(os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(SAVED_MODELS_DIR, "knn.pkl"), "wb") as f:
        pickle.dump(knn, f)
    with open(os.path.join(SAVED_MODELS_DIR, "log_reg.pkl"), "wb") as f:
        pickle.dump(log_reg, f)
    with open(os.path.join(SAVED_MODELS_DIR, "rf.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(SAVED_MODELS_DIR, "nn.pkl"), "wb") as f:
        pickle.dump(nn, f)
    with open(os.path.join(SAVED_MODELS_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

def load_models_from_disk():
    required_files = ["vectorizer.pkl", "knn.pkl", "log_reg.pkl", "rf.pkl", "nn.pkl", "metrics.pkl"]
    for fname in required_files:
        if not os.path.exists(os.path.join(SAVED_MODELS_DIR, fname)):
            return None
            
    try:
        with open(os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "knn.pkl"), "rb") as f:
            knn = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "log_reg.pkl"), "rb") as f:
            log_reg = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "rf.pkl"), "rb") as f:
            rf = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "nn.pkl"), "rb") as f:
            nn = pickle.load(f)
        with open(os.path.join(SAVED_MODELS_DIR, "metrics.pkl"), "rb") as f:
            metrics = pickle.load(f)
            
        return vectorizer, knn, log_reg, rf, nn, metrics
    except Exception as e:
        return None

def fetch_live_news(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&pageSize=5"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            return None
    except Exception as e:
        return None

def fetch_newsdata_io(query, api_key):
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={query}&language=en"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("results", [])[:5]
        else:
            return None
    except Exception as e:
        return None

def load_env_api_key(key_name):
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    parts = line.strip().split("=", 1)
                    if len(parts) == 2 and parts[0].strip() == key_name:
                        return parts[1].strip()
        except Exception:
            pass
    return ""

def extract_query_with_gemini(text):
    """Use Gemini API to extract the best search keywords from a news article/headline."""
    gemini_key = load_env_api_key("GEMINI_API_KEY")
    if not gemini_key or not text.strip():
        return _fallback_query(text)
    
    prompt = (
        "Extract a specific news search query from the article below.\n\n"
        "Rules:\n"
        "- Return 4 to 6 keywords that describe the SPECIFIC event or claim in the article.\n"
        "- NEVER return just a person's name alone (e.g. 'Trump' or 'Biden'). Always include WHAT happened.\n"
        "- Include key subjects, actions, and objects (e.g. 'Trump Mars space colony plan').\n"
        "- Remove source names (Reuters, AP), datelines, and filler words.\n"
        "- Return ONLY the search query. No quotes, no explanation, no numbering.\n\n"
        "Examples:\n"
        "Article: 'Trump announces plan to build colony on Mars' → Trump Mars colony plan\n"
        "Article: 'Pope Francis endorses Donald Trump for President' → Pope Francis endorses Trump President\n"
        "Article: 'India launches new space mission to study the Sun' → India space mission Sun study\n\n"
        f"Article:\n{text[:600]}"
    )
    
    # Try multiple Gemini models in case one has quota
    models = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 50}
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                query = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                query = query.strip('"').strip("'").strip()
                # Reject responses with fewer than 3 words — too generic for meaningful search
                if query and len(query.split()) >= 3:
                    return query
        except Exception:
            continue
    
    return _fallback_query(text)

def gemini_fact_check(text):
    """Ask Gemini to analyze a news article and determine if it is likely real or fake with reasoning."""
    gemini_key = load_env_api_key("GEMINI_API_KEY")
    if not gemini_key or not text.strip():
        return None
    
    prompt = (
        "You are an expert fact-checker and journalist. Analyze the following news article and determine "
        "whether it is likely REAL or FAKE news.\n\n"
        "Evaluate based on:\n"
        "1. Writing style (sensationalist language, clickbait, emotional manipulation)\n"
        "2. Source credibility clues (anonymous sources, vague attribution)\n"
        "3. Factual plausibility (are the claims realistic and verifiable?)\n"
        "4. Known misinformation patterns (too good/bad to be true, conspiracy theories)\n\n"
        "Respond in EXACTLY this format (3 lines only):\n"
        "VERDICT: FAKE or REAL\n"
        "CONFIDENCE: a number from 1 to 100\n"
        "REASONING: One paragraph explaining your analysis (2-4 sentences max)\n\n"
        f"Article:\n{text[:800]}"
    )
    
    models = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200}
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Parse the structured response
                result = {"verdict": "UNKNOWN", "confidence": 50, "reasoning": "Unable to analyze."}
                
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.upper().startswith("VERDICT:"):
                        v = line.split(":", 1)[1].strip().upper()
                        result["verdict"] = "FAKE" if "FAKE" in v else "REAL"
                    elif line.upper().startswith("CONFIDENCE:"):
                        try:
                            result["confidence"] = int(''.join(filter(str.isdigit, line.split(":", 1)[1].strip()))[:3])
                        except (ValueError, IndexError):
                            pass
                    elif line.upper().startswith("REASONING:"):
                        result["reasoning"] = line.split(":", 1)[1].strip()
                
                return result
        except Exception:
            continue
    
    return None

def _fallback_query(text):
    """Smart fallback: strip news prefixes and extract meaningful keywords without AI."""
    if not text.strip():
        return "politics"
    
    # Take first line / first sentence (handle abbreviations like U.S., Dr.)
    first_line = text.strip().split('\n')[0].strip()
    # Only split by period if it looks like a real sentence boundary (followed by space + uppercase)
    import re
    sentences = re.split(r'(?<!\b[A-Z])\.(?=\s+[A-Z])', first_line)
    first_sentence = sentences[0].strip() if sentences else first_line
    
    # Strip common dateline prefixes like "WASHINGTON (Reuters) - " 
    if ' - ' in first_sentence:
        parts = first_sentence.split(' - ', 1)
        # Check if the part before dash looks like a dateline (e.g. WASHINGTON, LONDON (AP), NEW YORK (Reuters))
        before_dash = parts[0].strip()
        # Remove parenthesized source like (Reuters), (AP)
        import re
        dateline_part = re.sub(r'\([^)]*\)', '', before_dash).strip()
        if dateline_part and dateline_part.replace(' ', '').isupper():
            first_sentence = parts[1]
    
    # Strip "BREAKING:" type prefixes
    prefixes_to_remove = ['BREAKING:', 'BREAKING NEWS:', 'UPDATE:', 'FLASH:', 'EXCLUSIVE:', 'OPINION:']
    upper_sentence = first_sentence.upper()
    for prefix in prefixes_to_remove:
        if upper_sentence.startswith(prefix):
            first_sentence = first_sentence[len(prefix):].strip()
            break
    
    # Remove common stopwords and take first 5 meaningful words
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'in', 'on', 'at', 
            'to', 'for', 'of', 'by', 'with', 'from', 'and', 'or', 'but', 'that', 'this', 'it',
            'its', 'his', 'her', 'their', 'our', 'your', 'today', 'said', 'says', 'say'}
    words = first_sentence.split()
    meaningful = [w.strip('.,!?:;\'"()') for w in words if w.lower().strip('.,!?:;\'"()') not in stop and len(w) > 1]
    
    if meaningful:
        return " ".join(meaningful[:5])
    return "politics"

# State Initialization (Must run before sidebar is drawn)
if "trained" not in st.session_state:
    loaded = load_models_from_disk()
    if loaded is not None:
        vectorizer_disk, knn_disk, log_reg_disk, rf_disk, nn_disk, metrics_disk = loaded
        st.session_state.vectorizer = vectorizer_disk
        st.session_state.knn = knn_disk
        st.session_state.log_reg = log_reg_disk
        st.session_state.rf = rf_disk
        st.session_state.nn = nn_disk
        st.session_state.metrics = metrics_disk
        st.session_state.trained = True
    else:
        st.session_state.trained = False

# Page Header
st.markdown("<h1 class='main-title'>📰 AI-Powered Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>A Complete Machine Learning Pipeline Implemented From Scratch</p>", unsafe_allow_html=True)

# Check for files
if not os.path.exists("True.csv") or not os.path.exists("Fake.csv"):
    st.error("⚠️ Dataset files (True.csv and Fake.csv) not found in the current folder. Please verify the environment.")
    st.stop()

# Sidebar: Hyperparameters and Settings
st.sidebar.markdown("## ⚙️ Configuration Settings")

sample_size = st.sidebar.slider("Total Samples (Balanced)", 200, 5000, 2000, step=200, 
                                help="Number of real & fake news articles to use. KNN & RF splits can be slow on larger sample counts.")
vocab_size = st.sidebar.slider("Vocabulary Size (Max Features)", 100, 2500, 1000, step=100,
                               help="Maximum distinct words in the TF-IDF feature space.")
test_split = st.sidebar.slider("Test Split Ratio", 0.1, 0.5, 0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Model Hyperparameters")

# KNN Params
with st.sidebar.expander("K-Nearest Neighbors Parameters"):
    k_neighbors = st.slider("Neighbors (k)", 1, 15, 5, step=2)
    knn_metric = st.selectbox("Distance Metric", ["cosine", "euclidean"])

# Logistic Reg Params
with st.sidebar.expander("Logistic Regression Parameters"):
    log_lr = st.slider("Learning Rate (LR)", 0.01, 1.0, 0.1, step=0.05)
    log_epochs = st.slider("Training Epochs ", 10, 500, 150, step=10)
    log_lambda = st.slider("L2 Penalty (Lambda)", 0.0, 0.5, 0.01, step=0.01)

# Random Forest Params
with st.sidebar.expander("Random Forest Parameters"):
    rf_trees = st.slider("Estimators (Trees)", 2, 30, 10, step=2)
    rf_depth = st.slider("Max Tree Depth", 2, 15, 8, step=1)

# Simple Neural Net Params
with st.sidebar.expander("Simple Neural Net Parameters"):
    nn_hidden = st.slider("Hidden Layer Dimension", 8, 128, 64, step=8)
    nn_lr = st.slider("Learning Rate (NN)", 0.001, 0.5, 0.05, step=0.005)
    nn_epochs = st.slider("NN Epochs", 10, 300, 100, step=10)
    nn_batch = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Pre-trained Models Status")
if st.session_state.trained:
    st.sidebar.success("🟢 Ready (Models loaded)")
else:
    st.sidebar.warning("⚠️ Retraining required (No models loaded)")
st.sidebar.markdown("---")
train_trigger = st.sidebar.button("🚀 Train All Models", use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dataset & EDA", "⚡ Model Comparison", "🔮 Live Article Predictor"])

# State Initialization completed above.

# TAB 1: DATASET & EDA
with tab1:
    st.markdown("### 📊 Dataset Overview & Statistics")
    
    n_real, n_fake = get_dataset_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:#10B981;'>{n_real:,}</div>
            <div class='metric-label'>Real News Articles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:#EF4444;'>{n_fake:,}</div>
            <div class='metric-label'>Fake News Articles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:#6366F1;'>{n_real + n_fake:,}</div>
            <div class='metric-label'>Total ISOT Dataset Size</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Exploratory Word Count Analysis")
    
    # Load a small snippet for EDA preview to keep it fast
    X_eda, y_eda, vectorizer_eda, clean_texts_eda, df_eda = load_and_preprocess_subset(1000, 500)
    
    # Compute text word counts
    df_eda['word_count'] = df_eda['text'].apply(lambda x: len(str(x).split()))
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Text Length Distribution (Word Count)")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        
        # Plot distributions
        sns.histplot(data=df_eda, x='word_count', hue='label', kde=True, bins=30, palette={0: '#10B981', 1: '#EF4444'}, ax=ax, alpha=0.6)
        
        # Aesthetic updates
        ax.set_xlabel("Word Count", color='#F8FAFC')
        ax.set_ylabel("Count", color='#F8FAFC')
        ax.set_title("Real (Green) vs Fake (Red) News length", color='#F8FAFC')
        ax.tick_params(colors='#F8FAFC')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        legend = ax.get_legend()
        if legend:
            legend.get_texts()[0].set_text('Real')
            legend.get_texts()[1].set_text('Fake')
            legend.get_frame().set_facecolor('#1E293B')
            legend.get_frame().set_edgecolor('#334155')
            for text in legend.get_texts():
                text.set_color('#F8FAFC')
                
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("#### Top 10 Most Common Words in Vocabulary")
        # Sum columns of word counts (from Count matrix equivalent)
        word_frequencies = np.sum(X_eda, axis=0)
        vocab = vectorizer_eda.feature_names_
        
        freq_df = pd.DataFrame({'word': vocab, 'tf_idf_weight': word_frequencies}).sort_values(by='tf_idf_weight', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        
        sns.barplot(x='tf_idf_weight', y='word', data=freq_df, palette="viridis", ax=ax)
        
        ax.set_xlabel("Cumulative TF-IDF Score", color='#F8FAFC')
        ax.set_ylabel("Words", color='#F8FAFC')
        ax.set_title("Top 10 High-Weight Words in Corpus", color='#F8FAFC')
        ax.tick_params(colors='#F8FAFC')
        for spine in ax.spines.values():
            spine.set_color('#334155')
            
        st.pyplot(fig)

    st.markdown("""
    > [!NOTE]
    > Notice how **Fake News** and **Real News** display different word length distributions. 
    > Fake news articles are often shorter or show higher variances in formatting. The TF-IDF weights help capture the exact distinguishing vocabularies.
    """)

# TAB 2: MODEL COMPARISON & TRAINING
with tab2:
    if train_trigger or st.session_state.trained:
        if train_trigger:
            with st.spinner("⏳ Loading dataset, cleaning text, and building TF-IDF vectors from scratch..."):
                X, y, vectorizer, clean_texts, df_sampled = load_and_preprocess_subset(sample_size, vocab_size)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=test_split, random_state=42)
                
                # KNN
                knn = KNNClassifier(k=k_neighbors, metric=knn_metric)
                # Logistic Reg
                log_reg = LogisticRegressionClassifier(lr=log_lr, epochs=log_epochs, lambda_reg=log_lambda, verbose=False)
                # RF
                rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth)
                # Neural Net
                nn = SimpleNeuralNetwork(hidden_dim=nn_hidden, lr=nn_lr, epochs=nn_epochs, batch_size=nn_batch, verbose=False)
                
                # Training operations & time tracking
                # 1. KNN
                t0 = time.time()
                knn.fit(X_train, y_train)
                t_knn = time.time() - t0
                
                # 2. Logistic Reg
                t0 = time.time()
                log_reg.fit(X_train, y_train)
                t_log = time.time() - t0
                
                # 3. Random Forest
                t0 = time.time()
                rf.fit(X_train, y_train)
                t_rf = time.time() - t0
                
                # 4. Neural Net
                t0 = time.time()
                nn.fit(X_train, y_train)
                t_nn = time.time() - t0
                
                # Test evaluations
                preds_knn = knn.predict(X_test)
                preds_log = log_reg.predict(X_test)
                preds_rf = rf.predict(X_test)
                preds_nn = nn.predict(X_test)
                
                # Compute metrics
                metrics = {}
                for name, preds, t_train, model in [
                    ("KNN", preds_knn, t_knn, knn),
                    ("Logistic Regression", preds_log, t_log, log_reg),
                    ("Random Forest", preds_rf, t_rf, rf),
                    ("Simple Neural Network", preds_nn, t_nn, nn)
                ]:
                    acc = accuracy_score_scratch(y_test, preds)
                    prec = precision_score_scratch(y_test, preds)
                    rec = recall_score_scratch(y_test, preds)
                    f1 = f1_score_scratch(y_test, preds)
                    tp, fp, tn, fn, mat = confusion_matrix_scratch(y_test, preds)
                    
                    metrics[name] = {
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'train_time': t_train,
                        'confusion_matrix': mat,
                        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                    }
                
                # Save to session state
                st.session_state.vectorizer = vectorizer
                st.session_state.knn = knn
                st.session_state.log_reg = log_reg
                st.session_state.rf = rf
                st.session_state.nn = nn
                st.session_state.metrics = metrics
                st.session_state.trained = True
                
                # Save to disk for persistence across restarts
                save_models_to_disk(vectorizer, knn, log_reg, rf, nn, metrics)
                
        metrics = st.session_state.metrics
        
        st.success("✅ Models trained and evaluated successfully!")
        
        # Display Metrics in beautiful format
        st.markdown("### 🏆 Algorithm Performance Comparison")
        
        compare_data = {
            "Model Name": list(metrics.keys()),
            "Accuracy": [metrics[m]['accuracy'] for m in metrics],
            "Precision": [metrics[m]['precision'] for m in metrics],
            "Recall": [metrics[m]['recall'] for m in metrics],
            "F1-Score": [metrics[m]['f1'] for m in metrics],
            "Train Time (s)": [metrics[m]['train_time'] for m in metrics]
        }
        
        compare_df = pd.DataFrame(compare_data)
        st.dataframe(
            compare_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Train Time (s)': '{:.3f}s'
            }), 
            use_container_width=True
        )
        
        # Plot accuracy and time side-by-side
        fig_comp, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(15, 6))
        fig_comp.patch.set_facecolor('#0F172A')
        ax_acc.set_facecolor('#1E293B')
        ax_time.set_facecolor('#1E293B')
        
        # Accuracy comparison
        sns.barplot(x="Accuracy", y="Model Name", data=compare_df, palette="coolwarm", ax=ax_acc)
        ax_acc.set_xlim(0, 1.05)
        ax_acc.set_title("Test Accuracy Comparison", color="#F8FAFC", fontsize=12)
        ax_acc.set_xlabel("Accuracy", color="#F8FAFC")
        ax_acc.set_ylabel("", color="#F8FAFC")
        ax_acc.tick_params(colors="#F8FAFC")
        for spine in ax_acc.spines.values():
            spine.set_color('#334155')
            
        # Training time comparison
        sns.barplot(x="Train Time (s)", y="Model Name", data=compare_df, palette="viridis", ax=ax_time)
        ax_time.set_title("Training Time Comparison (Seconds)", color="#F8FAFC", fontsize=12)
        ax_time.set_xlabel("Seconds", color="#F8FAFC")
        ax_time.set_ylabel("", color="#F8FAFC")
        ax_time.tick_params(colors="#F8FAFC")
        for spine in ax_time.spines.values():
            spine.set_color('#334155')
            
        st.pyplot(fig_comp)
        
        # Learning Curves Tab
        st.markdown("### 📈 Optimization Loss Curves")
        col_curve1, col_curve2 = st.columns(2)
        
        with col_curve1:
            st.markdown("#### Logistic Regression Cost History")
            losses_log = st.session_state.log_reg.loss_history
            fig_loss1, ax_l1 = plt.subplots(figsize=(8, 4))
            fig_loss1.patch.set_facecolor('#0F172A')
            ax_l1.set_facecolor('#1E293B')
            ax_l1.plot(losses_log, color='#38BDF8', linewidth=2)
            ax_l1.set_xlabel("Iteration / Epoch", color="#F8FAFC")
            ax_l1.set_ylabel("BCE Cost", color="#F8FAFC")
            ax_l1.set_title("Gradient Descent Convergence", color="#F8FAFC")
            ax_l1.tick_params(colors="#F8FAFC")
            ax_l1.grid(True, color="#334155", linestyle="--")
            for spine in ax_l1.spines.values():
                spine.set_color('#334155')
            st.pyplot(fig_loss1)
            
        with col_curve2:
            st.markdown("#### Neural Network Loss History")
            losses_nn = st.session_state.nn.loss_history
            fig_loss2, ax_l2 = plt.subplots(figsize=(8, 4))
            fig_loss2.patch.set_facecolor('#0F172A')
            ax_l2.set_facecolor('#1E293B')
            ax_l2.plot(losses_nn, color='#EC4899', linewidth=2)
            ax_l2.set_xlabel("Epoch", color="#F8FAFC")
            ax_l2.set_ylabel("BCE Loss", color="#F8FAFC")
            ax_l2.set_title("Backpropagation SGD Convergence", color="#F8FAFC")
            ax_l2.tick_params(colors="#F8FAFC")
            ax_l2.grid(True, color="#334155", linestyle="--")
            for spine in ax_l2.spines.values():
                spine.set_color('#334155')
            st.pyplot(fig_loss2)

        # Confusion Matrices Section
        st.markdown("### 🧮 Confusion Matrices")
        col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
        cms = [
            ("KNN", col_cm1),
            ("Logistic Regression", col_cm2),
            ("Random Forest", col_cm3),
            ("Simple Neural Network", col_cm4)
        ]
        
        for name, col in cms:
            with col:
                st.markdown(f"##### {name}")
                m = metrics[name]
                cm_data = np.array(m['confusion_matrix'])
                
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                fig_cm.patch.set_facecolor('#0F172A')
                ax_cm.set_facecolor('#1E293B')
                
                sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax_cm,
                            annot_kws={"size": 14, "weight": "bold"})
                
                ax_cm.set_xlabel("Predicted", color="#F8FAFC", fontsize=10)
                ax_cm.set_ylabel("Actual", color="#F8FAFC", fontsize=10)
                ax_cm.tick_params(colors="#F8FAFC")
                for spine in ax_cm.spines.values():
                    spine.set_color('#334155')
                    
                st.pyplot(fig_cm)
                
    else:
        st.info("👈 Please select your preferred configurations and click **🚀 Train All Models** in the sidebar to start!")

# TAB 3: LIVE PREDICTOR
with tab3:
    st.markdown("### 🔮 Verify News & Search Live Coverage")
    st.markdown("Paste a news headline or article body below. The system will classify the text using our trained models and automatically search both **NewsAPI** and **NewsData.io** to retrieve matching live news reports to support the model's decision.")

    # Load example buttons
    example_text = ""
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("📰 Load Real News Example", use_container_width=True):
            example_text = (
                "WASHINGTON (Reuters) - The U.S. Senate approved a sweeping tax reform bill early Saturday morning, "
                "marking a major legislative victory for President Donald Trump. The Republican-led chamber passed the bill "
                "51-49, following a marathon late-night session. The bill represents the largest overhaul of the U.S. tax code "
                "since the 1980s, cutting rates for corporations and individuals."
            )
    with col_ex2:
        if st.button("🚨 Load Fake News Example", use_container_width=True):
            example_text = (
                "BREAKING: Pope Francis has shocked the world today by endorsing Donald Trump for President. "
                "In a statement released by the Vatican, the Pope declared that Donald Trump is the only logical choice "
                "to lead the free world, praising his stances on border control and economic expansion. The statement has "
                "ignited a massive controversy across global religious organizations."
            )

    news_input = st.text_area("News text to verify:", value=example_text, height=180, 
                             placeholder="Enter headline or article paragraph here...")

    # Load API keys silently from .env
    news_api_key = load_env_api_key("NEWS_API_KEY")
    newsdata_api_key = load_env_api_key("NEWSDATA_API_KEY")

    verify_trigger = st.button("🔍 Verify Article & Search Live Coverage", use_container_width=True)

    if verify_trigger:
        if not st.session_state.trained:
            st.error("⚠️ Please train the models first using Tab 2 or the sidebar button!")
        elif not news_input.strip():
            st.warning("⚠️ Please enter some news text to verify.")
        else:
            # 1. Classification Phase
            with st.spinner("Analyzing text and running custom model predictions..."):
                clean_input = preprocess_pipeline(news_input)
                x_input = st.session_state.vectorizer.transform([clean_input])

                p_knn = st.session_state.knn.predict(x_input)[0]
                p_log = st.session_state.log_reg.predict(x_input)[0]
                p_rf = st.session_state.rf.predict(x_input)[0]
                p_nn = st.session_state.nn.predict(x_input)[0]

                prob_log = st.session_state.log_reg.predict_proba(x_input)[0]
                prob_nn = st.session_state.nn.predict_proba(x_input)[0]

                st.markdown("### 🏆 Trained Models Decisions")
                col_pred1, col_pred2 = st.columns(2)

                with col_pred1:
                    cls_knn = "model-card-fake" if p_knn == 1 else "model-card-real"
                    lbl_knn = "🚨 FAKE NEWS" if p_knn == 1 else "🟢 REAL NEWS"
                    st.markdown(f"""
                    <div class='{cls_knn}'>
                        <div class='card-title'>K-Nearest Neighbors</div>
                        <h2 style='margin:0;'>{lbl_knn}</h2>
                        <div style='font-size:0.85rem; opacity:0.85;'>Based on k={k_neighbors} closest articles</div>
                    </div>
                    """, unsafe_allow_html=True)

                    cls_rf = "model-card-fake" if p_rf == 1 else "model-card-real"
                    lbl_rf = "🚨 FAKE NEWS" if p_rf == 1 else "🟢 REAL NEWS"
                    st.markdown(f"""
                    <div class='{cls_rf}'>
                        <div class='card-title'>Random Forest</div>
                        <h2 style='margin:0;'>{lbl_rf}</h2>
                        <div style='font-size:0.85rem; opacity:0.85;'>Aggregated vote from {rf_trees} decision trees</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_pred2:
                    cls_log = "model-card-fake" if p_log == 1 else "model-card-real"
                    lbl_log = "🚨 FAKE NEWS" if p_log == 1 else "🟢 REAL NEWS"
                    pct_log = prob_log if p_log == 1 else (1.0 - prob_log)
                    st.markdown(f"""
                    <div class='{cls_log}'>
                        <div class='card-title'>Logistic Regression</div>
                        <h2 style='margin:0;'>{lbl_log}</h2>
                        <div style='font-size:0.85rem; opacity:0.85;'>Confidence probability: {pct_log:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    cls_nn = "model-card-fake" if p_nn == 1 else "model-card-real"
                    lbl_nn = "🚨 FAKE NEWS" if p_nn == 1 else "🟢 REAL NEWS"
                    pct_nn = prob_nn if p_nn == 1 else (1.0 - prob_nn)
                    st.markdown(f"""
                    <div class='{cls_nn}'>
                        <div class='card-title'>Simple Neural Network</div>
                        <h2 style='margin:0;'>{lbl_nn}</h2>
                        <div style='font-size:0.85rem; opacity:0.85;'>Confidence probability: {pct_nn:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # 2. Gemini Keyword Extraction Phase
            st.markdown("---")
            with st.spinner("🤖 Gemini AI is extracting search keywords from your input..."):
                search_query = extract_query_with_gemini(news_input)
            
            st.markdown("### 📡 Live Coverage & Supporting Evidence")
            st.success(f"🔑 **Gemini Extracted Keywords:** `{search_query}`")
            st.markdown(f"Searching both news APIs for live coverage matching: **\"{search_query}\"**...")

            col_newsapi, col_newsdata = st.columns(2)
            
            newsapi_count = 0
            newsdata_count = 0

            # Fetch NewsAPI Results
            with col_newsapi:
                st.markdown("#### 📰 NewsAPI Coverage")
                if not news_api_key:
                    st.info("⚠️ NewsAPI key is not configured in `.env` file.")
                else:
                    with st.spinner("Searching NewsAPI..."):
                        raw_api = fetch_live_news(search_query, news_api_key)
                        if raw_api is None:
                            st.error("❌ Failed to query NewsAPI.")
                        elif len(raw_api) == 0:
                            st.warning("🔍 No matching coverage found on NewsAPI.")
                        else:
                            newsapi_count = len(raw_api)
                            for idx, art in enumerate(raw_api):
                                title = art.get('title', 'No Title')
                                source = art.get('source', {}).get('name', 'Unknown Source')
                                url = art.get('url', '#')
                                desc = art.get('description', '')
                                date = art.get('publishedAt', '')[:10]

                                st.markdown(f"##### {idx+1}. [{source}] {title}")
                                st.markdown(f"Published: {date} | [Read full article]({url})")
                                if desc:
                                    st.write(desc)
                                st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

            # Fetch NewsData.io Results
            with col_newsdata:
                st.markdown("#### 📡 NewsData.io Coverage")
                if not newsdata_api_key:
                    st.info("⚠️ NewsData.io key is not configured in `.env` file.")
                else:
                    with st.spinner("Searching NewsData.io..."):
                        raw_data = fetch_newsdata_io(search_query, newsdata_api_key)
                        if raw_data is None:
                            st.error("❌ Failed to query NewsData.io.")
                        elif len(raw_data) == 0:
                            st.warning("🔍 No matching coverage found on NewsData.io.")
                        else:
                            newsdata_count = len(raw_data)
                            for idx, art in enumerate(raw_data):
                                title = art.get('title', 'No Title')
                                source = art.get('source_id', 'Unknown Source').upper()
                                url = art.get('link', '#')
                                desc = art.get('description', '')
                                date = art.get('pubDate', '')[:10] if art.get('pubDate') else ''

                                pub_str = f"Published: {date} | " if date else ""
                                st.markdown(f"##### {idx+1}. [{source}] {title}")
                                st.markdown(f"{pub_str}[Read full article]({url})")
                                if desc:
                                    st.write(desc)
                                st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

            # ============================================================
            # 3. UNIFIED AI VERDICT — Combining all signals
            # ============================================================
            st.markdown("---")
            st.markdown("### 🧠 Unified AI Verdict")
            
            with st.spinner("🤖 Gemini AI is analyzing the article for fact-checking..."):
                gemini_result = gemini_fact_check(news_input)
            
            # Signal 1: ML Model Ensemble Vote
            predictions = [p_knn, p_log, p_rf, p_nn]
            fake_votes = sum(predictions)
            real_votes = 4 - fake_votes
            model_verdict = "FAKE" if fake_votes > real_votes else "REAL"
            model_confidence = max(fake_votes, real_votes) / 4.0 * 100
            
            # Signal 2: Cross-Reference Score (more sources = more likely real)
            total_sources = newsapi_count + newsdata_count
            if total_sources >= 5:
                cross_ref_score = 80  # Well covered — likely real
            elif total_sources >= 2:
                cross_ref_score = 50  # Some coverage
            else:
                cross_ref_score = 20  # No/minimal coverage — suspicious
            
            # Signal 3: Gemini AI Verdict
            gemini_verdict = "UNKNOWN"
            gemini_confidence = 50
            gemini_reasoning = "Gemini API was unavailable for analysis."
            if gemini_result:
                gemini_verdict = gemini_result["verdict"]
                gemini_confidence = gemini_result["confidence"]
                gemini_reasoning = gemini_result["reasoning"]
            
            # Final Weighted Credibility Score
            # Weights: ML Models = 35%, Cross-Reference = 25%, Gemini AI = 40%
            model_score = (real_votes / 4.0) * 100  # 0 = all say fake, 100 = all say real
            gemini_score = gemini_confidence if gemini_verdict == "REAL" else (100 - gemini_confidence)
            
            final_credibility = int(model_score * 0.35 + cross_ref_score * 0.25 + gemini_score * 0.40)
            final_credibility = max(0, min(100, final_credibility))
            
            if final_credibility >= 60:
                final_verdict = "LIKELY REAL"
                verdict_color = "#10B981"
                verdict_emoji = "🟢"
                verdict_bg = "rgba(16, 185, 129, 0.15)"
            elif final_credibility >= 40:
                final_verdict = "UNCERTAIN"
                verdict_color = "#F59E0B"
                verdict_emoji = "🟡"
                verdict_bg = "rgba(245, 158, 11, 0.15)"
            else:
                final_verdict = "LIKELY FAKE"
                verdict_color = "#EF4444"
                verdict_emoji = "🔴"
                verdict_bg = "rgba(239, 68, 68, 0.15)"
            
            # Display the 3 signals side-by-side
            sig1, sig2, sig3 = st.columns(3)
            
            with sig1:
                st.markdown(f"""
                <div style='background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; border-radius: 12px; padding: 16px; text-align: center;'>
                    <div style='font-size: 0.8rem; color: #A5B4FC; margin-bottom: 4px;'>📊 ML Models ({fake_votes}/4 say Fake)</div>
                    <h3 style='margin: 0; color: {"#EF4444" if model_verdict == "FAKE" else "#10B981"};'>{"🔴 FAKE" if model_verdict == "FAKE" else "🟢 REAL"}</h3>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>Confidence: {model_confidence:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with sig2:
                src_label = f"{total_sources} sources found" if total_sources > 0 else "No sources found"
                st.markdown(f"""
                <div style='background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; border-radius: 12px; padding: 16px; text-align: center;'>
                    <div style='font-size: 0.8rem; color: #A5B4FC; margin-bottom: 4px;'>🔍 Cross-Reference ({src_label})</div>
                    <h3 style='margin: 0; color: {"#10B981" if cross_ref_score >= 50 else "#EF4444"};'>{"✅ Covered" if total_sources >= 2 else "⚠️ Not Covered"}</h3>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>Score: {cross_ref_score}/100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with sig3:
                st.markdown(f"""
                <div style='background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; border-radius: 12px; padding: 16px; text-align: center;'>
                    <div style='font-size: 0.8rem; color: #A5B4FC; margin-bottom: 4px;'>🤖 Gemini AI Analysis</div>
                    <h3 style='margin: 0; color: {"#EF4444" if gemini_verdict == "FAKE" else "#10B981" if gemini_verdict == "REAL" else "#F59E0B"};'>{"🔴 FAKE" if gemini_verdict == "FAKE" else "🟢 REAL" if gemini_verdict == "REAL" else "❓ N/A"}</h3>
                    <div style='font-size: 0.8rem; opacity: 0.7;'>Confidence: {gemini_confidence}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Gemini Reasoning
            if gemini_result:
                st.markdown(f"""
                <div style='background: rgba(30, 41, 59, 0.8); border-left: 4px solid #6366F1; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 12px 0;'>
                    <div style='font-size: 0.8rem; color: #A5B4FC; margin-bottom: 4px;'>🤖 Gemini's Reasoning:</div>
                    <div style='color: #E2E8F0;'>{gemini_reasoning}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Final Verdict Card with Credibility Bar
            st.markdown(f"""
            <div style='background: {verdict_bg}; border: 2px solid {verdict_color}; border-radius: 16px; padding: 24px; text-align: center; margin-top: 16px;'>
                <div style='font-size: 1rem; color: #94A3B8; margin-bottom: 8px;'>FINAL COMBINED VERDICT</div>
                <h1 style='margin: 0; color: {verdict_color}; font-size: 2.2rem;'>{verdict_emoji} {final_verdict}</h1>
                <div style='font-size: 1.1rem; color: #CBD5E1; margin-top: 8px;'>Credibility Score: <strong style="color: {verdict_color};">{final_credibility}/100</strong></div>
                <div style='background: #1E293B; border-radius: 10px; height: 20px; margin-top: 12px; overflow: hidden;'>
                    <div style='background: {verdict_color}; height: 100%; width: {final_credibility}%; border-radius: 10px; transition: width 0.5s;'></div>
                </div>
                <div style='font-size: 0.75rem; color: #64748B; margin-top: 8px;'>Weighted: ML Models (35%) + Cross-Reference (25%) + Gemini AI (40%)</div>
            </div>
            """, unsafe_allow_html=True)
