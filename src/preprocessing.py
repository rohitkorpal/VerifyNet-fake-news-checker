import re

# Comprehensive list of standard English stopwords to avoid NLTK downloads
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
    'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here',
    'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in',
    'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that',
    'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd',
    'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres',
    'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd',
    'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
}

def clean_text(text):
    """
    Cleans raw text by:
    1. Lowercasing.
    2. Stripping HTML tags.
    3. Removing punctuation and non-alphabetic characters.
    4. Removing extra whitespaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URL links
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Keep only letters and numbers, replace punctuation/special chars with space
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(text):
    """
    Tokenizes clean text into words.
    """
    return text.split()

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.
    """
    return [token for token in tokens if token not in STOPWORDS]

def strip_datelines_and_leakage(text):
    """
    Strips dateline prefixes commonly found in news agency reports.
    Example: 'WASHINGTON (Reuters) - ...' -> '...'
    Also removes explicit publisher tags like '(Reuters)', 'Reuters', etc.
    """
    if not isinstance(text, str):
        return ""
        
    # Match pattern: Location (Publisher) - or Location (Publisher) --
    # e.g., 'WASHINGTON (Reuters) - ' or 'SEOUL/LONDON (Reuters) -- '
    text_clean = re.sub(r'^[A-Z\s,\./]+ \([A-Za-z\s]+\)\s*-\s*-?', '', text)
    
    # Also strip just '(Reuters)' or 'Reuters' to prevent the model from learning shortcuts
    text_clean = re.sub(r'\b(reuters|reuters.com)\b', '', text_clean, flags=re.IGNORECASE)
    
    return text_clean

def preprocess_pipeline(text):
    """
    Complete pipeline: clean, tokenize, and remove stopwords.
    Returns a string of cleaned words joined by space.
    """
    scrubbed = strip_datelines_and_leakage(text)
    cleaned = clean_text(scrubbed)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    return " ".join(filtered)
