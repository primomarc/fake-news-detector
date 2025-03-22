import pandas as pd
import numpy as np
import nltk
nltk.download('punkt_tab')
nltk.download('omw-1.4') 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def load_dataset(filepath):
    """Load the fake news dataset."""
    df = pd.read_csv(filepath)
    print("Dataset Loaded Successfully!\n")
    print(df.head())
    return df

def preprocess_text(text):
    """Clean and preprocess text for machine learning."""
    if pd.isnull(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def preprocess_dataset(df):
    """Apply text preprocessing to the dataset."""
    df["clean_text"] = df["text"].apply(preprocess_text)
    print("Preprocessing Completed!\n")
    print(df.head())
    return df

if __name__ == "__main__":
    # Load dataset
    dataset_path = r"C:\Users\Admin\Desktop\Fakenews\fakenews.csv"  # Update with actual file path
    df = load_dataset(dataset_path)
    
    # Preprocess dataset
    df = preprocess_dataset(df)
    
    # Save cleaned dataset
    df.to_csv("clean_news_dataset.csv", index=False)
    print("Cleaned dataset saved successfully!")
