import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download required NLTK datasets (if not already present)
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Global flag to avoid repeated logging for fallback tokenizer
fallback_used = False

def safe_word_tokenize(text):
    """
    Tries to tokenize text using nltk.word_tokenize.
    If a LookupError occurs and mentions 'punkt_tab', it falls back to simple whitespace splitting.
    Logs a warning only once.
    """
    global fallback_used
    try:
        return nltk.word_tokenize(text)
    except LookupError as e:
        if "punkt_tab" in str(e):
            if not fallback_used:
                logger.warning("LookupError: 'punkt_tab' resource not found. Using basic split tokenization.")
                fallback_used = True
            return text.split()
        else:
            raise

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing,
    removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)

    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text using safe_word_tokenize
    words = safe_word_tokenize(text)
    # Remove non-alphanumeric tokens
    words = [word for word in words if word.isalnum()]
    # Remove stopwords and punctuation
    words = [word for word in words if word not in stop_words and word not in punctuation_set]
    # Stem the words
    words = [ps.stem(word) for word in words]
    # Return the processed text
    return " ".join(words)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates,
    and transforming the text column.
    """
    logger.debug('Starting preprocessing for DataFrame')
    
    # Encode the target column
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])
    logger.debug('Target column encoded')

    # Remove duplicate rows
    df = df.drop_duplicates(keep='first').copy()
    logger.debug('Duplicates removed')
    
    # Apply text transformation to the specified text column
    df[text_column] = df[text_column].astype(str).apply(transform_text)
    logger.debug('Text column transformed')
    
    return df

def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Define file paths for training and test data
        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Train or test file not found in data/raw directory")

        # Load the data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('Data loaded properly')

        # Preprocess the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Save processed data into the "interim" directory
        data_path = "./data/interim"
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
