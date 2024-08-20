import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file
df = pd.read_csv('responses.csv')  # Replace with your actual CSV file name

# Assuming the CSV has columns 'intent' and 'response'
text_data = df['response'].tolist()



# 1. Text Cleaning
def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    else:
        text = ''  # Handle non-string values, e.g., set them to an empty string
    return text

# 2. Tokenization
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# 3. Stop Word Removal
def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


# Preprocessing pipeline
preprocessed_data = []
for text in text_data:
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stop_words(tokens)
    preprocessed_data.append(filtered_tokens)

# Print the preprocessed data
for i, data in enumerate(preprocessed_data):
    print(f"Original: {text_data[i]}")
    print(f"Preprocessed: {data}")
    print()


# 4. Text Vectorization (TF-IDF Example)
vectorizer = TfidfVectorizer(tokenizer=lambda text: text, lowercase=False, preprocessor=None)
tfidf_matrix = vectorizer.fit_transform(preprocessed_data)

# Print TF-IDF Matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
