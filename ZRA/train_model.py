import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')

def preprocess_review(review_text):
    """
    Cleans and preprocesses a single review string.
    """
    review = re.sub('[^a-zA-Z]', ' ', review_text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    return review

# Load the dataset
# Use raw string for Windows path
try:
    df = pd.read_csv(r"D:\datasets\zomato review analysis.csv")
except FileNotFoundError:
    raise FileNotFoundError("The file 'zomato review analysis.csv' was not found. Please upload it to the directory.")

# Preprocess the entire 'Review' column
df['Review'] = df['Review'].apply(preprocess_review)

# Create the Bag of Words model (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X = tfidf_vectorizer.fit_transform(df['Review']).toarray()
y = df['Liked'].values  # Ensure it's a numpy array

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Save the trained model and the TF-IDF vectorizer to disk
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Model and TF-IDF vectorizer saved successfully!")
# to run this type this  pip install -r requirements.txt
# python app.py