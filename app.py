import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load or create a social media sentiment dataset (replace with your data)
data = {'text': ["I love this product!", "Terrible experience, never buying again.",
                 "Excited about the new features.", "Not impressed with the service.",
                 "Great customer support!", "Feeling neutral about the update."],
        'sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Neutral']}

df = pd.DataFrame(data)

# Data Cleaning and Preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = ' '.join(word for word in word_tokenize(text) if word.isalpha())
    # Remove stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Lemmatization
    text = ' '.join(lemmatizer.lemmatize(word) for word in word_tokenize(text))
    
    return text

df['text'] = df['text'].apply(preprocess_text)

# Data Analysis (optional)
st.title("Social Media Sentiment Analysis App")

# Display basic statistics
st.write("Dataset Overview:")
st.write(df.head())

# Model Building
X = df['text']
y = df['sentiment']

vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("Model Evaluation:")
st.write(f'Accuracy: {accuracy:.2%}')
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

# Save the model
joblib.dump((model, vectorizer), 'social_media_sentiment_model.pkl')

# Streamlit App
st.sidebar.header("Sentiment Prediction")

# User input for prediction
user_input = st.sidebar.text_area("Enter a social media post:")
if user_input:
    # Load the model
    loaded_model, loaded_vectorizer = joblib.load('social_media_sentiment_model.pkl')

    # Transform user input
    user_input_transformed = loaded_vectorizer.transform([preprocess_text(user_input)])

    # Make prediction
    prediction = loaded_model.predict(user_input_transformed)[0]

    # Display prediction
    st.sidebar.subheader("Prediction:")
    st.sidebar.write(f'The sentiment is: {prediction}')