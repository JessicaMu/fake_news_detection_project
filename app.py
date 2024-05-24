'''This Flask API enables real-time predictions of news articles. This API allows external applications to send news 
article data to the server and receive predictions on whether the articles are fake or real.'''

from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from textblob import TextBlob
import re
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained models and vectorizers.
rf_model = joblib.load('Random Forest Model.joblib')
svm_model = joblib.load('Support Vector Machine Model.joblib')
vectorizer_text = joblib.load('vectorizer_text.joblib')
vectorizer_title = joblib.load('vectorizer_title.joblib')
vectorizer_subject = joblib.load('vectorizer_subject.joblib')
encoder_sentiment = joblib.load('encoder_sentiment.joblib')

# Preprocessing the text to clean data.
def preprocess_text(text):
    text = text.lower() # Convert text to lowercase
    abbreviations = ["U.S.", "Dr.", "etc.", "e.g.", "i.e."]
    tokens = []
    for word in text.split():
        found_abbreviation = False
        for abbr in abbreviations:
            if abbr in word:
                abbr_without_punctuation = ''.join(char for char in abbr if char.isalnum())
                tokens.append(abbr_without_punctuation)
                found_abbreviation = True
                break
        if not found_abbreviation:
            # Tokenize the text
            tokens.extend(re.findall(r'[A-Z]{2,}(?:\.[A-Z]\.)?(?:[,.!?]|$)|[A-Z]?[a-z]+|[A-Z]+|[a-z]+(?=[A-Z])', word))
    # Remove stopwords
    stopwords = ["the", "and", "is", "it", "in", "to", "of", "an", "a"]
    tokens_without_stopwords = [word for word in tokens if word not in stopwords]
    # Join tokens back into a string and remove punctuation
    preprocessed_text = ' '.join(tokens_without_stopwords).lower()
    text_without_punctuation = re.sub(r'[^\w\s]', '', preprocessed_text)
    return text_without_punctuation


# Analyze the sentiment of the text
def analyse_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        df = pd.read_csv(file)
        
        # Ensure the required columns are present
        required_columns = ['title', 'text', 'subject']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'CSV file must contain title, text, and subject columns'}), 400

        results = []

        for index, row in df.iterrows():
            # Preprocess the text data
            title = preprocess_text(row['title'])
            text = preprocess_text(row['text'])
            subject = preprocess_text(row['subject'])
            sentiment_text = preprocess_text(row['text'])
            sentiment = analyse_sentiment(sentiment_text)

            print(f"Preprocessed Title: {title}")
            print(f"Preprocessed Text: {text}")
            print(f"Preprocessed Subject: {subject}")
            print(f"Sentiment: {sentiment}")

            # Transform the text data using the vectorizers
            x_title = vectorizer_title.transform([title])
            x_text = vectorizer_text.transform([text])
            x_subject = vectorizer_subject.transform([subject])
            x_sentiment = encoder_sentiment.transform(pd.DataFrame([[sentiment]], columns=['sentiment']))

            # Combine the transformed features
            x = hstack([x_title, x_text, x_subject, x_sentiment])

            # Make predictions using both models
            rf_prediction = rf_model.predict(x)
            svm_prediction = svm_model.predict(x)
            rf_label = 'fake' if rf_prediction[0] == 0 else 'real'
            svm_label = 'fake' if svm_prediction[0] == 0 else 'real'

            results.append({
                'title': row['title'],
                'rf_prediction': f'This article is {rf_label}',
                'svm_prediction': f'This article is {svm_label}'
            })

            print(f"RF Prediction: {rf_label}, SVM Prediction: {svm_label}")

        # Save the results to a new CSV file
        results_df = pd.DataFrame(results)
        results_csv = 'prediction_results.csv'
        results_df.to_csv(results_csv, index=False)

        return jsonify({'message': 'Predictions made successfully', 'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
