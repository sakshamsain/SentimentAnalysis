from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import tensorflow as tf
import pickle
import requests
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from threading import Timer
app = Flask(__name__)
CORS(app)


# Connect to MongoDB
client = MongoClient(Config.MONGO_URI)
db = client[Config.DATABASE_NAME]
collection = db[Config.COLLECTION_NAME]

# Load the new model and tokenizer
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Download stopwords if not already downloaded
nltk.download('stopwords')

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
SEQUENCE_LENGTH = Config.MAX_LEN  # Ensure this matches your training setup

def preprocess(text, stem=False):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

# Assuming you have defined NEUTRAL, NEGATIVE, POSITIVE, SENTIMENT_THRESHOLDS
NEUTRAL = "Neutral"
NEGATIVE = "Negative"
POSITIVE = "Positive"
SENTIMENT_THRESHOLDS = [0.4, 0.7]  # Example thresholds

def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def extract_video_id(url):
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url).query
    params = parse_qs(query)
    if "v" in params and params["v"]:
        return params["v"][0]
    else:
        # Handle shortened YouTube URLs (e.g., youtu.be)
        path = urlparse(url).path
        if path.startswith("/"):
            return path.split("/")[1]
        raise ValueError("Invalid YouTube URL. Could not extract video ID.")

def fetch_youtube_comments(video_url):
    """
    Fetches comments from YouTube using the YouTube Data API.
    """
    api_key = Config.YOUTUBE_API_KEY
    video_id = extract_video_id(video_url)
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults=100"
    response = requests.get(url)
    data = response.json()
    for item in data.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
        comments.append(comment)
    return comments
def predict_sentiment(comments):
    if not comments:
        raise ValueError("No comments to analyze.")
    
    # Preprocess comments
    preprocessed_comments = [preprocess(comment) for comment in comments]
    
    # Vectorize the preprocessed text
    sequences = tokenizer.texts_to_sequences(preprocessed_comments)
    padded = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
    
    # Make predictions
    scores = model.predict(padded, verbose=0)
    
    # Decode the predictions
    predictions = [decode_sentiment(score[0], include_neutral=True) for score in scores]
    
    return predictions

def schedule_mongo_deletion(video_id, delay=500000):
    def delete_docs():
        result = collection.delete_many({"video_id": video_id})
        logging.info(f"Deleted {result.deleted_count} documents for video_id: {video_id}")
    Timer(delay, delete_docs).start()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        video_url = data.get("video_url")
        if not video_url:
            return jsonify({"message": "No video URL provided"}), 400
        
        # Fetch comments for the URL.
        comments = fetch_youtube_comments(video_url)
        
        # Optionally store comments in MongoDB.
        if comments:
            video_id = extract_video_id(video_url)
            docs = [{"video_id": video_id, "comment": comment} for comment in comments]
            collection.insert_many(docs)
            
            # Schedule deletion of these documents after 5 seconds.
            schedule_mongo_deletion(video_id, delay=50000)
        
        # Get sentiment predictions using the new model.
        sentiments = predict_sentiment(comments)
        
        results = [{"comment": c, "sentiment": s} for c, s in zip(comments, sentiments)]
        
        return jsonify(results)
    
    except Exception as e:
        logging.exception("Error in /predict endpoint:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=Config.DEBUG)
