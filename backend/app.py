from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import torch
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences
import pickle
import requests
import numpy as np
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config
from sentiment_model import sentimentBiLSTM

# Configure logging for debugging.
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = Config.DEBUG

# Connect to MongoDB.
client = MongoClient(Config.MONGO_URI)
db = client[Config.DATABASE_NAME]
collection = db[Config.COLLECTION_NAME]

# Load the saved Keras tokenizer.
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    logging.exception("Error loading tokenizer:")
    raise e

# Determine vocabulary size from the tokenizer.
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300       # Must match training – typically 300.
hidden_dim = 64
output_size = 3

# Initialize the model.
# We use a dummy embedding matrix (zeros) of shape [vocab_size, embedding_dim] 
# because the actual trained embeddings are in the saved state.
dummy_embedding = np.zeros((vocab_size, embedding_dim))
model = sentimentBiLSTM(dummy_embedding, hidden_dim, output_size)
model = model.to(torch.device("cpu"))

# Load model weights saved during training (assumed saved as model_best.pt).
try:
    state_dict = torch.load("model_best.pt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    logging.exception("Error loading model weights:")
    raise e

def extract_video_id(url):
    """Extract the YouTube video ID from a standard URL format."""
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url).query
    params = parse_qs(query)
    if "v" in params and params["v"]:
        return params["v"][0]
    else:
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
    """
    Converts comments to padded sequences using the tokenizer and returns sentiment predictions.
    """
    if not comments:
        raise ValueError("No comments to analyze.")
    # Convert texts to sequences.
    sequences = tokenizer.texts_to_sequences(comments)
    # Pad sequences to the maximum length (Config.MAX_LEN).
    padded = pad_sequences(sequences, maxlen=Config.MAX_LEN, padding='post', truncating='post')
    inputs = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        outputs = model(inputs)
    # Get index of maximum predicted value for each sample.
    preds = torch.argmax(outputs, dim=1).numpy()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [label_map.get(int(p), "Unknown") for p in preds]

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
        
        # Get sentiment predictions.
        sentiments = predict_sentiment(comments)
        results = [{"comment": c, "sentiment": s} for c, s in zip(comments, sentiments)]
        return jsonify(results)
        
    except Exception as e:
        logging.exception("Error in /predict endpoint:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=Config.DEBUG)
