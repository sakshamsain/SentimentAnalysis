# YouTube Sentiment Analysis Project 
##model: https://drive.google.com/file/d/1oGhYKPYM0IggSGqWcsZJfB7lNzAne7bc/view?usp=sharing
This project provides an end-to-end solution to analyze the sentiment of YouTube video comments using a trained BiLSTM model. The solution consists of a model training pipeline (in `model.py`) and a Flask backend API (in `app.py`), which together perform the following steps:

- **Input**: A YouTube video URL is provided.
- **Extraction**: The video ID is extracted from the URL.
- **Fetching**: Up to 100 comments are fetched from the YouTube Data API.
- **Preprocessing**: Comments are tokenized and padded (using a saved Keras tokenizer and a fixed MAX_LEN).
- **Prediction**: A PyTorch BiLSTM model (loaded from a saved state dictionary) predicts the sentiment (Negative, Neutral, or Positive) of each comment.
- **Database Logging**: The fetched comments are temporarily stored in a MongoDB Atlas collection.
- **Automatic Cleanup**: A background timer deletes the inserted comments after a few seconds (default is 5 seconds).
- **Output**: The sentiment predictions are returned as a JSON response.

---
## Screenshots

### **Application Running**
![Screenshot 1](https://github.com/sakshamsain/SentimentAnalysis/blob/main/Screenshot%202025-02-05%20113444.png)

### **API Response**
![Screenshot 2](https://github.com/sakshamsain/SentimentAnalysis/blob/main/Screenshot%202025-02-05%20113559.png)

---
## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Backend](#running-the-backend)
- [How It Works](#how-it-works)
- [Training Pipeline (model.py)](#training-pipeline-modelpy)
- [MongoDB Data Deletion](#mongodb-data-deletion)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project leverages a trained BiLSTM model for sentiment classification to analyze YouTube comments. The backend, built with Flask, performs the following:

1. **Extract Video ID:**  
   Parses the URL to retrieve the YouTube video ID.

2. **Fetch Comments:**  
   Uses the YouTube Data API to fetch up to 100 comments for the video.

3. **Preprocess Comments:**  
   Converts comments to sequences (using a pretrained Keras tokenizer) and pads them to a maximum length (MAX_LEN).

4. **Predict Sentiment:**  
   Passes the padded sequences to a PyTorch BiLSTM model and produces sentiment predictions.

5. **Store and Cleanup:**  
   Temporarily stores the fetched comments in a MongoDB Atlas database, then deletes them after a short delay to keep the database clean.

---

## Directory Structure

Below is an example structure for the project:


---

## Setup and Installation

1. **Create a Virtual Environment & Install Requirements**  
   Navigate to your project directory and run:

2. **Configuration**  
- Update `config.py` with your MongoDB Atlas credentials and YouTube API key.
- Ensure the `MAX_LEN` (set to 167) matches the value used during training.

3. **Artifacts**  
Place the saved model weights (`model_best.pt`) and the tokenizer (`tokenizer.pkl`) in the backend folder.

---

## Running the Backend

1. **Start Flask Server**  
From the backend directory, run:
The server will start (by default on port 5000).

2. **Test the Endpoint**  
Use a tool like Postman to send a POST request to:
http://localhost:5000/predict
The API will then return a JSON array with each comment and its corresponding sentiment.

---

## How It Works

- **Comments & Preprocessing:**  
The YouTube Data API is called to fetch comments. These comments are then tokenized using a previously saved Keras tokenizer and padded to the required sequence length (MAX_LEN).

- **Model Prediction:**  
The processed sequences are passed into a PyTorch BiLSTM model which outputs sentiment probabilities. The highest probability is taken as the predicted sentiment.

- **Database & Deletion:**  
Fetched comments are inserted into MongoDB Atlas for logging. A background timer (using Python’s `threading.Timer`) automatically deletes these documents after a set delay (default is 5 seconds).

---

## Training Pipeline (model.py)

The provided `model.py` script (originally created in Colab) does the following:
- Loads and preprocesses a dataset.
- Tokenizes and pads text data.
- Loads GloVe embeddings and creates an embedding matrix.
- Defines and trains a BiLSTM model in PyTorch.
- Saves the trained model weights as `model_best.pt` and the tokenizer as `tokenizer.pkl`.

Refer to `model/model.py` for the complete training process.

---

## MongoDB Data Deletion

After each successful prediction:
- Comments (with the associated video_id) are stored in MongoDB Atlas.
- A timer is scheduled to delete these documents after 5 seconds.
- This helps ensure that the database remains clean and does not accumulate unnecessary data.

---

## Future Improvements

- **Enhanced Error Handling:**  
Improve exception management and logging throughout the backend.

- **Frontend Integration:**  
Develop an interactive frontend (e.g., using React) to allow users to input YouTube URLs and visualize sentiment results.

- **Model Improvements:**  
Experiment with additional epochs, hyperparameter tuning, and possibly use alternative architectures for better performance.

- **Deployment:**  
Containerize the application using Docker and deploy on a cloud platform with secure configuration for API keys and credentials.

---

## License

This project is open source and available under the [MIT License](LICENSE).

