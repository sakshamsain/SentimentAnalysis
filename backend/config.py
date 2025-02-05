import os

class Config:
    # MongoDB connection URI – update with your credentials
    MONGO_URI = "mongodb+srv://sakshamsainisaini94:71OCZxDAuko52bTf@cluster0.ek5cg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DATABASE_NAME = "youtube_comments"
    COLLECTION_NAME = "comments"
    
    DEBUG = True
    
    # YouTube Data API key – update with your own key
    YOUTUBE_API_KEY = "AIzaSyCogofgKd8wjSHvLN2eQAEy9ofS3bQwemQ"
    
    # Maximum sequence length (must be same as used during training)
    MAX_LEN = 167
