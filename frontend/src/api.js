import axios from "axios";

const API_URL = "http://localhost:5000/predict";

export const fetchSentiments = async (video_url) => {
  try {
    const response = await axios.post(API_URL, { video_url });
    return response.data;
  } catch (error) {
    console.error("Error fetching sentiments:", error);
    return [];
  }
};
