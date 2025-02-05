"use client";
import { useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import axios from "axios";

export default function YoutubeSentimentAnalyzer() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [sentimentData, setSentimentData] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5000/predict", { video_url: url });
      const results = response.data;
      const sentimentCounts = { Positive: 0, Negative: 0, Neutral: 0 };

      results.forEach((item) => {
        sentimentCounts[item.sentiment]++;
      });

      const total = results.length;
      const chartData = [
        { name: "Positive", value: (sentimentCounts.Positive / total) * 100, color: "#32CD32" },
        { name: "Negative", value: (sentimentCounts.Negative / total) * 100, color: "#FF4500" },
        { name: "Neutral", value: (sentimentCounts.Neutral / total) * 100, color: "#FFD700" }
      ];

      setSentimentData(chartData);
      setShowResults(true);
    } catch (error) {
      console.error("Error analyzing comments:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter YouTube video URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={{ width: "300px", padding: "8px" }}
        />
        <button type="submit" style={{ marginLeft: "10px", padding: "8px 12px" }}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>
      {showResults && (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={sentimentData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label
            >
              {sentimentData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
