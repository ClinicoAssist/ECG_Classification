import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [imagePreview, setImagePreview] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setImagePreview(URL.createObjectURL(selectedFile)); // Set image preview
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://192.168.231.169:4000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error in prediction:", error);
      setPrediction("Failed to get a prediction");
    }
  };

  return (
    <div className="container">
      <h2>ECG Classification</h2>
      <p className="description">
        This application allows you to upload ECG images for classification. The
        model will analyze the image and provide a prediction regarding the
        heart condition based on the ECG data.
      </p>
      <div className="file-input">
        <input type="file" id="file" onChange={handleFileChange} />
        <label htmlFor="file">Choose File</label>
      </div>
      <button className="upload-button" onClick={handleUpload}>
        Upload & Predict
      </button>
      {imagePreview && (
        <div className="image-preview">
          <h3>Uploaded Image:</h3>
          <img src={imagePreview} alt="Uploaded" className="preview-image" />
        </div>
      )}
      {prediction && (
        <div className="prediction-result">Prediction: {prediction}</div>
      )}
    </div>
  );
}
