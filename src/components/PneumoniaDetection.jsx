import React, { useState, useEffect } from "react";
import axios from "axios";
import "./PneumoniaDetection.css";

// Use environment variable for API URL
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

const PneumoniaDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [sampleImages, setSampleImages] = useState({
    normal: [],
    pneumonia: [],
  });
  const [sampleImagesLoading, setSampleImagesLoading] = useState(true);
  const [showSamples, setShowSamples] = useState(false);
  const [selectedSamplePath, setSelectedSamplePath] = useState(null);

  // Fetch list of sample images
  useEffect(() => {
    const fetchSampleImages = async () => {
      try {
        setSampleImagesLoading(true);

        // In a real implementation, you would fetch the actual file list from the server
        // For now, we'll just use our placeholder structure
        setSampleImages({
          normal: [
            {
              name: "Normal Sample 1",
              path: "/sample-xrays/normal/Normal-Sample-1.jpeg",
            },
            {
              name: "Normal Sample 2",
              path: "/sample-xrays/normal/Normal-Sample-2.jpeg",
            },
          ],
          pneumonia: [
            {
              name: "Pneumonia Sample 1",
              path: "/sample-xrays/pneumonia/Pneumonia-Sample-1.jpeg",
            },
            {
              name: "Pneumonia Sample 2",
              path: "/sample-xrays/pneumonia/Pneumonia-Sample-2.jpeg",
            },
          ],
        });
      } catch (error) {
        console.error("Error fetching sample images:", error);
      } finally {
        setSampleImagesLoading(false);
      }
    };

    fetchSampleImages();
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
      setSelectedSamplePath(null);
    }
  };

  const handleSampleImageSelect = async (imagePath) => {
    try {
      const response = await fetch(imagePath);
      const blob = await response.blob();
      const file = new File([blob], imagePath.split("/").pop(), {
        type: blob.type,
      });

      setSelectedFile(file);
      setPreview(URL.createObjectURL(blob));
      setResult(null);
      setError(null);
      setSelectedSamplePath(imagePath);
    } catch (err) {
      console.error("Error loading sample image:", err);
      setError(
        "Error loading sample image. Please try uploading your own image."
      );
    }
  };

  const toggleSampleSection = () => {
    setShowSamples(!showSamples);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await axios.post(`${API_URL}/api/predict`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        withCredentials: false,
      });

      setResult(response.data);
    } catch (err) {
      console.error("Error during prediction:", err);
      setError(
        err.response?.data?.error ||
          "An error occurred while processing your request."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="app-header">
        <h1>AI Pneumonia Detection</h1>
        <p className="subtitle">
          Upload a chest X-ray image to detect pneumonia using advanced machine
          learning algorithms
        </p>
      </div>

      <div className="main-content">
        <form onSubmit={handleSubmit}>
          <div className="file-upload">
            <label htmlFor="file-input" className="file-label">
              📁 Choose X-ray Image
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
              />
            </label>
            {selectedFile && (
              <span className="file-name">📄 {selectedFile.name}</span>
            )}
          </div>

          <div className="samples-card">
            <button
              type="button"
              className="toggle-samples-button"
              onClick={toggleSampleSection}
            >
              {showSamples ? "Hide Sample Images ▲" : "Show Sample Images ▼"}
            </button>

            {showSamples && (
              <div className="sample-images-section">
                <h3>Select a sample image:</h3>
                <div className="sample-categories">
                  <div className="sample-category">
                    <h4>Normal X-rays</h4>
                    <div className="sample-list">
                      {sampleImagesLoading ? (
                        <p>Loading samples...</p>
                      ) : (
                        sampleImages.normal.map((image, index) => (
                          <button
                            key={index}
                            className={`sample-button ${
                              selectedSamplePath === image.path
                                ? "selected"
                                : ""
                            }`}
                            onClick={() => handleSampleImageSelect(image.path)}
                            type="button"
                          >
                            {image.name}
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                  <div className="sample-category">
                    <h4>Pneumonia X-rays</h4>
                    <div className="sample-list">
                      {sampleImagesLoading ? (
                        <p>Loading samples...</p>
                      ) : (
                        sampleImages.pneumonia.map((image, index) => (
                          <button
                            key={index}
                            className={`sample-button ${
                              selectedSamplePath === image.path
                                ? "selected"
                                : ""
                            }`}
                            onClick={() => handleSampleImageSelect(image.path)}
                            type="button"
                          >
                            {image.name}
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {preview && (
            <div className="preview-container">
              <img
                src={preview}
                alt="X-ray preview"
                className="image-preview"
              />
              <div className="image-label">X-ray Image Preview</div>
            </div>
          )}

          <button
            type="submit"
            className="predict-button"
            disabled={loading || !selectedFile}
          >
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                Processing...
              </>
            ) : (
              "Analyze X-ray"
            )}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-container">
            <div className="result-header">
              <h2>Analysis Result</h2>
            </div>

            <div className="result-badge">
              <p
                className={`prediction ${
                  result.prediction === "PNEUMONIA" ? "pneumonia" : "normal"
                }`}
              >
                {result.prediction}
              </p>
              <p className="confidence">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </p>
            </div>

            <div className="visualization-container">
              <h3>Visual Explanation</h3>
              <p className="visualization-explainer">
                The heatmap overlay highlights regions that influenced the AI's
                prediction, with warmer colors (red/yellow) indicating areas of
                higher importance.
              </p>
              <div className="images-container">
                <div className="image-box">
                  <h4>Original X-ray</h4>
                  <img
                    src={`data:image/png;base64,${result.original_image}`}
                    alt="Original X-ray"
                    className="result-image"
                  />
                </div>
                <div className="image-box">
                  <h4>Heatmap Overlay</h4>
                  <img
                    src={`data:image/png;base64,${result.overlay_image}`}
                    alt="Heatmap overlay"
                    className="result-image"
                  />
                </div>
              </div>
            </div>

            <p className="result-description">
              {result.prediction === "PNEUMONIA"
                ? "The AI model has detected patterns consistent with pneumonia in this X-ray image. These patterns may include areas of opacity, consolidation, or infiltrates in the lung fields. Please consult a healthcare professional for proper diagnosis and treatment."
                : "The AI model found no significant indicators of pneumonia in this X-ray. The lung fields appear to be clear of concerning opacities. However, this is not a medical diagnosis - please consult a healthcare professional for proper evaluation."}
            </p>
          </div>
        )}
      </div>

      <footer className="app-footer">
        <p>
          <strong>Disclaimer:</strong> This is an AI-assisted tool and not a
          substitute for professional medical advice. Always consult with
          qualified healthcare professionals for diagnosis and treatment.
        </p>
      </footer>
    </div>
  );
};

export default PneumoniaDetection;
