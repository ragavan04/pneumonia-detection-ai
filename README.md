# Pneumonia Detection AI

A web application for detecting pneumonia from chest X-ray images using artificial intelligence.

## Project Structure

```
pneumonia-detection-ai/
├── backend/              # Backend Flask server
│   ├── app.py            # Flask application
│   ├── requirements.txt  # Python dependencies
│   ├── test.py           # Test script for model
│   ├── train.py          # Training script for model
│   ├── data/             # Training and test data
│   ├── models/           # Trained models
│   └── venv/             # Python virtual environment
└── public/               # Static files including sample X-rays
└── src/                  # React source code
    ├── components/       # React components
    ├── App.js            # Main App component
    └── index.js          # Entry point
```

## Features

- Upload chest X-ray images
- Real-time prediction with confidence scores
- Visualization with Grad-CAM heatmaps
- Sample X-ray library for testing
- Modern React-based UI

## Local Setup Instructions

### Backend Setup

1. Navigate to the backend directory:

   ```
   cd backend
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the Flask backend:
   ```
   python app.py
   ```
   The backend will run on http://localhost:5000

### Frontend Setup

1. From the project root, install Node.js dependencies:

   ```
   npm install
   ```

2. Start the React development server:
   ```
   npm start
   ```
   The frontend will run on http://localhost:3000

## Deployment Instructions

### Backend Deployment (Render)

1. Push your code to GitHub
2. Sign up for a Render account at https://render.com
3. Create a new Web Service and connect your GitHub repository
4. Use the following settings:
   - Name: pneumonia-detection-api
   - Environment: Python 3
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && gunicorn app:app`
   - Add the following environment variable: PYTHON_VERSION=3.9.0

### Frontend Deployment (Vercel)

1. Push your code to GitHub
2. Sign up for a Vercel account at https://vercel.com
3. Create a new project and import your GitHub repository
4. Use the following settings:
   - Framework Preset: Create React App
   - Root Directory: ./
   - Build Command: `npm run build`
   - Output Directory: build
   - Install Command: `npm install`
5. Add the following environment variable:
   - REACT_APP_API_URL=https://your-render-backend-url.onrender.com
6. Click Deploy

## Usage

1. Open the application in your browser
2. Upload a chest X-ray image or select a sample image
3. Click the "Predict" button
4. View the prediction result, confidence score, and heatmap visualization

## Notes

- The model uses TensorFlow and MobileNetV2 for classification
- The Grad-CAM visualization highlights areas of interest in the X-ray
- For Apple Silicon Macs, TensorFlow installation may require additional steps
