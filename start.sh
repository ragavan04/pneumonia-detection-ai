#!/bin/bash

# Start backend
echo "Starting backend server..."
cd backend
source venv/bin/activate

# Install required packages (including new ones for Grad-CAM)
echo "Installing backend dependencies..."
pip install -r requirements.txt

python app.py &
BACKEND_PID=$!
cd ..

# Start frontend
echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!

# Function to handle Ctrl+C
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

# Register the cleanup function for SIGINT (Ctrl+C)
trap cleanup INT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID 