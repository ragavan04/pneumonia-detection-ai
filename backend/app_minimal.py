import os
import sys
from flask import Flask, jsonify, request

# Print startup debugging information
print(f"Starting minimal app...", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Working directory: {os.getcwd()}", file=sys.stderr)
print(f"Environment variables: PORT={os.environ.get('PORT')}", file=sys.stderr)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}", file=sys.stderr)

app = Flask(__name__)

# Add CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Minimal Pneumonia Detection API is running"})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"result": "success", "message": "Test endpoint working"})

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":
        return "", 200
    
    # For debugging, just return a simple response
    return jsonify({
        "prediction": "PNEUMONIA", 
        "confidence": 0.95,
        "original_image": "test_image_data",
        "message": "This is a test response from the minimal app"
    })

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 10000))
        print(f"Starting minimal app on port {port}", file=sys.stderr)
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Error during startup: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) 