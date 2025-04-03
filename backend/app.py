from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os
from flask import Flask, request, jsonify, send_from_directory, after_this_request
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from matplotlib import cm
import io
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, 'build'))

# Load model with better error handling
model_path = os.path.join(BASE_DIR, "models", "pneumonia_mobilenetv2.keras")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}", file=sys.stderr)
    # Create models directory if it doesn't exist
    models_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory at {models_dir}", file=sys.stderr)
    # Exit gracefully instead of crashing
    model = None
else:
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        model = None


# Add CORS headers to every response
@app.after_request
def add_cors_headers(response):
    # Allow requests from localhost and our Vercel app
    allowed_origins = [
        'http://localhost:3000',
        'https://pneumonia-detection-ai.vercel.app',
        'https://pneumonia-detection-ai-*.vercel.app',  # For preview deployments
        'https://pneumonia-detection-api-5tb5.onrender.com'  # Allow the Render domain itself
    ]
    
    origin = request.headers.get('Origin')
    if origin in allowed_origins or request.method == 'OPTIONS':
        response.headers.add('Access-Control-Allow-Origin', origin or '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    return response

# Add an OPTIONS handler for preflight requests
@app.route('/api/predict', methods=['OPTIONS'])
def options_handler():
    return '', 204

# Add a root route for health checks
@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Pneumonia Detection API is running"})

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

def model_modifier(m):
    base_model = m.layers[0]
    
    last_conv_layer = None
    
    # Loop through the layers of the base model in reverse order
    for layer in reversed(base_model.layers):
        # Check if it's a convolutional layer with a reasonable output shape
        if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
            # Use layer.output.shape instead of layer.output_shape
            output_shape = layer.output.shape.as_list()
            if len(output_shape) == 4 and output_shape[1] >= 7:  # We want a feature map of reasonable size
                last_conv_layer = layer
                break
    
    if last_conv_layer is None:
        for layer in reversed(base_model.layers):
            if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                last_conv_layer = layer
                break
        
        # If still None, use the last layer as a last resort
        if last_conv_layer is None:
            last_conv_layer = base_model.layers[-1]
    
    # Return a model that outputs the activations of the chosen layer
    return tf.keras.Model(inputs=m.inputs, outputs=last_conv_layer.output)

def generate_gradcam(img_array, model):
    # Create Gradcam object
    gradcam = Gradcam(model, model_modifier=model_modifier)
    
    def loss_function(output):
        pred = output[0][0]
        return (output[:, 0] if pred > 0.7 else 1 - output[:, 0])
    
    # Generate heatmap
    cam = gradcam(loss_function, img_array, penultimate_layer=-1)
    cam = normalize(cam)
    
    return cam[0]

def create_heatmap_overlay(original_img, heatmap, alpha=0.4):
    # Resize heatmap to match the original image size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_img.size))
    
    # Normalize the heatmap between 0 and 1 if needed
    heatmap_normalized = heatmap_resized / 255.0
    
    # Apply jet colormap
    cmap = plt.get_cmap('jet')
    colored_heatmap = cmap(heatmap_normalized)[:, :, :3]
    
    # Convert to 8-bit RGB
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    # Convert to PIL Image for display
    heatmap_img = Image.fromarray(colored_heatmap)
    
    return heatmap_img

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def generate_gradcam_simple(img_array, model):
    base_model = model.layers[0]
    
    target_layer_name = 'Conv_1'
    target_layer = None
    
    for layer in base_model.layers:
        if layer.name == target_layer_name:
            target_layer = layer
            break
    
    if target_layer is None:
        raise ValueError(f"Could not find layer {target_layer_name} in the model")
    
    feature_model = tf.keras.Model(inputs=base_model.inputs, outputs=target_layer.output)
    
    with tf.GradientTape() as tape:
        feature_map = feature_model(img_array)
        tape.watch(feature_map)
        
        x = base_model.get_layer('Conv_1_bn')(feature_map)
        x = base_model.get_layer('out_relu')(x)
        x = model.layers[1](x)  # Global Average Pooling
        x = model.layers[2](x)  # Dense layer
        x = model.layers[3](x)  # Dropout layer
        prediction = model.layers[4](x)  # Final dense layer with sigmoid
        
        score = prediction[:, 0]
    
    gradients = tape.gradient(score, feature_map)
    
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    
    # Multiply each channel in the feature map by its importance
    # Weighted sum to get the heatmap
    cam = tf.zeros(tf.shape(feature_map)[1:3], dtype=tf.float32)
    
    # Iterate over each channel and add its contribution to the heatmap
    for i in range(weights.shape[1]):
        cam += weights[0, i] * feature_map[0, :, :, i]
    
    # Apply ReLU to keep only positive contributions
    cam = tf.nn.relu(cam)
    
    # Normalize the heatmap
    if tf.reduce_max(cam) > 0:
        cam = cam / tf.reduce_max(cam)
    
    # Resize to the input size (224x224)
    cam = tf.image.resize(tf.expand_dims(tf.expand_dims(cam, axis=0), axis=-1), 
                         (224, 224))[0, :, :, 0]
    
    return cam.numpy()

def create_colored_heatmap(heatmap, size):
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100, frameon=False)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Plot the heatmap with jet colormap
    ax.imshow(heatmap, cmap='jet', aspect='auto')
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)  # Make sure to close the figure to free memory
    
    # Open the BytesIO object as an image
    buf.seek(0)
    colored_heatmap = Image.open(buf)
    colored_heatmap = colored_heatmap.resize(size, Image.LANCZOS)
    
    return colored_heatmap

def create_overlay_heatmap(original_img, heatmap, alpha=0.5):
    heatmap_size = original_img.size
    
    fig = plt.figure(figsize=(heatmap_size[0]/100, heatmap_size[1]/100), dpi=100, frameon=False)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(heatmap, cmap='jet', aspect='auto')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)  # Make sure to close the figure to free memory
    
    # Open the BytesIO object as an image
    buf.seek(0)
    colored_heatmap = Image.open(buf)
    colored_heatmap = colored_heatmap.resize(heatmap_size, Image.LANCZOS)
    
    if original_img.mode != 'RGBA':
        original_rgba = original_img.convert('RGBA')
    else:
        original_rgba = original_img.copy()
    
    if colored_heatmap.mode != 'RGBA':
        heatmap_rgba = colored_heatmap.convert('RGBA')
    else:
        heatmap_rgba = colored_heatmap.copy()
    
    overlay = Image.new('RGBA', original_rgba.size, (0, 0, 0, 0))
    overlay.paste(original_rgba, (0, 0))
    
    heatmap_data = heatmap_rgba.getdata()
    transparent_heatmap_data = []
    for item in heatmap_data:
        transparent_heatmap_data.append((item[0], item[1], item[2], int(255 * alpha)))
    
    heatmap_rgba.putdata(transparent_heatmap_data)
    
    overlay = Image.alpha_composite(overlay, heatmap_rgba)
    
    return overlay

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200
        
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file:
        file_bytes = file.read()
        file_stream = BytesIO(file_bytes)
        try:
            original_img = Image.open(file_stream).convert("RGB")
            original_size = original_img.size
            
            img = original_img.resize((224, 224))
            
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prob = model.predict(img_array)[0][0]
            
            try:
                print("Generating Grad-CAM heatmap...")
                heatmap = generate_gradcam_simple(img_array, model)
                print("Successfully generated Grad-CAM heatmap")
                
                heatmap_img = create_colored_heatmap(heatmap, original_size)
                heatmap_base64 = encode_image_to_base64(heatmap_img)
                
                overlay_img = create_overlay_heatmap(original_img, heatmap, alpha=0.6)
                overlay_base64 = encode_image_to_base64(overlay_img)
                
            except Exception as e:
                print(f"Failed to generate heatmap: {str(e)}")
                heatmap_img = Image.new('RGB', original_size, (200, 200, 200))
                draw = ImageDraw.Draw(heatmap_img)
                draw.text((20, 20), "Heatmap generation failed", fill=(0, 0, 0))
                heatmap_base64 = encode_image_to_base64(heatmap_img)
                
                overlay_img = original_img.copy()
                overlay_base64 = encode_image_to_base64(overlay_img)
            
            original_base64 = encode_image_to_base64(original_img)
            
            if prob > 0.5:
                return jsonify({
                    "prediction": "PNEUMONIA", 
                    "confidence": float(prob),
                    "original_image": original_base64,
                    "heatmap_image": heatmap_base64,
                    "overlay_image": overlay_base64
                })
            else:
                return jsonify({
                    "prediction": "NORMAL", 
                    "confidence": float(1.0 - prob),
                    "original_image": original_base64,
                    "heatmap_image": heatmap_base64,
                    "overlay_image": overlay_base64
                })
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400
            
    return jsonify({"error": "File upload failed"}), 400

if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

