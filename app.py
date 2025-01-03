import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model

# Flask app setup
app = Flask(__name__, static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG', '.jpeg', '.png']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

# Global variables
n_classes = 5
Class_Penyakit = ['Whitefly', 'Yellowish', 'Healthy', 'Leaf Curl', 'Leaf Spot']

# Load model (ensure consistent architecture with saved weights)
def make_model(n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model

# Load the model weights
model = make_model(n_classes)
model.summary()
model.load_weights("./clm.h5", by_name=True, skip_mismatch=True)  # Adjust path if necessary

# Routing
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'
    confidence_score = 0.0

    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = os.path.join(app.config['UPLOAD_PATH'], filename)

        # Create directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_PATH']):
            os.makedirs(app.config['UPLOAD_PATH'])

        if file_ext in app.config['UPLOAD_EXTENSIONS']:
            uploaded_file.save(gambar_prediksi)

            try:
                # Preprocess the image
                test_image = Image.open(gambar_prediksi)
                if test_image.mode != "RGB":
                    test_image = test_image.convert("RGB")  # Ensure RGB format
                test_image_resized = test_image.resize((256, 256))  # Resize to model input size
                image_array = np.array(test_image_resized)

                # Normalize the image
                test_image_x = image_array / 255.0
                test_image_x = np.expand_dims(test_image_x, axis=0)

                # Predict the image
                y_pred_test_single = model.predict(test_image_x)
                y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)

                # Get the result
                hasil_prediksi = Class_Penyakit[y_pred_test_classes_single[0]]
                confidence_score = np.max(y_pred_test_single)
                confidence_score = round(float(confidence_score), 4)  # Round to 4 decimal places

            except Exception as e:
                hasil_prediksi = 'Error processing image'
                confidence_score = 0.0

            return jsonify({
                "prediksi": hasil_prediksi,
                "confidence": confidence_score,
                "gambar_prediksi": gambar_prediksi
            })

    return jsonify({
        "prediksi": hasil_prediksi,
        "confidence": confidence_score,
        "gambar_prediksi": gambar_prediksi
    })

@app.route('/process', methods=['POST'])
def process_image():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Convert FileStorage to PIL Image
        image = Image.open(uploaded_file.stream).convert('RGB')
        # Resize and preprocess image
        image = image.resize((256, 256))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0

        # Get model predictions and intermediate layer outputs
        layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name or 'dense' in layer.name]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(image_array)

        # Prepare visualization data
        visualizations = {}
        for i, activation in enumerate(activations):
            layer_name = model.layers[i].name
            if len(activation.shape) == 4:  # Conv2D layer
                # Compute mean activation for heatmap
                mean_activation = np.mean(activation[0], axis=-1)
                plt.imshow(mean_activation, cmap='viridis')
                plt.title(f'{layer_name} Activations')

                # Save the plot to a string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                encoded_image = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                visualizations[layer_name] = encoded_image

            elif len(activation.shape) == 2:  # Dense layer
                # Visualize activation values as a bar chart
                plt.bar(range(activation.shape[1]), activation[0])
                plt.title(f'{layer_name} Activations')

                # Save the plot to a string
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                encoded_image = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                visualizations[layer_name] = encoded_image

        # Return visualizations as JSON
        return jsonify(visualizations)
    return jsonify({"error": "No file uploaded"})

# Main entry point
if __name__ == '__main__':
    app.run(debug=True, port=5500)