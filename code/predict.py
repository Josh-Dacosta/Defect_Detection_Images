# This code is a test environment for analyzing a folder
# of images using a pre-trained TensorFlow model and storing
# the prediction results in a SQLite database.
# For the code to work, it will load in a folder of test images,
# preferably of a single part, and run predictions on them.

import os
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image

# Load the TensorFlow model from the specified path
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess the image to match the model's expected input
def preprocess_image(image_path, img_size=(224, 224)):  
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    arr = np.array(img) / 255.0  # Normalize to [0, 1]
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    return arr

# Run a predition on a single image
def single_image_prediction(model, image_path, class_names):
    arr = preprocess_image(image_path)
    predictions = model.predict(arr)
    predictions_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    return class_names[predictions_index], confidence

# Run a prediction for all images in a folder
def predict_folder(model, folder_path, class_names):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")
    
    model = load_model(model_path)

    # Supported image file extensions
    extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ]

    if not images:
        print("No images found in the specified folder.")
        return
    
    print("\n=============================================")
    print(f"Model utilized: {model_path}")
    print(f" Total images found: {len(images)}")
    print(f" Predictions for images in folder: {folder_path}")
    print("=============================================\n")

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        predicted_label, confidence = single_image_prediction(model, img_path, class_names)

        print(f" Image:      {img_name}")
        print(f" Predicted:  {predicted_label}")
        print(f" Confidence: {confidence:.4f}\n")
        print("=============================================\n")

# Take the image folder as input argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction for images in a folder.")
    parser.add_argument("--model", type=str, required=True, help="Path to the pre-trained TensorFlow model.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images for prediction.")
    parser.add_argument("--classes", default="good,bad", help="Comma-separated list of class names. (Good, Bad)")
    args = parser.parse_args()

    model_path = args.model
    class_names = args.classes.split(',')

    predict_folder(load_model(model_path), args.folder, class_names)