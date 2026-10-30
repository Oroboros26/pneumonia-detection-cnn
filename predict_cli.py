import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_HEIGHT = 224
IMG_WIDTH = 224


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path: str):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict(model, image_path: str):
    image_batch = preprocess_image(image_path)
    prediction = model.predict(image_batch)
    confidence = float(prediction[0][0])
    label = 'Pneumonia' if confidence > 0.5 else 'Normal'
    score = confidence if confidence > 0.5 else 1 - confidence
    return label, score


def main():
    parser = argparse.ArgumentParser(description='Pneumonia detection CLI for chest X-ray images.')
    parser.add_argument('--model', '-m', default='pneumonia_detection_model_transfer.h5', help='Path to the .h5 model file')
    parser.add_argument('--image', '-i', required=True, help='Path to the chest X-ray image file')
    args = parser.parse_args()

    model = load_model(args.model)
    label, confidence = predict(model, args.image)

    print(f'Image: {args.image}')
    print(f'Prediction: {label}')
    print(f'Confidence: {confidence:.4f}')


if __name__ == '__main__':
    main()
