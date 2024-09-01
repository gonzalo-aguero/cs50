import argparse
import cv2
import numpy as np
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the trained model (model.h5)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", action="store_true", help="Use the camera for real-time predictions")
    group.add_argument("--image", help="Path to the image file for prediction")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    if args.camera:
        predict_from_camera(model)
    elif args.image:
        predict_from_file(model, args.image)

def predict_from_camera(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        processed_img = preprocess_image(frame)
        predictions = model.predict(np.array([processed_img]))
        predicted_category = np.argmax(predictions[0])

        cv2.putText(frame, f"Prediction: {predicted_category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Traffic Sign Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_from_file(model, file_path):
    img = cv2.imread(file_path)
    if img is None:
        print(f"Error: Could not load image from {file_path}")
        return

    processed_img = preprocess_image(img)
    predictions = model.predict(np.array([processed_img]))
    predicted_category = np.argmax(predictions[0])

    print(f"Predicted category: {predicted_category}")
    # cv2.putText(img, f"Prediction: {predicted_category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print("Prediction:", predicted_category)
    # cv2.imshow("Traffic Sign Prediction", img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(img):
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img

if __name__ == "__main__":
    main()
