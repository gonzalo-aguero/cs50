import os
import time
import sys
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()
    
    start_time = time.time()
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    end_time = time.time()
    
    print(f"Entrenamiento completado en {end_time - start_time} segundos.")

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_dir):
            img_path = os.path.join(category_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))            
            images.append(img)
            labels.append(category)
            
    print("Carga de imagenes completada.")
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([
        # Convolutional layer. It will learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 2x2 pool size
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def visualize_filters(model):
    """
    Visualizes the filters of the given model's first layer.
    Parameters:
    - model: A neural network model.
    Returns: None
    This function displays the filters of the first layer of the given model. It normalizes the weights for visualization and converts them to grayscale images. Each filter is displayed for 500 milliseconds.
    Note: This function requires the OpenCV library to be installed.
    """
    weights, biases = model.layers[0].get_weights()

    # Normalize the weights for visualization
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    # Number of filters
    num_filters = weights.shape[-1]

    # Create a window to display the filters
    cv2.namedWindow('Filters', cv2.WINDOW_NORMAL)

    for i in range(num_filters):
        # Extract the filter and convert to 8-bit grayscale
        filter_img = (weights[:, :, 0, i] * 255).astype(np.uint8)

        # Convert grayscale to BGR
        filter_img_bgr = cv2.cvtColor(filter_img, cv2.COLOR_GRAY2BGR)

        # Display the filter
        cv2.imshow('Filters', filter_img_bgr)
        cv2.waitKey(500)  # Display each filter for 500 milliseconds

    # Destroy the window after displaying all filters
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
