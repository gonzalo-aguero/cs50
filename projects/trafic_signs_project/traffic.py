import json
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

LEARNING_RATE = 0.001
FILTERS = 32
KERNEL_SIZE = 3
DENSE1 = 256
DENSE2 = 256
DENSE3 = 256
DROPOUT = 0.15


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print(f"Set of {len(images)} images.")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=EPOCHS)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Entrenamiento completado en {training_time} segundos.")

    # Evaluate neural network performance
    test_loss, test_accuracy = model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
    
    save_log_data(history, training_time, test_accuracy, test_loss, "training_log_(FilFlaDenDenDenDrop).json")


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
            filters=FILTERS, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(DENSE1, activation="relu"),
        tf.keras.layers.Dense(DENSE2, activation="relu"),
        tf.keras.layers.Dense(DENSE3, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
    
    
def save_log_data(history, training_time, test_accuracy, test_loss, log_file='training_log.json'):
    """
    Saves the log data from training and testing into a JSON file.
    Parameters:
    - history (tf.keras.callbacks.History): The history object containing the training history.
    - test_accuracy (float): The accuracy of the model on the test dataset.
    - test_loss (float): The loss of the model on the test dataset.
    - log_file (str): The path to the JSON file where the log data will be saved. Default is 'training_log.json'.
    Returns: None
    """
    # Cargar el contenido existente del archivo o inicializar un array vacío
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            try:
                log_data = json.load(file)
            except json.JSONDecodeError:
                log_data = []
    else:
        log_data = []

    # Extraer los hiperparámetros y resultados del historial de entrenamiento
    new_log_entry = {
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            'epochs': EPOCHS,
            'filters': FILTERS,
            'kernel_size': KERNEL_SIZE,
            'dense1': DENSE1,
            'dense2': DENSE2,
            'dense3': DENSE3,
            'dropout': DROPOUT
        },
        "results": {
            "accuracy": history.history['accuracy'][-1],
            "loss": history.history['loss'][-1],
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            "training_time": round(training_time, 3)
        }
    }
    log_data.append(new_log_entry)

    # Escribir el array completo de nuevo en el archivo
    with open(log_file, 'w') as file:
        json.dump(log_data, file, indent=4)
    
        
if __name__ == "__main__":
    main()
