import json
import sys

def load_data(file_name):
    """
    Load data from a JSON file.

    Args:
        file_name (str): The name of the JSON file to load.

    Returns:
        list: A list of dictionaries containing the data from the JSON file.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def get_best_hyperparameters(data):
    """
    Find the hyperparameters with the highest test_accuracy.

    Args:
        data (list): A list of dictionaries containing hyperparameters and results.

    Returns:
        dict: The hyperparameters of the record with the highest test_accuracy.
    """
    best_record = None
    highest_accuracy = -1
    loss = -1

    for record in data:
        if record['results']['test_accuracy'] > highest_accuracy:
            highest_accuracy = record['results']['test_accuracy']
            best_record = record
            loss = record['results']['test_loss']

    return (best_record['hyperparameters'] if best_record else None), highest_accuracy, loss

# Check if a file name is provided as a command line argument
if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    print("Usage: python analyze_training.py <file_name>")
    sys.exit(1)
data = load_data(file_name)
best_hyperparameters, acuraccy, loss = get_best_hyperparameters(data)

print("Acuraccy:", acuraccy)
print("Loss:", loss)
print("Parameters:", best_hyperparameters)