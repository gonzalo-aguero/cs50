## EXPERIENCE
At first, I based my work on the structure of the digits/handwriting.py model, but the results were not good enough. Without knowing where to start, I began by changing the values of some hyperparameters, like the size of the filter kernels, the number of filters, the number of dense layers, and the number of neurons in each one.

To make a better comparison as I tested different hyperparameters, I wrote the necessary code to save the hyperparameters of interest and the results (accuracy, loss, and training time) in a JSON file. This allowed me to choose the values that gave me better results.

During testing, I faced the following problems:
- Big differences between the accuracy achieved with the training set and the evaluation set. Changing the dropout value helped me get better results with this issue.
- Accuracy too low: Increasing the number of neurons per layer helped improve accuracy a lot without a significant increase in training time. On the other hand, changing the number of filters and the kernel size barely produced improvements, with a big increase in training time.

## NOTES
- For each test, save the hyperparameters and the results. Store this information in separate files based on the structure of the neural network.
- Test the model with images (run_model.py).
- Open and display images from the obtained category to check if it is correct.
- Analyze the results.

## RUN
- When using venv: Don’t use the "python" alias, use the python3.12 binary name instead.
