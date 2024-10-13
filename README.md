
**Gender Classification CNN with Images**
*This project was done during the academic studies
This project aims to classify gender based on handwriting samples using a Convolutional Neural Network (CNN). The model predicts the biological gender of the writer (0 for male, 1 for female) based on unique handwriting samples.

Data Description: The training data consists of .tiff images with corresponding labels in Excel files (train-c3-labels). The validation data is used for model validation, and the test data contains .tiff images without labels for final predictions.

Model Architecture: The CNN model consists of multiple convolutional and pooling layers, concluding with a dense layer with a sigmoid activation function to output the binary classification. The loss function used is binary_crossentropy, and accuracy is tracked as the evaluation metric.

Results: The model is evaluated on the test set, and predictions are saved in a CSV file .
