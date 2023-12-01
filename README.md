# Spam_Classification_Deep_Learning (ANN)


## Overview

This project involves building a machine learning model for spam classification using a neural network. The task is to identify whether a given SMS message is spam or not spam. The project includes data preprocessing, model building, training, and evaluation.

## Project Structure

- **1. Data Preparation:**
  - Load and preprocess the spam classification dataset.
  - Tokenize and lemmatize text using NLTK.
  - Convert text data into TF-IDF vectors.

- **2. Model Building:**
  - Construct a neural network model using TensorFlow and Keras.
  - Set hyperparameters for the number of classes and hidden units.
  - Compile the model with categorical crossentropy loss and accuracy metric.

- **3. Model Training:**
  - Train the model using the training dataset.
  - Monitor training progress and visualize accuracy improvements over epochs.
  - Evaluate the model on a test dataset.

- **4. Prediction:**
  - Use the trained model to make predictions for new SMS messages.
  - Convert input text into TF-IDF vectors for prediction.

## Files

- `spam_classification.ipynb`: Jupyter Notebook containing the entire code for the project.
- `Spam-Classification.csv`: Dataset used for training and testing the model.

## Dependencies

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, nltk, tensorflow, matplotlib

## Usage

1. Install the required dependencies using the following command:


2. Run the Jupyter Notebook (`spam_classification.ipynb`) to execute the project.

## Results

The model achieves [insert accuracy or other metric] on the test dataset, demonstrating its effectiveness in spam classification.

## Future Improvements

- Fine-tune hyperparameters for better performance.
- Experiment with different neural network architectures.
- Collect more diverse data for training.

## Credits

This project is created by [Your Name]. Feel free to contact for any questions or improvements.

## License

This project is licensed under the [MIT License](LICENSE).

