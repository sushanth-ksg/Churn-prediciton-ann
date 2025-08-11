# Customer Churn Prediction

## Project Overview

This project builds a deep learning model to predict customer churn using a public dataset. The model is trained with TensorFlow and includes preprocessing steps like one-hot encoding and feature scaling. It is deployed as an interactive web app using Gradio and hosted on Hugging Face Spaces.

## Features

- Categorical encoding with `pandas.get_dummies`  
- Feature scaling using `StandardScaler`  
- Neural network with 2 hidden layers and sigmoid output layer  
- Model and preprocessing objects saved for reuse  
- Easy-to-use web interface hosted on Hugging Face Spaces  

## Workflow

- Importing Libraries: Necessary libraries such as NumPy, Pandas, TensorFlow, and Keras are imported.
- Data Preprocessing: The dataset is loaded, and data preprocessing steps include handling categorical data, label encoding, one-hot encoding, splitting the dataset, and feature scaling.
- Building the ANN: A Sequential model is created using TensorFlow and Keras. The model architecture consists of an input layer, two hidden layers with ReLU activation, and an output layer with sigmoid activation.
- Training the ANN: The model is compiled using the Adam optimizer and binary crossentropy loss. It is then trained on the training set for 100 epochs.
- Making Predictions and Evaluating the Model: Predictions are made on the test set, and the model's performance is evaluated using a confusion matrix and accuracy score.
