import gradio as gr
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np

model = tf.keras.models.load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('dummy_columns.pkl', 'rb') as f:
    dummy_columns = pickle.load(f)

def preprocess_input(input_data):
    df = pd.DataFrame([input_data], columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    df = pd.get_dummies(df, columns=['Geography', 'Gender'])
    for col in dummy_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[dummy_columns]
    scaled = scaler.transform(df)
    return scaled


def predict_churn(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    input_data = [CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    processed = preprocess_input(input_data)
    pred_prob = model.predict(processed)[0][0]
    churn = pred_prob > 0.5
    return f"Churn Probability: {pred_prob:.2f}", "Yes" if churn else "No"


inputs = [
    gr.Number(label="Credit Score", value=600),
    gr.Dropdown(choices=["France", "Spain", "Germany"], label="Geography", value="France"),
    gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Male"),
    gr.Number(label="Age", value=40),
    gr.Number(label="Tenure", value=3),
    gr.Number(label="Balance", value=60000),
    gr.Number(label="Number of Products", value=2),
    gr.Radio(choices=[0, 1], label="Has Credit Card", value=1),
    gr.Radio(choices=[0, 1], label="Is Active Member", value=1),
    gr.Number(label="Estimated Salary", value=50000)
]

outputs = [
    gr.Textbox(label="Churn Probability"),
    gr.Textbox(label="Churn Prediction (Yes/No)")
]

app = gr.Interface(fn=predict_churn, inputs=inputs, outputs=outputs, title="Customer Churn Prediction")

app.launch()
