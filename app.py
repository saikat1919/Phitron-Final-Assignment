import gradio as gr
import pandas as pd
import numpy as np
import pickle

with open("RandomForest.pkl", "rb") as file:
    RandomForest = pickle.load(file)

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_df = pd.DataFrame(
        [[Pregnancies,
         Glucose,
         BloodPressure,
         SkinThickness,
         Insulin,
         BMI,
         DiabetesPedigreeFunction,
         Age]],
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    )

    predictions = RandomForest.predict(input_df)
    return "Diabetic" if predictions[0] == 1 else "Not Diabetic"

inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="BloodPressure(mm Hg)", value=70),
    gr.Number(label="SkinThickness(mm)", value=30),
    gr.Number(label="Insulin(mu U/ml)", value=30),
    gr.Number(label="BMI(kg/m2)", value=30),
    gr.Number(label="DiabetesPedigreeFunction", value=0),
    gr.Number(label="Age", value=30),
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs='text',
    title="Diabetes Prediction",
)

app.launch()
