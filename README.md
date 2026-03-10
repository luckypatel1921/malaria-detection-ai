# Malaria_Detection_Project
Malaria Detection using Deep Learning
# 🧬 AI Malaria Detection System

This project detects malaria parasites in blood cell images using Deep Learning (CNN).

## Features
- Upload blood smear image
- AI detects infected / uninfected cells
- Real-time analysis
- Streamlit dashboard interface

## Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy

## Files

app.py → Streamlit web application  
train_model.py → CNN model training  
predict.py → Batch image prediction  
malaria_model.h5 → Trained deep learning model  

## How to Run

Install dependencies

pip install streamlit tensorflow numpy pillow opencv-python

Run the app

streamlit run app.py

## Dataset

NIH Malaria Dataset containing parasitized and uninfected blood cell images.

## Output

AI predicts whether the blood cell is infected with malaria or healthy.
