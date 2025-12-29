📡 Classification of Channel Encoders Using CNN

This project focuses on classifying channel encoding techniques using a Convolutional Neural Network (CNN).
It integrates Machine Learning (Deep Learning) with a Django web application to provide an end-to-end solution for uploading data, running predictions, and viewing results.

## Project Overview

Channel encoding plays a crucial role in reliable digital communication systems.
This project uses a trained CNN model to automatically identify different channel encoders from input data, improving accuracy and reducing manual analysis.

## Technologies Used

Python
TensorFlow / Keras
Convolutional Neural Networks (CNN)
Django (Web Framework)
NumPy
SQLite
HTML / CSS

## Project Structure

CHANNEL ENCODERS USING CNN/
│
├── Abstract/                     # Project abstract
├── BASE PAPER/                   # Reference research papers
│
├── code/
│   ├── admins/                   # Django admin configurations
│   ├── assets/                   # Static files (CSS, JS, images)
│   ├── ChannelEncodersCNN/       # Main Django project files
│   ├── media/                    # Uploaded input files
│   ├── users/                    # User and prediction modules
│   │
│   ├── cnn_fec_classifier.h5     # Trained CNN model
│   ├── db.sqlite3                # Database
│   ├── manage.py                 # Django management file
│   └── req.txt                   # Required Python packages
│
└── documentation/                # Reports, diagrams, screenshots

## How the System Works

User logs into the Django web application
Input data is uploaded
The trained CNN model processes the data
Channel encoder type is predicted
Results are displayed on the web interface
Prediction data is stored in the database

## How to Run the Project

  python manage.py runserver

## Model Details

Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Model File: cnn_fec_classifier.h5
Purpose: Classify channel encoder types accurately

## Applications

Digital Communication Systems
Error Control Coding Analysis
AI-based Signal Processing
Academic & Research Projects

## Key Features

Deep Learning based classification
User-friendly web interface
Real-time prediction
Database integration
Scalable and modular design

## Future Enhancements

Support for more encoding schemes
Performance optimization
Cloud deployment
Real-time signal processing
Improved UI/UX

## Author

Vijay Kumar
B.Tech – Computer Science Engineering
AI / ML Enthusiast
