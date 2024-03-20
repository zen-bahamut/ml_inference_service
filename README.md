PAWS Sequence Classification
Overview
This project involves a machine learning application for sequence classification using the PAWS (Paraphrase Adversaries from Word Scrambling) dataset. It employs a DistilBert model for evaluating sentence similarities. The Flask API serves model inferences, and the setup is containerized using Docker for ease of deployment.

Features
Model Training: Train a DistilBert model on the PAWS dataset.
Inference API: A Flask application to serve predictions from the trained model.
Evaluation: Script to evaluate the model performance on a test dataset.
Requirements
Docker
Python 3.x
Libraries: datasets, Flask, gunicorn, transformers, torch, scikit-learn (See requirements.txt for versions)
Quick Start
Set up the environment:
Install Docker and Docker Compose.
Ensure Python is installed with the necessary libraries.
Train the model:
Run python train_model.py to train the model and save the artifacts.
Build and run the Docker container:
Use docker-compose up to build the Flask app container and start the service.
Evaluate the model:
Run python evaluate.py to test the model's performance on the PAWS dataset.
API Usage
Send a POST request to http://localhost:8080/infer with JSON payload: {"texts": ["sentence1", "sentence2"]}.
The API returns the classification predictions.
Project Structure
Dockerfile and docker-compose.yml: Docker configuration files.
requirements.txt: Python dependencies.
train_model.py: Script for training the model.
evaluate.py: Script for evaluating the model.
main.py: Flask application for serving model inferences.