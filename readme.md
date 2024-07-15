# Age, Gender & Facial Expression Recognition

This project is a web application developed in Python that utilizes facial recognition technology to predict the age, gender, and facial expression of a person in real-time. It leverages pre-trained machine learning models and OpenCV technology for face detection in images. The application allows users to access it through a web interface and view predictions directly.

## Running the Application

To run the application, ensure you have Python installed along with all required dependencies. You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, you can start the application by running the following command:

```bash
python app.py
```

## Table of Contents
1. [Introduction](#introduction)
2. [How it Works](#how-it-works)
3. [Features](#features)
4. [Used Models](#used-models)
5. [Project Structure](#project-structure)
6. [Running the Application](#running-the-application)

## Introduction

This project is designed to predict age, gender, and facial expression from images or video streams in real-time. It serves as an interactive tool for users to analyze faces and obtain demographic information quickly and conveniently.

## How it Works

The application operates as follows:

1. **Starting the Camera**: Users can start the camera to begin capturing images.
2. **Brightness and Contrast Adjustment**: Users can adjust the brightness and contrast of the image using provided sliders.
3. **Resize Intensity Adjustment**: Users can adjust the intensity of the video Resize operation for upscaling images.
4. **Face Detection, Age, Facial Expression and Gender Prediction**: The application detects faces in the image using a pre-trained face detection model. It then predicts the age, facial expression and gender of the detected faces.

## Features

- Start and stop the camera.
- Adjust the brightness and contrast of the image.
- Adjust the intensity of the video resize operation for upscaling images.
- Real-time face detection, age, and gender prediction.

## Used Models

### 1. Age Model (`age_model`)
   - **Description:** This model is used to predict the age from facial images.
   - **Files:** 
     - `age_deploy.prototxt`: Prototxt file used to deploy the model using the Caffe framework.
     - `age_net.caffemodel`: Model file containing trained weights for the age model.

### 2. Face Detection Model (`face_detection_model`)
   - **Description:** This model is responsible for detecting faces in images.
   - **Files:**
     - `opencv_face_detector_uint8.pb`: TensorFlow file containing the computation graph for face detection using the Haar Cascade method.
     - `opencv_face_detector.pbtxt`: Text file containing the description of the computation graph for face detection.

### 3. Face Expression Model (`face_expression`)
   - **Description:** This model is used to detect facial expressions such as happy, sad, angry, etc.
   - **Files:**
     - `haarcascade_frontalface_default.xml`: XML file containing the description of the face detection model using the Haar Cascade method.

### 4. Gender Model (`gender_model`)
   - **Description:** This model is used to predict the gender from facial images.
   - **Files:**
     - `gender_deploy.prototxt`: Prototxt file used to deploy the model using the Caffe framework.
     - `gender_net.caffemodel`: Model file containing trained weights for the gender model.

### 5. Additional Files
   - `detection_matrix.txt`: File that may contain detection matrices for the face detection model.
   - `document.txt`: Additional documentation file that may contain important information about the usage of the models.

## Project Structure

- app/
    - guide_st.py
    - model.py
- pretrain_models/
    - detection_matrix.txt
    - document.txt
    - age_model/
        - age_deploy.prototxt
        - age_net.caffemodel
    - face_detection_model/
        - opencv_face_detector_uint8.pb
        - opencv_face_detector.pbtxt
    - face_expression_model/
        - haarcascade_frontalface_default.xml
    - gender_model/
        - gender_deploy.prototxt
        - gender_net.caffemodel
- app.py
- requirements.txt

This will start the application and launch it in your default web browser. You can then interact with the application to perform age, gender, and facial expression recognition.
For more details on how to use the application, refer to the user guide provided within the application.# FEGA-Detection
# FEGA-Detection
