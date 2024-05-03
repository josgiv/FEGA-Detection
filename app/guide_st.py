import streamlit as st

def main():
    """
    Main function to define the Streamlit app and its content.
    """
    st.title("FEGA - Face, Expression, Gender and Age Detection")
    st.subheader("User Guide - Age & Gender Recognition Project")

    # Table of Contents
    st.markdown("## Table of Contents")
    st.markdown("1. [Introduction](#introduction)")
    st.markdown("2. [How it Works](#how-it-works)")
    st.markdown("3. [Features](#features)")
    st.markdown("4. [Used Models](#used-models)")
    st.markdown("5. [Project Structure](#project-structure)")
    st.markdown("6. [Running the Application](#running-the-application)")

    # Introduction
    st.markdown("<a name='introduction'></a>", unsafe_allow_html=True)
    st.header("1. Introduction")
    st.write("This project is a web application developed in Python that utilizes facial recognition technology to predict the age, gender, and facial expression of a person in real-time. It serves as an interactive tool for users to analyze faces and obtain demographic information quickly and conveniently.")

    # How it Works
    st.markdown("<a name='how-it-works'></a>", unsafe_allow_html=True)
    st.header("2. How it Works")
    st.write("The application operates as follows:")
    st.markdown("- **2.1. Starting the Camera:** Users can start the camera to begin capturing images.")
    st.markdown("- **2.2. Brightness and Contrast Adjustment:** Users can adjust the brightness and contrast of the image using provided sliders.")
    st.markdown("- **2.3. Resize Intensity Adjustment:** Users can adjust the intensity of the video Resize operation for upscaling images using provided sliders.")
    st.markdown("- **2.4. Face Detection, Age, Facial Expression, and Gender Prediction:** The application detects faces in the image using a pre-trained face detection model. It then predicts the age, facial expression, and gender of the detected faces.")

    # Features
    st.markdown("<a name='features'></a>", unsafe_allow_html=True)
    st.header("3. Features")
    st.markdown("- **Start and stop the camera.**")
    st.markdown("- **Adjust the brightness and contrast of the image.**")
    st.markdown("- **Adjust the intensity of the video resize operation for upscaling images.**")
    st.markdown("- **Real-time face detection, age, facial expression, and gender prediction.**")

    # Used Models
    st.markdown("<a name='used-models'></a>", unsafe_allow_html=True)
    st.header("4. Used Models")
    st.subheader("**Age Model (`age_model`)**")
    st.write("- **Description**: This model is used to predict the age from facial images.")
    st.write("- **Files**: ")
    st.write("  - `age_deploy.prototxt`: Prototxt file used to deploy the model using the Caffe framework.")
    st.write("  - `age_net.caffemodel`: Model file containing trained weights for the age model.")
    st.write("  - `detection_matrix.txt`: File that may contain detection matrices for the face detection model.")
    st.subheader("**Face Detection Model (`face_detection_model`)**")
    st.write("- **Description**: This model is responsible for detecting faces in images.")
    st.write("- **Files**: ")
    st.write("  - `opencv_face_detector_uint8.pb`: TensorFlow file containing the computation graph for face detection using the Haar Cascade method.")
    st.write("  - `opencv_face_detector.pbtxt`: Text file containing the description of the computation graph for face detection.")
    st.subheader("**Face Expression Model (`face_expression_model`)**")
    st.write("- **Description**: This model is used to detect facial expressions such as happy, sad, angry, etc.")
    st.write("- **Files**: ")
    st.write("  - `haarcascade_frontalface_default.xml`: XML file containing the description of the face detection model using the Haar Cascade method.")
    st.subheader("**Gender Model (`gender_model`)**")
    st.write("- **Description**: This model is used to predict the gender from facial images.")
    st.write("- **Files**: ")
    st.write("  - `gender_deploy.prototxt`: Prototxt file used to deploy the model using the Caffe framework.")
    st.write("  - `gender_net.caffemodel`: Model file containing trained weights for the gender model.")
    st.write("  - `detection_matrix.txt`: File that may contain detection matrices for the face detection model.")
    st.write("  - `document.txt`: Additional documentation file that may contain important information about the usage of the models.")

    # Project Structure
    st.markdown("<a name='project-structure'></a>", unsafe_allow_html=True)
    st.header("5. Project Structure")
    st.write("```\n")
    st.write("app/")
    st.write("├── guide_st.py")
    st.write("├── model.py")
    st.write("pretrain_models/")
    st.write("| detection_matrix.txt")
    st.write("| document.txt")
    st.write("|   age_model/")
    st.write("├── age_deploy.prototxt")
    st.write("├── age_net.caffemodel")
    st.write("|   face_detection_model/")
    st.write("├── opencv_face_detector_uint8.pb")
    st.write("├── opencv_face_detector.pbtxt")
    st.write("|   face_expression_model/")
    st.write("└── haarcascade_frontalface_default.xml")
    st.write("|   gender_model/")
    st.write("├── gender_deploy.prototxt")
    st.write("├── gender_net.caffemodel")
    st.write("```\n")

    # Running the Application
    st.markdown("<a name='running-the-application'></a>", unsafe_allow_html=True)
    st.header("6. Running the Application")
    st.write("To run the application, ensure you have Python installed along with all required dependencies. You can install the dependencies by running the following command:")
    st.code("pip install -r requirements.txt", language="bash")
    st.write("Once the dependencies are installed, you can start the application by running the following command:")
    st.code("python app.py", language="bash")
    st.write("This will start the application and launch it in your default web browser. You can then interact with the application to perform age, gender, and facial expression recognition. For more details on how to use the application, refer to the user guide provided within the application.")

if __name__ == "__main__":
    main()
