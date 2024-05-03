#model.py

import cv2
import os
import subprocess
import threading
import numpy as np

from deepface import DeepFace
import streamlit as st

# Function to run the guide_st.py script asynchronously
def run_guide_script():
    script_path = os.path.join(os.path.dirname(__file__), "guide_st.py")
    subprocess.run(["streamlit", "run", script_path])

def run_guide_async():
    guide_thread = threading.Thread(target=run_guide_script)
    guide_thread.start()

# Setup Streamlit sidebar
st.sidebar.title("Navigation Sidebar")
st.sidebar.markdown("---")

#  "Home" button for reloading model.py
if st.sidebar.button("🏠 Home", key="home"):
    st.experimental_rerun()

# Button for load user guide page
guide_button = st.sidebar.button("📖 Open User Guide", key="open_guide")
if guide_button:
    run_guide_async()

# Status log on sidebar
st.sidebar.markdown("---")
st.sidebar.text("Status:")
st.sidebar.text("Ready")

global dominant_emotion
global dominant_emotion_text
global emotion_result

# Global variables
camera_started = False
brightness = 0
contrast = 0
resize_intensity = 1.0  # Default intensity
dominant_emotion_text = "No faces detected."

# Function to adjust brightness and contrast
def adjust_brightness_contrast(frame, brightness, contrast):
    global dominant_emotion_text
    frame = frame.astype(np.float32)
    frame = frame + brightness
    frame = frame * (contrast / 127 + 1) - contrast
    frame = np.clip(frame, 0, 255)
    frame = frame.astype(np.uint8)
    return frame

# Function to detect faces and draw bounding boxes
def detect_faces(faceNet, frame):
    global dominant_emotion_text
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bounding_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bounding_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bounding_boxes

# Function to upscale frame
def upscale_frame(frame, scale_factor):
    # Upscale the frame using cv2.resize()
    upscaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    return upscaled_frame

# Function to load age and gender prediction models
def load_opencv_models():
    model_directory = os.path.dirname(os.path.realpath(__file__))
    face_model = os.path.join(model_directory, "..", "pretrain_models", "face_detection_model", "opencv_face_detector_uint8.pb")
    face_proto = os.path.join(model_directory, "..", "pretrain_models", "face_detection_model", "opencv_face_detector.pbtxt")
    age_model = os.path.join(model_directory, "..", "pretrain_models", "age_model", "age_net.caffemodel")
    age_proto = os.path.join(model_directory,"..", "pretrain_models", "age_model", "age_deploy.prototxt")
    gender_model = os.path.join(model_directory,"..", "pretrain_models", "gender_model", "gender_net.caffemodel")
    gender_proto = os.path.join(model_directory, "..", "pretrain_models", "gender_model", "gender_deploy.prototxt")
    face_net = cv2.dnn.readNet(face_model, face_proto)
    age_net = cv2.dnn.readNet(age_model, age_proto)
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    return face_net, age_net, gender_net

# Function to load facial expression detection model
def load_expression_model():
    model_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(model_directory, "..", "pretrain_models", "face_expression", "haarcascade_frontalface_default.xml")
    model = cv2.CascadeClassifier(model_path)
    return model

# Function to predict age and gender using OpenCV models
def predict_age_gender_cv(face, age_net, gender_net):
    # Preprocess the face image
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    # Predict gender
    gender_net.setInput(blob)
    gender_pred = gender_net.forward()
    gender = ['Male', 'Female'][gender_pred[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_pred = age_net.forward()
    age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'][age_pred[0].argmax()]

    return age, gender

# Function to predict facial expressions
def predict_facial_expression(face):
    global dominant_emotion
    global dominant_emotion_text
    global emotion_result
    global dominant_emotion_facebox
    global dynamic_dominant_result_text
    global dynamic_emotion_result
    
    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
    print("Result:", result)
    
    # Check if the result contains 'emotion' key
    if 'emotion' in result:
        emotion_result = result['emotion']
        # Check if 'emotion' is a dictionary
        if isinstance(emotion_result, dict):
            dominant_emotion = max(emotion_result, key=emotion_result.get)
            dominant_emotion_text = f"Dominant Emotion: {dominant_emotion} ({emotion_result[dominant_emotion]:.2f}%)"
        else:
            dominant_emotion_text = "No faces detected."
    else:
        dominant_emotion_text = "No faces detected."

    dominant_emotion_facebox = result[0]['dominant_emotion']
    dynamic_emotion_result = result[0]['emotion']

# Main function
def main():
    global camera_started, brightness, contrast, resize_intensity, dominant_emotion_text, dominant_emotion, emotion_result
    
    st.title('Age, Gender & Facial Expression Recognition')
    st.write('This Machine Learning project is a web application that utilizes facial recognition technology to predict the age, gender, and facial expression of a person in real-time. It leverages pre-trained machine learning models and OpenCV technology for face detection in images. The application allows users to access it through a web interface and view predictions directly.')

    # Load OpenCV models    
    face_net, age_net, gender_net = load_opencv_models()
    expression_model = load_expression_model()

    # Get available camera devices
    num_camera_devices = 2  # Adjust this number according to your system
    available_camera_devices = list(range(num_camera_devices))

    # Display UI elements
    col1, col2 = st.columns([1, 1])

    if not camera_started:  
        start_button = col1.button('Start Camera', help="Start the camera")
        if start_button:
            camera_started = True

    brightness = col2.slider("Brightness", min_value=-100, max_value=100, value=brightness, step=1)
    contrast = col2.slider("Contrast", min_value=-100, max_value=100, value=contrast, step=1)
    resize_intensity = col2.slider("Resize Intensity", min_value=1.0, max_value=3.0, value=resize_intensity, step=0.1)

    if camera_started:
        stop_button = col1.button('Stop Camera', help="Stop the camera")
        if stop_button:
            camera_started = False

    if camera_started:
        st.write("Camera is running...")
        selected_camera = st.selectbox("Select Camera", available_camera_devices, index=0)
        cap = cv2.VideoCapture(selected_camera)

        padding = 20
        video_stream = st.empty()
        fps_text = st.empty()
        emotion_text = st.empty()

        # Initialize frame rate calculation
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()

        while camera_started:
            t1 = cv2.getTickCount()

            ret, frame = cap.read()
            if not ret:
                st.error("Error reading frame from camera!")
                break

            frame = adjust_brightness_contrast(frame, brightness, contrast)

            frame_with_boxes, bboxs = detect_faces(face_net, frame)

            
            for bbox in bboxs:
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                             max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                
                age, gender = predict_age_gender_cv(face, age_net, gender_net)
                predict_facial_expression(face)

                label = f"{gender}, {age}"
                cv2.rectangle(frame_with_boxes, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
                cv2.putText(frame_with_boxes, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Draw the dominant emotion text inside the face box
                cv2.putText(frame_with_boxes, dominant_emotion_facebox, (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            frame_with_boxes = upscale_frame(frame_with_boxes, resize_intensity)

            frame_bytes = cv2.imencode('.jpg', frame_with_boxes)[1].tobytes()
            video_stream.image(frame_bytes, channels="BGR")

            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1

            fps_text.text(f"Current FPS: {frame_rate_calc:.2f}")
            
            # Display the dominant emotion dynamically below the current FPS
            emotion_text.text(f"Dominant emotion : {dominant_emotion_facebox}")

        cap.release()

if __name__ == '__main__':
    main()
