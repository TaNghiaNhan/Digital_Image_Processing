import cv2
import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

def main():

    st.title("Real-time Emotion Detection")

    # Initialize face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Function to analyze frame
    def analyze_frame(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        emotion_label = ""

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            emotion_label = emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame, emotion_label

    # Function to process video with playback control
    def process_video(video_source):
        cap = cv2.VideoCapture(video_source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_placeholder = st.empty()
        emotion_placeholder = st.empty()
        frame_slider = st.slider("Frame", 0, total_frames-1, 0, 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
        ret, frame = cap.read()
        if ret:
            frame, emotion = analyze_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            emotion_placeholder.text(f"Emotion: {emotion}")
        
        cap.release()

    # Main app logic
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose input source:", ("Webcam", "Upload Video File"))

    if option == "Webcam":
        if st.sidebar.button("Start Webcam"):
            process_video(0)  # 0 is the default camera

    elif option == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            temp_file = "temp_video." + uploaded_file.name.split('.')[-1]
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            process_video(temp_file)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()