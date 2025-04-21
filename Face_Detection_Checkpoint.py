import cv2
import streamlit as st

# Load the pre-trained Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier('C:/Users/djebb/Downloads/haarcascade_frontalface_default.xml')

def detect_faces(scale_factor, min_neighbors, color):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("""
    ### Instructions
    1. Ensure your webcam is connected.
    2. Adjust parameters as needed.
    3. Click "Detect Faces" to start real-time detection.
    4. Press 'q' to stop the detection.
    """)
    
    # Sliders for parameters
    scale_factor = st.slider("Scale Factor", 1.1, 2.0, 1.3, step=0.1)
    min_neighbors = st.slider("Min Neighbors", 1, 10, 5)
    color = st.color_picker("Pick Rectangle Color", "#00FF00")
    bgr_color = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))

    if st.button("Detect Faces"):
        detect_faces(scale_factor, min_neighbors, bgr_color)

if __name__ == "__main__":
    app()
