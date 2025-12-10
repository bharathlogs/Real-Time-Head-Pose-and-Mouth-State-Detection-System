import streamlit as st
import numpy as np
import cv2

st.title("Head Pose & Mouth State - Image mode")

uploaded = st.camera_input("Take a photo or upload an image")

if uploaded is None:
    st.info("No image received. Use 'Take photo' or upload an image.")
else:
    # read image bytes -> OpenCV BGR
    image_bytes = uploaded.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Failed to decode image.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_column_width=True)
        # Run your detection function here using `img` (BGR) or `img_rgb` (RGB)
        # Example placeholder:
        # head_pose, mouth_state = detect_pose_and_mouth(img)
        # st.write(head_pose, mouth_state)

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from g_helper import bgr2rgb, rgb2bgr, mirrorImage
from fp_helper import pipelineHeadTiltPose, draw_face_landmarks_fp
from ms_helper import pipelineMouthState

# Initiate Camera
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          continue

        # Mirror image (Optional)
        image = mirrorImage(image)

        # Generate face mesh
        results = face_mesh.process(bgr2rgb(image))

        # Processing Face Landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # FACE MESH ----------------------------------------
                draw_face_landmarks_fp(image, face_landmarks)

                # HEAD TILT POSE -----------------------------------
                head_tilt_pose = pipelineHeadTiltPose(image, face_landmarks)

                # MOUTH STATE --------------------------------------
                mouth_state = pipelineMouthState(image, face_landmarks)
        # Show Image
        cv2.imshow('Face Mesh', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
