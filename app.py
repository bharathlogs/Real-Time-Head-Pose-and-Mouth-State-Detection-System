import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
import mediapipe as mp

from g_helper import bgr2rgb, rgb2bgr, mirrorImage
from fp_helper import pipelineHeadTiltPose, draw_face_landmarks_fp
from ms_helper import pipelineMouthState


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# ------------------------------
# WebRTC Video Processor
# ------------------------------
class Processor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = mirrorImage(img)

        results = self.face_mesh.process(bgr2rgb(img))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_face_landmarks_fp(img, face_landmarks)

                head_pose = pipelineHeadTiltPose(img, face_landmarks)
                mouth_state = pipelineMouthState(img, face_landmarks)

                cv2.putText(img, f"Head Pose: {head_pose}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Mouth: {mouth_state}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Real-Time Head Pose & Mouth State Detection")

st.write("WebRTC camera → Mediapipe FaceMesh → Real-time inference")

webrtc_streamer(
    key="realtime-headpose",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
