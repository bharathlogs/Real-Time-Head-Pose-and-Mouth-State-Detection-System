Real-Time Head Pose, Eye State, and Mouth State Detection

This project performs real-time facial analysis using MediaPipe FaceMesh and OpenCV.
It detects:

Head pose (Up, Down, Left, Right, Forward)
Mouth state (Closed, Slightly Open, Wide Open)
Face mesh visualization with annotated landmarks
The system runs on a standard webcam and processes frames in real time.

1. Features
Head Pose Detection
Uses 3D-to-2D landmark projection + cv2.solvePnP to estimate rotation (pitch, yaw, roll).

Mouth State Detection
Uses upper and lower lip landmark distance to determine mouth opening.

Built Modules
fp_helper.py — head pose pipeline
ms_helper.py — mouth state module
g_helper.py — image utilities
app.py — main real-time execution script

2. Requirements
Install dependencies using the provided requirements file.

Install:
pip install -r requirements.txt

Key Versions
MediaPipe 0.10.14
OpenCV 4.8+ / 4.9 contrib
NumPy 1.26
Matplotlib 3.8 (optional)

3. Running the Application
After installation:

Start the real-time detection:
python app.py

Controls

Press q or ctrl+c to quit the application.

4. Project Structure
realtime-head-pose-detection-master/
│
├── app.py                 # Main real-time detection script
├── fp_helper.py           # Head pose computation module
├── ms_helper.py           # Mouth state module
├── g_helper.py            # Image processing utilities
├── requirements.txt       # Dependency list
├── assets/                # Reference images for documentation
└── README.md              # Project documentation

5. How It Works
MediaPipe FaceMesh

Provides 468 facial landmarks.
Eye and mouth states are computed by measuring vertical distances between key landmarks.

Head Pose
Extract specific 3D landmark coordinates
Convert normalized points → pixel coordinates
Apply solvePnP to estimate rotation vectors
Convert to Euler angles and categorize movement

6. Notes
Works with any standard webcam
Performance depends on CPU/GPU capability
Lighting conditions influence accuracy
To add multi-face tracking, update max_num_faces in app.py

7. Future Enhancements

Eye State Detection
Multi-face recognition
Smoother temporal filtering for drowsiness
Alert system integration
Export logs for analytics