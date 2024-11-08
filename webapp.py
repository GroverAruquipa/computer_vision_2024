import streamlit as st
import cv2
import time
import Kinect_detection_BF_CANY
import Kinect_detection_BF_DIFF_SameSize
import Kinect_detection_Matcher_CANY
import Kinect_detection_Matcher_DIFF_SameSize
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

st.title("Détection automatique des pièces")

# select an algorithm
algo = st.selectbox(
    "Choisir un algorithm",
    ("Diff+template", "Diff+feature", "Canny+template", "Canny+feature"),
)

# Run only once checked
run = st.checkbox("Démarrer la vidéo")

# Placeholder for the video frame
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")  # button to stop the stream

# Initialize Kinect
kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
time.sleep(3)  # Enough time to let the Kinect power on

# Background for background difference methods
background = cv2.imread("background.jpg")

while True and run and not stop_button_pressed:
    if kinect.has_new_color_frame():
        # Get the color frame
        frame = kinect.get_last_color_frame()

    if frame is not None:
        frame = frame.reshape((1080, 1920, 4))  # reshape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # convert to openCV format

        # Template matching with background contour
        if algo == "Diff+template":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_Matcher_DIFF_SameSize.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)

        # Feature matching with background contour
        elif algo == "Diff+feature":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_BF_DIFF_SameSize.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)

        # Template matching with Canny contour
        elif algo == "Canny+template":
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_Matcher_CANY.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)

        # Feature matching with Canny contour
        elif algo == "Canny+feature":
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_BF_CANY.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)

        frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format for Streamlit
        frame_placeholder.image(frame, channels="RGB")  # Fill empty placeholder with the camera frame using st.image

    # If press «esc» or hit stop button, end stream
    if cv2.waitKey(1) == 27 or stop_button_pressed:
        break
