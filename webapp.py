import os
import streamlit as st
import cv2
import time
import Kinect_detection_BF_CANY_Smooth
import Kinect_detection_BF_DIFF_SameSize_Smooth
import Kinect_detection_Matcher_CANY_Smooth
import Kinect_detection_Matcher_DIFF_SameSize_Smooth
import Kinect_detection_CNN_DIFF_SameSize_Smooth
import Kinect_detection_CNN_CANY_SameSize_Smooth
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

from src.application.calibration import CalibrationStep
from src.application.render import RenderStep
from src.domain.calibration.aruco import ArucoCalibrationStrategy
from src.domain.context import ArucoCalibrationConfig, GrayscaleFilterConfig, PipelineContext, RenderConfig
from src.domain.image_filter.filters import GrayscaleFilter


banner = st.empty()
st.title("Détection automatique des pièces")

if st.button("Calibration de la caméra"):
    with st.spinner("Début de la prise de la photo"):
        calibrate_camera()
    banner.success("Le fond d'écran a bien été pris")

if st.button("Prise du fond d'écran"):
    with st.spinner("Début de la prise de la photo"):
        take_background()
    banner.success("Le fond d'écran a bien été pris")

# select an algorithm
algo = st.selectbox(
    "Choisir un algorithm",
    ("Diff+Template", 'Diff+CNN',"Diff+Feature", "Canny+Template", "Canny+Feature", "Canny+CNN"),
)

# Run only once checked
run = st.checkbox("Démarrer la vidéo")

# Placeholder for the video frame
frame_placeholder = st.empty()
hardware_placeholder = st.empty()
#stop_button_pressed = st.button("Stop") # button to stop the stream

# Initialize Kinect
kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
time.sleep(3)  # Enough time to let the Kinect power on

# Background for background difference methods
background = cv2.imread("assets/background.jpg")


def render_frame(frame):
    # Convert to RGB format for Streamlit and fill the placeholder
    st_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(st_frame,channels="RGB")


def calibrate_camera():

    context = PipelineContext()
    calibrator = CalibrationStep(ArucoCalibrationStrategy(
        ArucoCalibrationConfig(type='aruco'), GrayscaleFilter()
    ))
    render = RenderStep(RenderConfig(type='render', show=False))
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    time.sleep(3)

    while True and context.calibration.calibration is None:
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()

            if frame is not None:
                frame = frame.reshape((1080, 1920, 4))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                context.capture.frame = frame
                context.capture.viz_frame = frame
                context = calibrator.execute(context)
                context = render.execute(context)
                render_frame(context.capture.viz_frame)

    calibrator.cleanup()
    render.cleanup()


def take_background():
    path = 'assets/'
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    time.sleep(3)  # Enough time to let the Kinect power on
    # Get the background
    background = kinect.get_last_color_frame()
    background = background.reshape((1080, 1920, 4))
    background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(os.path.join(path, 'background.jpg'), background)


while True and run:
    if kinect.has_new_color_frame():
        # Get the color frame
        frame = kinect.get_last_color_frame()

    string_list = []

    if frame is not None:
        frame = frame.reshape((1080, 1920, 4)) # reshape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # convert to openCV format

        if algo=="Diff+Template":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_Matcher_DIFF_SameSize_Smooth.ObjectDetector()
            detector.calibration() # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)
            string_list = detector.average_materials.get_materials()

        
        if algo=="Diff+CNN":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_CNN_DIFF_SameSize_Smooth.ObjectDetector()
            detector.calibration() # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)
            string_list = detector.average_materials.get_materials()

        # Feature matching with background contour
        elif algo=='Diff+Feature':
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_BF_DIFF_SameSize_Smooth.ObjectDetector()
            detector.calibration() # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)
            string_list = detector.average_materials.get_materials()

        # Template matching with Canny contour
        elif algo=='Canny+Template':
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_Matcher_CANY_Smooth.ObjectDetector()
            detector.calibration() # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)
            string_list = detector.average_materials.get_materials()

        # Feature matching with Canny contour
        elif algo=='Canny+Feature':
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_BF_CANY_Smooth.ObjectDetector()
            detector.calibration() # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)
            string_list = detector.average_materials.get_materials()

        elif algo=='Canny+CNN':
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_CNN_CANY_SameSize_Smooth.ObjectDetector()
            detector.calibration() # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)  
            string_list = detector.average_materials.get_materials()          


        render_frame(frame)

        # Create a single Markdown string for the list
        markdown_content = "**Liste du matériels détectés:**\n"
        for item in string_list:
            markdown_content += f"- {item}\n"

        # Update the placeholder with the full Markdown content
        hardware_placeholder.markdown(markdown_content)

    # If press «esc» or hit stop button, end stream
    if cv2.waitKey(1) == 27:
        break
