from enum import Enum
from typing import Any, Union

import cv2
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from calibration import ArucoCalibrationConfig, ArucoCalibrationStrategy
from capture import KinectCapture, WebcamCapture


def render_frame(frame, frame_placeholder) -> DeltaGenerator:
    # Convert to RGB format for Streamlit and fill the placeholder
    st_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(st_frame, channels="RGB")
    return frame_placeholder


def calibrate_camera(capture: KinectCapture, frame_placeholder) -> DeltaGenerator:
    calibrator = ArucoCalibrationStrategy(ArucoCalibrationConfig())

    while True and not calibrator.is_finished():
        frame = capture.get_frame()
        if frame is not None:
            frame = calibrator.calibrate(frame)
            frame_placeholder = render_frame(frame, frame_placeholder)

    return frame_placeholder


def detection_loop(frame: Any, background: Any,algo: str, hardware_placeholder: DeltaGenerator) -> DeltaGenerator:
    import Kinect_detection_BF_CANY_Smooth
    import Kinect_detection_BF_DIFF_SameSize_Smooth
    import Kinect_detection_CNN_CANY_SameSize_Smooth
    import Kinect_detection_CNN_DIFF_SameSize_Smooth
    import Kinect_detection_Matcher_CANY_Smooth
    import Kinect_detection_Matcher_DIFF_SameSize_Smooth

    string_list = []

    if frame is not None:
        frame = frame.reshape((1080, 1920, 4))  # reshape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # convert to openCV format

        if algo == "Diff+Template":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_Matcher_DIFF_SameSize_Smooth.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)
            string_list = detector.average_materials.get_materials()

        if algo == "Diff+CNN":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_CNN_DIFF_SameSize_Smooth.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)
            string_list = detector.average_materials.get_materials()

        # Feature matching with background contour
        elif algo == "Diff+Feature":
            # get background
            # background = cv2.imread("background.jpg")

            # Initialize ObjectDetector
            detector = Kinect_detection_BF_DIFF_SameSize_Smooth.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(diff_frame, frame)
            string_list = detector.average_materials.get_materials()

        # Template matching with Canny contour
        elif algo == "Canny+Template":
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_Matcher_CANY_Smooth.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)
            string_list = detector.average_materials.get_materials()

        # Feature matching with Canny contour
        elif algo == "Canny+Feature":
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_BF_CANY_Smooth.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)
            string_list = detector.average_materials.get_materials()

        elif algo == "Canny+CNN":
            # get background
            # background = kinect.get_last_color_frame()
            # background = background.reshape((1080, 1920, 4))
            # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

            # Initialize ObjectDetector
            detector = Kinect_detection_CNN_CANY_SameSize_Smooth.ObjectDetector()
            detector.calibration()  # Perform calibration

            # remove background
            # diff_frame = cv2.absdiff(background, frame)

            # Process the frame to detect objects
            processed_frame = detector.process_frame(frame)
            string_list = detector.average_materials.get_materials()

        frame_placeholde = render_frame(frame, frame_placeholder)

        # Create a single Markdown string for the list
        markdown_content = "**Liste du matériels détectés:**\n"
        for item in string_list:
            markdown_content += f"- {item}\n"

        # Update the placeholder with the full Markdown content
        hardware_placeholder.markdown(markdown_content)

    return hardware_placeholder


################################################################################
# Streamlit Setup                                                              #
################################################################################


class ApplicationSteps(Enum):
    CALIBRATION = 1
    DETECTION = 2
    BACKGROUND = 3
    WAIT = 4
    STOP = 5


def main(capture: Union[KinectCapture, WebcamCapture]):
    application_step = ApplicationSteps.WAIT
    banner = st.empty()

    st.title("Détection automatique des pièces")

    if st.button("Calibration de la caméra"):
        application_step = ApplicationSteps.CALIBRATION

    if st.button("Prise du fond d'écran"):
        application_step = ApplicationSteps.BACKGROUND

    # select an algorithm
    algo = st.selectbox(
        "Choisir un algorithm",
        ("Diff+Template", "Diff+CNN", "Diff+Feature", "Canny+Template", "Canny+Feature", "Canny+CNN"),
    )

    # Placeholder for the video frame
    frame_placeholder = st.empty()
    hardware_placeholder = st.empty()
    # stop_button_pressed = st.button("Stop") # button to stop the stream

    # Run only once checked
    run = st.checkbox("Démarrer la detection")

    background = capture.background
    while True and application_step != ApplicationSteps.STOP:
        frame = capture.get_frame()

        if run:
            application_step = ApplicationSteps.DETECTION

        if application_step == ApplicationSteps.DETECTION:
            hardware_placeholder = detection_loop(frame, background, algo, hardware_placeholder)

        if application_step == ApplicationSteps.WAIT:
            frame_placeholder = render_frame(frame, frame_placeholder)

        if application_step == ApplicationSteps.CALIBRATION:
            with st.spinner("Début de la calibration"):
                frame_placeholder = calibrate_camera(capture, frame_placeholder)
            banner.success("Calibration complétée")
            application_step = ApplicationSteps.WAIT

        if application_step == ApplicationSteps.BACKGROUND:
            with st.spinner("Début de la prise de la photo"):
                background = capture.get_frame()
            banner.success("Le fond d'écran a bien été pris")
            application_step = ApplicationSteps.WAIT

        if cv2.waitKey(1) == ord("s"):
            application_step = ApplicationSteps.STOP

        # If press «esc» or hit stop button, end stream
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    # capture: KinectCapture = KinectCapture()
    capture: WebcamCapture = WebcamCapture(0)
    main(capture)
