# README

# Étape 1 : Calibration de la caméra

# 1. Calibration avec une pièce prédéterminée :
#    - Pour commencer la calibration de la caméra, commentez les lignes 200 et 201 dans le code.
#    - Exécutez le programme.
#    - Prenez note de la largeur et de la longueur de la pièce de calibration.
#    - Décommentez les lignes 200 et 201, puis indiquez les valeurs de largeur et de longueur sur les lignes 197 et 198.

# Étape 2 : Exécution du programme

# Fonctionnement du programme
# Le programme utilise la détection de contours via l'algorithme Canny pour identifier les contours des objets
# placés sur le plan de travail. Lorsqu'un contour est détecté, le programme mesure le plus petit rectangle
# englobant cet objet. 

# Ce rectangle est ensuite comparé avec les références initialisées dans la méthode `register_material`.
# Si un match est trouvé en fonction du plus petit rectangle, l'algorithme passe à la phase 2 de la détection.
# À cette étape, l'image recadrée (selon le plus petit rectangle) est comparée à une série d'images préenregistrées
# à l'aide de la fonction `featureMatching`.

# Pour plus de détails sur le fonctionnement de la correspondance de modèles, consultez la documentation d'OpenCV :
# [Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

import cv2
import cv2.version
import numpy as np
import os
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime
import time
from material import Material
import matplotlib.pyplot as plt

# Initialize global materials list
materials = []

class ObjectDetector:
    def __init__(self):
        self.ratio_width = 1
        self.ratio_length = 1
        self.register_material()  # Register materials during initialization

    def process_frame(self, frame):

        # Convert the difference frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Optional: Apply Gaussian blur if needed
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        med_val = np.median(blurred)
        lower = int(max(0, 0.4 * med_val))
        upper = int(min(255, 1.3 * med_val))
        edges = cv2.Canny(blurred, lower, upper)

        # Threshold the image
        _, thresh = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)

        # Display combined and inverted images for debugging
        # plt.imshow(edges, cmap='gray')
        # plt.show()

        # Find contours from the inverted image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw bounding boxes and capture dimensions
        for contour in contours:
            #if cv2.contourArea(contour) > 5:  # Filter out small contours
            # Get the minimum area rectangle
            box = cv2.minAreaRect(contour)
            center, (width, height), angle = box

            # Get the four corner points of the rectangle
            box_points = cv2.boxPoints(box)
            box_points = np.int32(box_points)

            # Calculate dimensions in mm
            width_mm = width * self.ratio_width
            length_mm = height * self.ratio_length

                # Make sure to swap width and length if needed
            bb_width = min(width_mm, length_mm)
            bb_length = max(width_mm, length_mm)

            # Find the object
            matched_material = None
            for mat in materials:
                if mat.compare_dimension(bb_width = bb_width, bb_length =bb_length):
                    if self.compare_images(frame,box_points,mat):
                        matched_material = mat
                        break

            # Prepare multi-line label text
            if matched_material:
                    # Draw bounding box and log the result
                color = (0, 255, 0)
                cv2.polylines(frame, [box_points], isClosed=True, color=color, thickness=2)
                label_lines = [
                    f"Nom: {matched_material.name}",
                    f"Largeur: {bb_width:.2f} mm",
                    f"Longueur: {bb_length:.2f} mm"
                ]
                # Position for the first line of the label
                label_x = int(center[0]+(height/2)+10)
                label_y = int(center[1]-(width/2)) # Slightly above the center

                # Add each line of the label to the frame
                for i, line in enumerate(label_lines):
                    cv2.putText(frame, line, (label_x, label_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                #print(f'Bounding Box: center={center}, width={width_mm}, height={length_mm}, angle={angle}')

        return frame
    
    def load_templates(self, folder_location):
        templates = []
        for filename in os.listdir(folder_location):
            img_path = os.path.join(folder_location, filename)
            img = cv2.imread(img_path, cv2.COLOR_BGRA2BGR)
            if img is not None:
                templates.append((filename, img))
        return templates

    def compare_images(self, frame, box_points, mat) -> bool:
        templates = self.load_templates(mat.folder_location)

        # Get the bounding rectangle from the box points
        x, y, w, h = cv2.boundingRect(box_points)

        # Define padding
        padding = 50

        # Calculate the new coordinates with padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)

        # Get the ROI with padding
        roi = frame[y_start:y_end, x_start:x_end]

        # Convert ROI to grayscale for template matching
        #roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB.create()

        for template_name, template in templates:
            # Ensure template is not larger than the ROI
            if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                #print(f"Template {template_name} is larger than ROI. Skipping.")
                continue
            
            # plt.subplot(121)
            # plt.imshow(roi,cmap="gray")
            # plt.subplot(122)
            # plt.imshow(template_resized,cmap="gray")
            # plt.show()

            _,des1 = orb.detectAndCompute(template,None)
            _,des2 = orb.detectAndCompute(roi,None)

            if des1 is None or des2 is None:
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches,key=lambda x:x.distance)


            if len(matches) > 0:
            # Compare the distance of the best match against a threshold
                best_match_distance = matches[0].distance
                threshold = 24  # A lower value indicate a good match
                #print(best_match_distance)
                if best_match_distance <= threshold:
                    #print(f"Match found for: {template_name} with distance: {best_match_distance}")
                    return True

        return False


    
    def register_material(self):
        bolt = Material(name="Boulon M10 x 60", width=18.1, length=66.5,folder_location="assets/img_boulon")
        ecrou = Material(name="Ecrou M5", width=11,length=11,folder_location="assets/img_ecrou")
        vis = Material(name="Vis M6 x 38",width=12.5,length=38,folder_location="assets/img_vis")
        vis_blanche = Material(name="Vis Blanche M5 x 50",width=10.4,length=54,folder_location="assets/img_vis_blanche")
        materials.append(bolt)
        materials.append(ecrou)
        materials.append(vis)
        materials.append(vis_blanche)

    def calibration(self):
        length_calibration_mm = 136.5
        width_calibration_mm = 38
        length_calibration_px = 224.4
        width_calibration_px = 64.6

        self.ratio_width = width_calibration_mm / width_calibration_px
        self.ratio_length = length_calibration_mm / length_calibration_px


def main():
    #print(cv2.__version__)
    # Initialize Kinect
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    time.sleep(5)  # Enough time to let the Kinect power on

    # Get the background
    background = kinect.get_last_color_frame()
    background = background.reshape((1080, 1920, 4))
    background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)

    detector = ObjectDetector()
    detector.calibration()  # Perform calibration

    while True:
        if kinect.has_new_color_frame():
            # Get the color frame
            frame = kinect.get_last_color_frame()
            if frame is not None:
                # Reshape and convert to OpenCV format
                frame = frame.reshape((1080, 1920, 4))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Remove the background from the frame
                diff_frame = cv2.absdiff(background, frame)

                # Process the frame to detect objects
                processed_frame = detector.process_frame(frame)

                # Display the frame with bounding boxes
                cv2.imshow('Kinect Video with Object Detection', processed_frame)

                # Exit on pressing 'ESC'
                if cv2.waitKey(1) == 27:
                    break

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
