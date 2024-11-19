# README

# Étape 1 : Calibration de la caméra

# 1. Calibration avec une pièce prédéterminée :
#    - Pour commencer la calibration de la caméra, commentez les lignes 208 et 209 dans le code.
#    - Exécutez le programme.
#    - Prenez note de la largeur et de la longueur de la pièce de calibration.
#    - Décommentez les lignes 208 et 209, puis indiquez les valeurs de largeur et de longueur sur les lignes 205 et 206.

# Étape 2 : Prendre une photo du plan de travail
# Prenez une photo du plan de travail sans objet.
# Copiez la photo dans le dossier du projet et nommez-la background.jpg.

# Étape 3 : Exécution du programme

# Fonctionnement du programme
# Le programme utilise la détection de contours via la différence entre background.jpg et la capture de la caméra.
# Ensuite, les opérations suivantes sont effectuées sur la photo :
#    - L'image de la différence est convertie en noir et blanc.
#    - Un flou est appliqué.
#    - Un seuil (Threshold) est utilisé pour convertir les pixels en noir ou blanc (0, 255).
# Cet algorithme permet de détecter les objets préenregistrés ainsi que les objets intrus.
# Lorsqu'un contour est détecté, le programme mesure le plus petit rectangle englobant cet objet.

# Ce rectangle est ensuite comparé avec les références initialisées dans la méthode `register_material`.
# Si un match est trouvé en fonction du plus petit rectangle, l'algorithme passe à la phase 2 de la détection.
# À cette étape, l'image recadrée (selon le plus petit rectangle) est comparée à une série d'images préenregistrées
# à l'aide de la fonction `matchTemplate`.

# Pour plus de détails sur le fonctionnement de la correspondance de modèles, consultez la documentation d'OpenCV :
# [Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)


import cv2
import cv2.version
import numpy as np
import os
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime
import time
from material import Material
import matplotlib.pyplot as plt
from collections import Counter

# Initialize global materials list
materials = []
tracked_materials = []  # List to store corresponding materials

class ObjectDetector:
    def __init__(self):
        self.ratio_width = 1
        self.ratio_length = 1
        self.register_material()  # Register materials during initialization
        self.trackers = []  # List to store trackers for each object
        
        self.tracker_boxes = []

    def process_frame(self, diff, frame):
        # Update existing trackers and remove those that fail
        for i in range(len(self.trackers) - 1, -1, -1):  # Iterate in reverse to safely remove items
            success, box = self.trackers[i].update(frame)
            if success:
                box_points = np.array([
                    [box[0], box[1]],           # Top-left
                    [box[0] + box[2], box[1]],       # Top-right
                    [box[0], box[1] + box[3]],       # Bottom-left
                    [box[0] + box[2], box[1] + box[3]]    # Bottom-right
                ])
                # #Check if image still fit
                # if not self.compare_images(frame,box_points,tracked_materials[i]) :
                #     self.trackers.pop(i)
                #     tracked_materials.pop(i)
                #     self.tracker_boxes.pop(i)
                #     continue
                #Update the position in tracker_boxes
                self.tracker_boxes[i] = box
                # Draw tracked bounding box
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                color = (0, 255, 0)
                cv2.rectangle(frame, p1, p2, color, 2)

                # Display material information on each tracked bounding box
                matched_material = tracked_materials[i]
                cv2.putText(frame, f"Nom: {matched_material.name}", (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Remove tracker, material, and box if tracking failed
                self.trackers.pop(i)
                tracked_materials.pop(i)
                self.tracker_boxes.pop(i)

        # Perform detection if not all materials are tracked
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter out small contours
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

                if matched_material:
                    # Calculate bounding box for ROI around the contour
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check if this object is already tracked based on proximity
                    is_tracked = False
                    for tracker_box in self.tracker_boxes:
                        if self.check_proximity((x, y, w, h), tracker_box):
                            is_tracked = True
                            break

                    # If not tracked, add a new tracker
                    if not is_tracked:
                        tracker = cv2.TrackerKCF_create()  # CSRT is recommended for accuracy
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        tracked_materials.append(matched_material)
                        self.tracker_boxes.append((x, y, w, h))

        return frame

    def check_proximity(self, box1, box2, threshold=40):
        # Check if two boxes are close enough to be considered the same object
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return (abs(x1 - x2) < threshold and abs(y1 - y2) < threshold and 
                abs(w1 - w2) < threshold and abs(h1 - h2) < threshold)
    
    def load_templates(self, folder_location):
        templates = []
        for filename in os.listdir(folder_location):
            img_path = os.path.join(folder_location, filename)
            img = cv2.imread(img_path)
            if img is not None:
                templates.append((filename, img))
        return templates

    def compare_images(self, frame, box_points, mat) -> bool:
        templates = self.load_templates(mat.folder_location)

        # Get the bounding rectangle from the box points
        x, y, w, h = cv2.boundingRect(box_points)

        for template_name, template in templates:
            # Get the template size
            template_height = template.shape[0]
            template_width = template.shape[1]
            
            # Calculate the required padding
            pad_height = max(0, template_height - h)
            pad_width = max(0, template_width - w)

            # Calculate new ROI coordinates with padding
            roi_y_start = max(0, y - pad_height // 2)
            roi_y_end = min(frame.shape[0], y + h + pad_height // 2)
            roi_x_start = max(0, x - pad_width // 2)
            roi_x_end = min(frame.shape[1], x + w + pad_width // 2)

            # Create ROI
            roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # Resize the ROI if necessary to match the template size
            if roi.shape[0] != template_height or roi.shape[1] != template_width:
                roi_resized = cv2.resize(roi, (template_width, template_height))
            else:
                roi_resized = roi

            from skimage.metrics import structural_similarity as ssim

            # Ensure both ROI and template have a minimum size
            if roi_resized.shape[0] < 7 or roi_resized.shape[1] < 7 or template.shape[0] < 7 or template.shape[1] < 7:
                print(f"Skipping SSIM, image too small (ROI: {roi_resized.shape}, Template: {template.shape})")
                continue  # Skip SSIM comparison for this template

            # Compare the ROI with the template
            # Perform SSIM comparison for multi-channel (RGB) images
            bool_ssim = False
            score, _ = ssim(roi_resized, template, win_size=3, full=True, channel_axis=-1)
            if score > 0.84:  # Adjust threshold based on experimentation
                return True
            # # Template matching
            # res = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
            # # Define a more strict threshold
            # threshold = .65
            # loc = np.where(res >= threshold)
            # bool_matcher = False
            # if loc[0].size > 0:
            #     return True

            #Check if both are True
            # if bool_ssim and bool_matcher:
            #     return True

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
        length_calibration_px = 232
        width_calibration_px = 69

        self.ratio_width = width_calibration_mm / width_calibration_px
        self.ratio_length = length_calibration_mm / length_calibration_px


    def get_detected_material(self):
    # Extracting the names of the materials
        material_names = [mat.name for mat in tracked_materials]
        # Count the occurrences of each material name
        name_counts = Counter(material_names)
        materials = []
        for name, count in name_counts.items():
            materials.append(f"Matériel: {name}, Quantité: {count}")
        return materials

def main():
    #print(cv2.__version__)
    # Initialize Kinect
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    time.sleep(3)  # Enough time to let the Kinect power on

    # Get the background
    # background = kinect.get_last_color_frame()
    # background = background.reshape((1080, 1920, 4))
    # background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    background = cv2.imread("assets/background.jpg")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    # plt.imshow(background)
    # plt.show()

    detector = ObjectDetector()
    detector.calibration()  # Perform calibration

    while True:
        if kinect.has_new_color_frame():
            # Get the color frame
            frame = kinect.get_last_color_frame()
            if frame is not None:
                # Reshape and convert to OpenCV format
                frame = frame.reshape((1080, 1920, 4))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

                # plt.imshow(frame)
                # plt.show()

                # Remove the background from the frame
                diff_frame = cv2.absdiff(background, frame)

                # plt.subplot(131)
                # plt.imshow(background)
                # plt.subplot(132)
                # plt.imshow(frame)
                # plt.subplot(133)
                # plt.imshow(diff_frame)
                # plt.show()


                # Process the frame to detect objects
                processed_frame = detector.process_frame(diff_frame, frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

                # Display the frame with bounding boxes
                cv2.imshow('Kinect Video with Object Detection', processed_frame)
                print("*************************")
                
                #Print Material Detections
                mats = detector.get_detected_material()
                for mat in mats :
                    print(mat)


                # Exit on pressing 'ESC'
                if cv2.waitKey(1) == 27:
                    break

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
