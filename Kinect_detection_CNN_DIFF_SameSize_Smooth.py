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
import torch
from material import Material
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from skimage.metrics import structural_similarity as ssim
from collections import Counter
from datetime import datetime, timedelta
from collections import defaultdict
from cnn_classifier import cnn_fasteners_classifier


# Initialize global materials list
materials = []
matched_material = [[]]

# Initialize cnn classifier
cnn_model = cnn_fasteners_classifier()
cnn_model.load_state_dict(torch.load("./cnn_model_double_v3.pth"))
cnn_model = cnn_model.eval()

class Matched_Material:
    def __init__(self, material: Material, bounding_box : List[float], min_bounding_box: List[float]):
        self.material = material
        self.bounding_box = bounding_box
        self.min_bounding_box = min_bounding_box
        self.timestamp = datetime.now()
    def get_center(self):
        # Assuming 'contour' is already defined
        x, y, w, h = self.bounding_box

        # Calculate the center
        center_x = x + w // 2
        center_y = y + h // 2

        return (center_x, center_y)
    def compare_center(self,new_center):
        center = self.get_center()

        # Calculate the allowed range for both x and y based on 10%
        x_range = center[0] * 0.1
        y_range = center[1] * 0.1

        # Check if new_center is within the range of the original center
        if (abs(center[0] - new_center[0]) <= x_range) and (abs(center[1] - new_center[1]) <= y_range):
            return True
        return False
        

class Average_Materials:
    TIME = 1 #in secondes
    
    def __init__(self):
        self.average_list = []  # List of lists, each containing matched materials
        self.timeout = timedelta(seconds=self.TIME)  # Use timedelta for duration

    def add_material(self, materials: List[Matched_Material]):
        current_time = datetime.now()  # Current timestamp
        for mat in materials:
            matched = False

            # If list is empty, add the first material
            if not self.average_list:
                self.average_list.append([mat])
                continue

            # Check if the material matches any in the existing lists
            for match in self.average_list:
                if match[0].compare_center(mat.get_center()):
                    match.append(mat)
                    matched = True
                    break

            # If no match was found, create a new list for this material
            if not matched:
                self.average_list.append([mat])

            # Filter out expired materials and remove empty matches
            self.average_list = [
                [mat for mat in match if current_time - mat.timestamp <= self.timeout]
                for match in self.average_list
            ]

            # Remove any empty lists
            self.average_list = [match for match in self.average_list if match]

        return self.get_matches()

    def get_matches(self):
        matched_materials = []

        # For each group of materials in the average list
        for match_group in self.average_list:
            # Find the most frequent material based on the name
            material_names = [mat.material.name for mat in match_group]
            most_common_name = Counter(material_names).most_common(1)[0][0]

            # Filter materials by the most common name (assuming it's the correct match)
            filtered_materials = [mat for mat in match_group if mat.material.name == most_common_name]

            # Calculate the average bounding box and min bounding box
            avg_bbox = self.calculate_average_bounding_box([mat.bounding_box for mat in filtered_materials])
            avg_min_bbox = self.calculate_average_minarearect([mat.min_bounding_box for mat in filtered_materials])

            # Create a new Matched_Material object with the average values
            matched_material = Matched_Material(
                material=filtered_materials[0].material,  # Use the material from the filtered list
                bounding_box=avg_bbox,
                min_bounding_box=avg_min_bbox
            )
            matched_materials.append(matched_material)

        return matched_materials

    def calculate_average_bounding_box(self, bounding_boxes):
        # Calculate the average bounding box from a list of bounding boxes
        #print(bounding_boxes)
        avg_x = np.mean([bbox[0] for bbox in bounding_boxes])
        avg_y = np.mean([bbox[1] for bbox in bounding_boxes])
        avg_w = np.mean([bbox[2] for bbox in bounding_boxes])
        avg_h = np.mean([bbox[3] for bbox in bounding_boxes])

        return (avg_x, avg_y, avg_w, avg_h)
    
    def calculate_average_minarearect(self, rects):
        # Unpack the input rectangles
        centers = np.array([rect[0] for rect in rects])  # x, y centers
        sizes = np.array([rect[1] for rect in rects])  # width, height
        angles = np.array([rect[2] for rect in rects])  # angles
        
        # Average the center (x, y)
        avg_center = np.mean(centers, axis=0)  # Average x and y
        
        # Average the size (width, height)
        avg_size = np.mean(sizes, axis=0)  # Average width and height
        
        # Average the angle
        avg_angle = np.mean(angles)  # Average the angles
        
        # Return the average minAreaRect
        return (tuple(avg_center), tuple(avg_size), avg_angle)

    def get_materials(self):
        #Return a list[string] of material, material of the same need to be added ex : mat1 qty : 2, mat2 qty : 4
        #Use the method get_matches that return as list of Matrial. Check the Material.Name to find the similar one
        # Use defaultdict to count quantities of materials by name
        material_count = defaultdict(int)

        # Get the matches
        matches = self.get_matches()  # Assuming this returns a list of Matched_Material

        # Count materials with the same name
        for match in matches:
            material_name = match.material.name  # Assuming 'material.name' exists in Material
            material_count[material_name] += 1

        # Format the output as a list of strings
        return ([f"{name} qty: {qty}" for name, qty in material_count.items()])


class ObjectDetector:
    def __init__(self):
        self.ratio_width = 1
        self.ratio_length = 1
        self.register_material()  # Register materials during initialization
        self.prev_centers = []  # To store previous centers for smoothing
        self.average_materials = Average_Materials()

    def get_contours(self,diff):
        # Convert frame to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold the image
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect contours that fit criteria
        match_contours = []

        # Draw bounding boxes and capture dimensions
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter out small contours
                match_contours.append(contour)
        return match_contours

    def find_match(self,contours: List[np.ndarray], frame):
        matchs = []
        # Get the minimum area rectangle
        for contour in contours:
            box = cv2.minAreaRect(contour)
            bbox = cv2.boundingRect(contour)
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
                    if self.compare_images(frame,box_points, mat, cnn_model):
                        matched_material = Matched_Material(mat,bbox,box)
                        matchs.append(matched_material)
                        break
        return matchs

    def draw_bounding_box(self,matchs : List[Matched_Material],frame): 
        for match in matchs:
            mat = match.material
            minbox = match.min_bounding_box
            center, (width, height), angle = minbox
            # Get the four corner points of the rectangle
            box_points = cv2.boxPoints(minbox)
            box_points = np.int32(box_points)

            # Calculate dimensions in mm
            width_mm = width * self.ratio_width
            length_mm = height * self.ratio_length

                # Make sure to swap width and length if needed
            bb_width = min(width_mm, length_mm)
            bb_length = max(width_mm, length_mm)

            # Draw bounding box
            color = (0, 255, 0);
            cv2.polylines(frame, [box_points], isClosed=True, color=color, thickness=2)

            label_lines = [
                f"Nom: {mat.name}",
                f"Largeur: {bb_width:.2f} mm",
                f"Longueur: {bb_length:.2f} mm"
            ]


            # Position for the first line of the label
            label_x = int(center[0]+(height/2)+10)
            label_y = int(center[1]-(width/2)) # Slightly above the center

            # Add each line of the label to the frame
            for i, line in enumerate(label_lines):
                cv2.putText(frame, line, (label_x, label_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return frame

    def process_frame(self, diff, frame):
        contours = self.get_contours(diff)
        matches = self.find_match(contours,frame)
        #Average the last x seconds
        new_matches = self.average_materials.add_material(matches)
        #Draw the matches
        new_frame = self.draw_bounding_box(new_matches,frame)
        return new_frame
    
    def load_templates(self, folder_location):
        templates = []
        for filename in os.listdir(folder_location):
            img_path = os.path.join(folder_location, filename)
            img = cv2.imread(img_path)
            if img is not None:
                templates.append((filename, img))
        return templates

    def compare_images(self, frame, box_points, mat, model) -> bool:
        x, y, w, h = cv2.boundingRect(box_points)

        pad_height = max(0, 150 - h)
        pad_width = max(0, 150 - w)

        # Calculate new ROI coordinates with padding
        roi_y_start = max(0, y - pad_height // 2)
        roi_y_end = min(frame.shape[0], y + h + pad_height // 2)
        roi_x_start = max(0, x - pad_width // 2)
        roi_x_end = min(frame.shape[1], x + w + pad_width // 2)

        # Create ROI
        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]      
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) # grayscale
        roi = cv2.resize(roi, (150, 150))
        roi = roi/255. # convert to 0-1 range
        roi = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float() # get correct shape for classifier

        # Get model predictions and apply softmax to normalize preds between 0-1
        sfmax = torch.nn.Softmax(-1)
        res = sfmax(model.forward(roi))

        labels_map = {
        0: "Background",
        1: "Boulon M10 x 60",
        2: "2x Boulon M10 x 60",
        3: "Ecrou M5",
        4: "Vis M6 x 38",
        5: "Vis Blanche M5 x 50",
        }

        # Define a more strict threshold
        threshold = .95

        if (torch.max(res) > threshold) and (labels_map[res.argmax(-1).item()] == mat.name):
            return True

        return False

    
    def register_material(self):
        bolt = Material(name="Boulon M10 x 60", width=21, length=68,folder_location="assets/img_boulon")
        ecrou = Material(name="Ecrou M5", width=13,length=13,folder_location="assets/img_ecrou")
        vis = Material(name="Vis M6 x 38",width=14,length=35,folder_location="assets/img_vis")
        vis_blanche = Material(name="Vis Blanche M5 x 50",width=10.4,length=50,folder_location="assets/img_vis_blanche")
        double_bolt = Material(name="2x Boulon M10 x 60",width=68,length=68,folder_location="assets/img_boulon_double")
        materials.append(bolt)
        materials.append(ecrou)
        materials.append(vis)
        materials.append(vis_blanche)
        materials.append(double_bolt)


    def calibration(self):
        length_calibration_mm = 136.5
        width_calibration_mm = 38
        length_calibration_px = 232
        width_calibration_px = 69

        self.ratio_width = width_calibration_mm / width_calibration_px
        self.ratio_length = length_calibration_mm / length_calibration_px


def main():
    #print(cv2.__version__)
    # Initialize Kinect
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    time.sleep(3)  # Enough time to let the Kinect power on

    # Get the background
    background = cv2.imread("assets/background.jpg")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

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

                # Remove the background from the frame
                diff_frame = cv2.absdiff(background, frame)

                # Process the frame to detect objects
                processed_frame = detector.process_frame(diff_frame, frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

                # Display the frame with bounding boxes
                cv2.imshow('Kinect Video with Object Detection', processed_frame)

                # Exit on pressing 'ESC'
                if cv2.waitKey(1) == 27:
                    break

    # Cleanup
    cv2.destroyAllWindows()






    frame = frame.reshape((1080, 1920, 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # Remove the background from the frame
    diff_frame = cv2.absdiff(background, frame)

    # Process the frame to detect objects
    processed_frame = detector.process_frame(diff_frame, frame)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

    # Display the frame with bounding boxes
    cv2.imshow('Kinect Video with Object Detection', processed_frame)

    # Exit on pressing 'ESC'
    cv2.waitKey(0)

    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
