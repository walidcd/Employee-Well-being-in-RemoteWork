import cv2
import mediapipe as mp
import numpy as np
import math
import os

def calculate_mouth_area(face_landmarks, frame_shape):
    """Calcule une zone plus large autour de la bouche pour une meilleure détection"""
    h_frame, w_frame = frame_shape[:2]
    mouth_points = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 91, 95, 146, 178, 308, 317, 318, 324, 402, 405]
    x_coords = []
    y_coords = []
    for point_idx in mouth_points:
        pt = face_landmarks.landmark[point_idx]
        x_coords.append(pt.x * w_frame)
        y_coords.append(pt.y * h_frame)
    mouth_center_x = sum(x_coords) / len(x_coords)
    mouth_center_y = sum(y_coords) / len(y_coords)
    return (int(mouth_center_x), int(mouth_center_y))

def is_bottle_at_mouth(bottle_box, mouth_center, threshold=150):
    """Vérifie si la bouteille est proche de la bouche"""
    x, y, w, h = bottle_box
    bottle_top = (x + w // 2, y)
    bottle_top_left = (x, y)
    bottle_top_right = (x + w, y)
    distances = [
        math.hypot(bottle_top[0] - mouth_center[0], bottle_top[1] - mouth_center[1]),
        math.hypot(bottle_top_left[0] - mouth_center[0], bottle_top_left[1] - mouth_center[1]),
        math.hypot(bottle_top_right[0] - mouth_center[0], bottle_top_right[1] - mouth_center[1])
    ]
    return min(distances) < threshold

def detect_bottles(frame, net, classes, target_class="bottle", conf_threshold=0.2, nms_threshold=0.4):
    """Détecte les bouteilles dans l'image"""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(ln)
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] == target_class:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w_box, h_box) = box.astype("int")
                x = int(centerX - (w_box / 2))
                y = int(centerY - (h_box / 2))
                boxes.append([x, y, int(w_box), int(h_box)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    final_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
    return final_boxes

# Configuration des chemins
config_path = r"C:\Users\salma\Downloads\models\yolov3.cfg"
weights_path = r"C:\Users\salma\Downloads\models\yolov3.weights"
classes_path = r"C:\Users\salma\Downloads\models\coco.names"

# Vérification des fichiers
assert os.path.exists(config_path), "Fichier yolov3.cfg introuvable !"
assert os.path.exists(weights_path), "Fichier yolov3.weights introuvable !"
assert os.path.exists(classes_path), "Fichier coco.names introuvable !"

# Chargement des classes
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialisation du modèle YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Configuration de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
sip_count = 0
is_drinking_previous = False  # État précédent
drinking_cooldown = 0  # Compteur pour éviter les doubles détections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bottle_boxes = detect_bottles(frame, net, classes)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    mouth_center = None
    if results.multi_face_landmarks:
        mouth_center = calculate_mouth_area(results.multi_face_landmarks[0], frame.shape)
        cv2.circle(frame, mouth_center, 150, (0, 255, 0), 2)

    if mouth_center and drinking_cooldown == 0:
        is_drinking_current = False
        for bottle_box in bottle_boxes:
            if is_bottle_at_mouth(bottle_box, mouth_center):
                cv2.putText(frame, "Drinking Detected!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                is_drinking_current = True
                break

        # Incrémenter le compteur quand on commence à boire
        if is_drinking_current and not is_drinking_previous:
            sip_count += 1
            drinking_cooldown = 30  # Attendre 30 frames avant la prochaine détection

        is_drinking_previous = is_drinking_current

    if drinking_cooldown > 0:
        drinking_cooldown -= 1

    # Affichage des bouteilles détectées
    for (x, y, w_box, h_box) in bottle_boxes:
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
        cv2.putText(frame, "Bottle", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Affichage du compteur de gorgées
    cv2.putText(frame, f"Sips: {sip_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Drinking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()