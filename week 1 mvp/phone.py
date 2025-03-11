import cv2 # type: ignore
import mediapipe as mp # type: ignore
import torch 
import time

# Load YOLOv8 model for phone detection (pretrained weights)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open Webcam
cap = cv2.VideoCapture(0)

phone_hold_count = 0
holding_phone = False
start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    hand_results = hands.process(rgb_frame)

    # Detect phone using YOLOv8
    phone_results = model(rgb_frame)
    phones = [det for det in phone_results.xyxy[0] if int(det[5]) == 67]  
    # Draw hands on the frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Check if a phone is detected
    if phones:
        for phone in phones:
            x1, y1, x2, y2, conf, cls = phone.tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # If hands and phones appear together, count as a hold
        if hand_results.multi_hand_landmarks:
            if not holding_phone:
                holding_phone = True
                phone_hold_count += 1
                start_time = time.time()  # Start timing phone hold duration

    else:
        holding_phone = False

    # Display count
    cv2.putText(frame, f"Phone Holds: {phone_hold_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Phone Hold Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
