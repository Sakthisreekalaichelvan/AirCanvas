import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Variables for drawing
draw_color = (0, 255, 255)  # Default purple color
brush_thickness = 15
eraser_thickness = 50
xp, yp = 0, 0  # Previous x and y points
img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Blank canvas

# Capture video from camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)   # Set frame height

# Colors available for drawing (BGR format)
colors = {
    "Yellow": (0, 255, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    
    "Eraser": (0, 0, 0)
}

# Helper function to draw buttons for selecting colors
def draw_color_buttons(img):
    cv2.rectangle(img, (0, 0), (120, 120), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, 'Yellow', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(img, (140, 0), (260, 120), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, 'Blue', (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(img, (280, 0), (400, 120), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, 'Green', (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(img, (420, 0), (540, 120), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, 'Eraser', (440, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

while True:
    # 1. Capture frames from the camera
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a natural feel
    img = cv2.flip(img, 1)

    # Draw color selection buttons
    draw_color_buttons(img)

    # 2. Convert the image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Detect hand landmarks
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for the index finger
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) > 0:
                # Get tip of index finger
                index_finger_tip = lm_list[8][1], lm_list[8][2]

                # 4. Check if the hand is in the color selection area
                if index_finger_tip[1] < 120:
                    if 0 < index_finger_tip[0] < 120:
                        draw_color = colors["Purple"]
                    elif 140 < index_finger_tip[0] < 260:
                        draw_color = colors["Blue"]
                    elif 280 < index_finger_tip[0] < 400:
                        draw_color = colors["Green"]
                    elif 420 < index_finger_tip[0] < 540:
                        draw_color = colors["Eraser"]

                # 5. Draw on the canvas
                if xp == 0 and yp == 0:
                    xp, yp = index_finger_tip

                if draw_color == (0, 0, 0):  # Erasing
                    cv2.line(img, (xp, yp), index_finger_tip, draw_color, eraser_thickness)
                    cv2.line(img_canvas, (xp, yp), index_finger_tip, draw_color, eraser_thickness)
                else:  # Drawing
                    cv2.line(img, (xp, yp), index_finger_tip, draw_color, brush_thickness)
                    cv2.line(img_canvas, (xp, yp), index_finger_tip, draw_color, brush_thickness)

                xp, yp = index_finger_tip

    # 6. Merge the drawing with the original image
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # 7. Display the final output
    cv2.imshow("Air Canvas", img)

    # 8. Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
