import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

global recent_action
recent_action = False

height = 1280
width = 720


def getRecentAction():
    return recent_action


def setRecentAction(value):
    global recent_action
    recent_action = value
    return recent_action


global is_idle
is_idle = False

global idle_time
idle_time = 0.0


def getIdle():
    return is_idle, idle_time


def SetIdle(isIdle, IdleTime):
    global is_idle
    global idle_time
    is_idle = isIdle
    idle_time = IdleTime
    return is_idle, idle_time


url = ''

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

blue_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=5)  # Blue color for landmark 0
purple_spec = mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2, circle_radius=2)  # pruple color for landmark 0
test_spec = mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=2)  # pruple color for landmark 0
default_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                      circle_radius=2)  # Default color for other landmarks


def scroll_mouse(direction):
    if direction == 'up':
        pyautogui.scroll(600)  # Changez -10 à une valeur positive pour défiler vers le haut
    if direction == 'down':
        pyautogui.scroll(-600)


fingers_dict = {
    4: "thumb",
    8: "index",
    12: "major",
    16: "anular",
    20: "auriculaire",
    20: "base_hand"
}


def distance3D(landmark1, landmark2, round_value=2):
    return [round(landmark1.x - landmark2.x, round_value),
            round(landmark1.y - landmark2.y, round_value),
            round(landmark1.z - landmark2.z, round_value)
            ]


# Define gestures based on keypoints or simple rules
def detect_gesture(landmarks, prev_positions):
    # Example rule-based detection: Check if thumb is near index finger (signifying a "pinch" gesture)

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    major_tip = landmarks[12]
    anunlar_tip = landmarks[16]
    auriculaire_tip = landmarks[20]
    base_hand = landmarks[0]

    distance_thumbs_index = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    hand_ratio = ((landmarks[0].x - landmarks[5].x) / (landmarks[0].y - landmarks[5].y)) / 0.19

    if len(prev_positions) > 7:

        # print('dists index & previous', distance3D(thumb_tip, prev_positions[-3]["index_tip"]))5
        is_dist_index = ((distance3D(index_tip, prev_positions[-3]["index"])[1]) ** 2) ** 0.5 > 0.12
        is_dist_major = ((distance3D(major_tip, prev_positions[-3]["major"])[1]) ** 2) ** 0.5 > 0.12
        is_dist_annu = ((distance3D(anunlar_tip, prev_positions[-3]["anular"])[1]) ** 2) ** 0.5 > 0.12
        dim3_dist_base = distance3D(base_hand, prev_positions[-3]["base_hand"])
        is_base_hand_moving = False

        x_margin_moving = (width / 4700) * hand_ratio
        y_margin_moving = (height / 3350) * hand_ratio
        # Calculate the coordinates for the square

        if ((dim3_dist_base[0]) ** 2) ** 0.5 > x_margin_moving:  # x coords
            print("x axis")
            is_base_hand_moving = True
        if dim3_dist_base[1] > y_margin_moving:  # y coords
            print("y axis")
            is_base_hand_moving = True

        # attention, les valeurs de seuil sont configurés pour le device nino-laptop
        # il se peut qu'en changeant de camera, on doive re changer les valeurs.
        # le cas échéant faire un ratio à partir de la taille de pixels de la camera.
        if not is_base_hand_moving:
            if not getIdle()[0]:
                SetIdle(isIdle=True, IdleTime=time.time())
                print("idling")
            elif getIdle()[0]:
                if time.time() - getIdle()[1] > 0.4:
                    print("idle")

                    setRecentAction(False)
        else:
            SetIdle(isIdle=False, IdleTime=time.time())


        if is_dist_index and is_dist_major and is_dist_annu and not is_base_hand_moving:
            if not getRecentAction():
                print("swipe", distance3D(index_tip, prev_positions[-3]["index"])[1])
                print(recent_action, time.time())
                if distance3D(index_tip, prev_positions[-3]["index"])[1] > 0:
                    scroll_mouse('down')
                    setRecentAction(True)
                    SetIdle(isIdle=False, IdleTime=time.time())
                    print("scroll down",time.time())
                else:
                    scroll_mouse('up')
                    setRecentAction(True)
                    SetIdle(isIdle=False, IdleTime=time.time())
                    print("scroll up", time.time())

            return "swipe"
    if distance_thumbs_index < 0.05:
        return "Pinch"
    return "Unknown Gesture"


# Start video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
prev_positions = []

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Draw hand annotations and detect gestures if hands are present
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Use mapping to apply blue color for index 0 and default color for other landmarks
            landmark_styles = {0: blue_spec}
            for i in range(1, 21):
                landmark_styles[i] = default_spec
            for i in [5, 9, 13, 17]:
                landmark_styles[i] = purple_spec
                # Draw landmarks with the specified styles
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_styles
            )

            # Display landmark numbers
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i in list(fingers_dict.keys()) and len(prev_positions) > 6:
                    prev_coord = prev_positions[-7][fingers_dict[i]]
                    actual_coords = landmark
                    dist = distance3D(prev_coord, landmark)

                    cv2.putText(frame, f"{fingers_dict[i]}:{x, y} - {dist}", (10, 10 + i * 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # Get the coordinates of the landmark
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Display the landmark number
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Detect gesture based on hand landmarks
            gesture = detect_gesture(hand_landmarks.landmark, prev_positions)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

        emplacements = {}
        for e in fingers_dict:
            emplacements[fingers_dict[e]] = hand_landmarks.landmark[e]

        prev_positions.append(emplacements)
    # Show the frame
    if getIdle()[0]:
        cv2.putText(frame, 'idle', (int(0.9*width), 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'idling', (int(0.9*width), 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
