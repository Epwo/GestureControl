import cv2
import mediapipe as mp
from dotenv import load_dotenv
import os
from src.logic import GestureDetection

height = 1280
width = 720
load_dotenv()
video_source = os.environ.get("VIDEO_SOURCE")
print("video_srouce : ", video_source)
# Start video capture
GD = GestureDetection(height, width)
cap = cv2.VideoCapture(int(video_source), cv2.CAP_DSHOW)
prev_positions = []

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

blue_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=5)
purple_spec = mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2, circle_radius=2)
test_spec = mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=2)
default_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

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
    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Check if this is a right hand
            handedness = result.multi_handedness[idx]
            if handedness.classification[0].label != "Right":
                continue  # Skip if not right hand

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
                landmark_drawing_spec=landmark_styles,
            )

            # Display landmark numbers
            for i, landmark in enumerate(hand_landmarks.landmark):
                if (
                    i in list(GD.get_fingers_nb_name_dict().keys())
                    and len(prev_positions) > 6
                ):
                    prev_coord = prev_positions[-7][GD.get_fingers_nb_name_dict()[i]]
                    actual_coords = landmark
                    dist = GD.distance3D(prev_coord, landmark)

                    cv2.putText(
                        frame,
                        f"{GD.get_fingers_nb_name_dict()[i]}:{x, y} - {dist}",
                        (10, 10 + i * 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                # Get the coordinates of the landmark
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Display the landmark number
                cv2.putText(
                    frame,
                    str(i),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Detect gesture based on hand landmarks
            gesture = GD.detect_gesture(hand_landmarks.landmark, prev_positions)
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            emplacements = {}
            for e in GD.get_fingers_nb_name_dict():
                emplacements[GD.get_fingers_nb_name_dict()[e]] = (
                    hand_landmarks.landmark[e]
                )

            prev_positions.append(emplacements)
            # Only process one right hand (the first one found)
            break

    # Show the frame
    if GD.is_idle:
        cv2.putText(
            frame,
            "idle",
            (int(0.9 * width), 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "idling",
            (int(0.9 * width), 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
