import cv2
import mediapipe as mp
import pyautogui
import time

from src.tips import GetTips
from src.gestures.swipes import detect_Swipe


class GestureDetection:
    def __init__(self, height, width):
        self.recent_action = False
        self.is_idle = False
        self.idle_time = 0.0
        self.height = height
        self.width = width

    def get_fingers_nb_name_dict(self):
        return {
            4: "thumb",
            8: "index",
            12: "major",
            16: "anular",
            20: "auriculaire",
            0: "base_hand",
        }

    def get_fingers_name_nb(self):
        return {
            "thumb": 4,
            "index": 8,
            "major": 12,
            "anular": 16,
            "auriculaire": 20,
            "base_hand": 0,
        }

    def scroll_mouse(self, direction):
        if direction == "up":
            pyautogui.scroll(600)
            # Changez -10 à une valeur positive pour défiler vers le haut
        if direction == "down":
            pyautogui.scroll(-600)

    def press_key(seld, key):
        pyautogui.press(key)

    def distance3D(self, landmark1, landmark2, round_value=2):
        return [
            round(landmark1.x - landmark2.x, round_value),
            round(landmark1.y - landmark2.y, round_value),
            round(landmark1.z - landmark2.z, round_value),
        ]

    def detect_gesture(self, landmarks, prev_positions):
        # Example rule-based detection: Check if thumb is near index finger (signifying a "pinch" gesture)
        fingers_name_nb = self.get_fingers_name_nb()

        tips = GetTips(landmarks, fingers_name_nb)

        distance_thumbs_index = (
            (tips.thumb.x - tips.index.x) ** 2 + (tips.thumb.y - tips.index.y) ** 2
        ) ** 0.5

        hand_ratio = (
            (tips.base_hand.x - landmarks[5].x) / (landmarks[0].y - landmarks[5].y)
        ) / 0.19

        # ⬇️ This is where the custom functions are supposed to be called ⬇️

        x_margin_moving = (self.width / 4700) * hand_ratio
        y_margin_moving = (self.height / 3350) * hand_ratio
        # Calculate the coordinates for the square

        if len(prev_positions) > 7:
            # we need at least 7 previous positions to detect a gesture

            dim3_dist_base = self.distance3D(
                tips.base_hand, prev_positions[-3]["base_hand"]
            )
            is_base_hand_moving = False

            if ((dim3_dist_base[0]) ** 2) ** 0.5 > x_margin_moving:  # x coords
                print("x axis")
                print(((dim3_dist_base[0]) ** 2) ** 0.5, ">", x_margin_moving)
                is_base_hand_moving = True
            if dim3_dist_base[1] > y_margin_moving:  # y coords
                print("y axis")
                is_base_hand_moving = True

            # attention, les valeurs de seuil sont configurés pour le device nino-laptop
            # il se peut qu'en changeant de camera, on doive re changer les valeurs.
            # le cas échéant faire un ratio à partir de la taille de pixels de la camera.
            if not is_base_hand_moving:
                if not self.is_idle:
                    self.is_idle = True
                    self.idle_time = time.time()
                    print("idling")
                elif self.is_idle:
                    if time.time() - self.idle_time > 0.4:
                        print("idle")
                        self.recent_action = False
            else:
                self.is_idle = False
                self.idle_time = time.time()
            # ---------
            detect_Swipe(self, tips, prev_positions, is_base_hand_moving)

            # ---------
            if distance_thumbs_index < 0.05:
                return "Pinch"
            return "Unknown Gesture"
