import cv2
import mediapipe as mp
import pyautogui
import time

from src.tips import GetTips
from src.gestures.swipes import detect_Swipe
from src.gestures.volume import volume_up_and_down

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

        if len(prev_positions) > 7:
            # we need at least 7 previous positions to detect a gesture
            #detect_Swipe(self, tips, prev_positions, hand_ratio)
            volume_up_and_down(self, tips, prev_positions, hand_ratio)
        if distance_thumbs_index < 0.05:
            return "Pinch"
        return "Unknown Gesture"
