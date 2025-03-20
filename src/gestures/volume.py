import time
from src.tips import GetTips
import pyautogui


def increase_volume():
    # Simule la pression de la touche "Volume Up" 5 fois
    for _ in range(5):
        pyautogui.press("volumeup")


def decrease_volume():
    # Simule la pression de la touche "Volume Down" 5 fois
    for _ in range(5):
        pyautogui.press("volumedown")


def detect_volume_change(
    self, tips: GetTips, prev_positions, closed_fingers, hand_ratio
):

    is_dist_thumb = (
        (self.distance3D(tips.anular, prev_positions[-3]["thumb"])[1]) ** 2
    ) ** 0.5 > 0.12

    thumb_x_movement = self.distance3D(tips.thumb, prev_positions[-3]["thumb"])[0]
    anular_x_movement = self.distance3D(tips.anular, prev_positions[-3]["anular"])[0]
    major_x_movement = self.distance3D(tips.anular, prev_positions[-3]["major"])[0]
    index_x_movement = self.distance3D(tips.anular, prev_positions[-3]["index"])[0]
    print("mouvement ", tips.thumb)
    if (thumb_x_movement + 0.07) < index_x_movement:
        print("c'est bon")
    if is_dist_thumb:
        if (
            not self.recent_action
            and (thumb_x_movement + 0.07) < anular_x_movement
            and thumb_x_movement < index_x_movement
            and thumb_x_movement < major_x_movement
        ):
            print("hih", self.recent_action, time.time())
            if (
                self.distance3D(tips.thumb, prev_positions[-3]["thumb"])[1] > 0
                and -0.2 < thumb_x_movement < 0.2
                and -0.2 < index_x_movement < 0.2
                and -0.2 < major_x_movement < 0.2
                and -0.2 < anular_x_movement < 0.2
            ):
                increase_volume()
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("baisse de volume", time.time())
                print("baisse de volume", time.time())
                print("baisse de volume", time.time())
                return "swipe_vert down"
            elif (
                self.distance3D(tips.thumb, prev_positions[-3]["thumb"])[1] < 0
                and -0.2 < thumb_x_movement < 0.2
                and -0.2 < index_x_movement < 0.2
                and -0.2 < major_x_movement < 0.2
                and -0.2 < anular_x_movement < 0.2
            ):
                decrease_volume()
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("augmentation de volume", time.time())
                print("augmentation de volume", time.time())
                print("augmentation de volume", time.time())
                return "swipe_vert up"
