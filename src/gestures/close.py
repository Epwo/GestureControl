import time
import sys
from src.tips import GetTips
import pyautogui


def close_browser_click():
    pyautogui.hotkey("alt", "f4")
    print("Application fermée via Alt+F4.")


def close_browser_ctrl():
    pyautogui.hotkey("ctrl", "w")  # Ou ('command', 'w') pour Mac
    print("Onglet fermé via raccourci.")


def detect_close_gesture(self, tips: GetTips, prev_positions, hand_ratio):
    """
    Detects if the hand is fully closed, with the thumb hidden beneath the other fingers,
    and the hand is not moving (idle).
    """
    closed_fingers = 0
    finger_tips = [tips.index, tips.major, tips.anular, tips.auriculaire]
    base_hand = tips.base_hand

    # Vérification si assez de positions précédentes
    if len(prev_positions) < 3:
        print("Not enough previous positions to detect gesture.")
        return

    # Vérification des distances avec les positions précédentes des doigts
    for fingertip, prev_fingertip in zip(
        finger_tips,
        [
            prev_positions[-3].get("index", None),
            prev_positions[-3].get("major", None),
            prev_positions[-3].get("anular", None),
            prev_positions[-3].get("auriculaire", None),
        ],
    ):
        if prev_fingertip is None:
            print("Error: Previous fingertip not available.")
            continue
        dist_to_prev = self.distance3D(fingertip, prev_fingertip)
        if dist_to_prev[1] < 0.05:  # Seuil pour déterminer si le doigt est fermé
            closed_fingers += 1

    # Vérification de si la main est en idle (immobile)
    dim3_dist_base = self.distance3D(tips.base_hand, prev_positions[-3]["base_hand"])
    is_base_hand_moving = False

    x_margin_moving = (self.width / 4700) * hand_ratio
    y_margin_moving = (self.height / 3350) * hand_ratio

    if ((dim3_dist_base[0]) ** 2) ** 0.5 > x_margin_moving:
        print("base hand moving on x axis")
        print(((dim3_dist_base[0]) ** 2) ** 0.5, ">", x_margin_moving)
        is_base_hand_moving = True
    if ((dim3_dist_base[1]) ** 2) ** 0.5 > y_margin_moving:
        print("base hand moving on Y axis")
        print(dim3_dist_base[1], ">", y_margin_moving)
        is_base_hand_moving = True

    if not is_base_hand_moving and closed_fingers == 0:
        print("base hand >", is_base_hand_moving, "closed fingers >", closed_fingers)
        if not self.is_idle:
            self.is_idle = True
            self.idle_time = time.time()
            print("idling")
            print(
                "base hand >", is_base_hand_moving, "| closed fingers >", closed_fingers
            )

        elif self.is_idle:
            if time.time() - self.idle_time > 0.4:
                print("idle")
                self.recent_action = False
    else:
        self.is_idle = False
        self.idle_time = time.time()

    # Vérification si le pouce est sous les autres doigts (en prenant en compte la distance 3D)
    thumb_hidden = False
    for finger in finger_tips:
        dist_thumb_to_finger = self.distance3D(tips.thumb, finger)
        if (
            dist_thumb_to_finger[1] < 0.05
        ):  # Si le pouce est suffisamment proche du doigt sur l'axe Y
            thumb_hidden = True
            break

    # Affichage des données pour le débogage
    print(
        f"Closed fingers: {closed_fingers}, Thumb hidden: {thumb_hidden}, Base hand moving: {is_base_hand_moving}"
    )
    print(f"Recent action: {self.recent_action}")

    # Vérification si tous les doigts sont fermés et si le pouce est bien caché sous les autres doigts
    if closed_fingers == 4 and thumb_hidden and not is_base_hand_moving:
        if not self.recent_action:
            print("Fermeture de l'application")
            # close_browser_ctrl()
            self.recent_action = True
            close_browser_click()
        # sys.exit()
