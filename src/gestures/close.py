import time
import sys
from src.tips import GetTips

import pyautogui

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.logic import GestureDetection


def close_browser_click():
    pyautogui.hotkey("alt", "f4")
    print("Application fermée via Alt+F4.")


def close_browser_ctrl():
    pyautogui.hotkey("ctrl", "w")  # Ou ('command', 'w') pour Mac
    print("Onglet fermé via raccourci.")


def detect_close_gesture(
    self: "GestureDetection",
    tips: GetTips,
    is_base_hand_moving,
    closed_fingers,
    hand_ratio,
):
    """
    Detects if the hand is fully closed, with the thumb hidden beneath the other fingers,
    and the hand is not moving (idle).
    """

    # Vérification si le pouce est sous les autres doigts (en prenant en compte la distance 3D)
    thumb_hidden = False
    # Calculate distance between thumb and base hand
    dist_thumb_to_base = self.distance3D(tips.thumb, tips.base_hand)

    # Check if thumb is closer to the base hand than it normally would be
    # when the hand is open (negative X value means thumb "inside" the hand)
    if dist_thumb_to_base[0] > (-0.0591 * hand_ratio):
        thumb_hidden = True

    # Affichage des données pour le débogage
    # Vérification si tous les doigts sont fermés et si le pouce est bien caché sous les autres doigts
    if closed_fingers == 4 and thumb_hidden and not is_base_hand_moving:
        if not self.close_recent_action:

            # close_browser_ctrl()
            close_browser_click()
            self.close_recent_action = True
            # TODO: make a validation time for closing the app (like loading circle, that then close the app)
            return "ON/OFF"
