import time
import sys
from src.tips import GetTips

import pyautogui

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.logic import GestureDetection


def press_enter():
    pyautogui.press("enter")
    print("Entrée effectuée via la touche Enter.")


def detect_victory_gesture(
    self: "GestureDetection", closed_fingers, is_base_hand_moving, landmarks
):
    """
    Détecte le geste du 'V de la victoire' (index et majeur étendus, autres doigts repliés)
    et simule un appui sur Entrée.
    """

    fingers_name_nb = self.get_fingers_name_nb()

    # Vérification des doigts : 1 = étendu, 0 = replié
    index_extended = landmarks[fingers_name_nb["index"]].y < landmarks[5].y
    major_extended = landmarks[fingers_name_nb["major"]].y < landmarks[9].y

    is_victory = index_extended and major_extended and (closed_fingers == 2)

    # Affichage pour le débogage
    print(f"V détecté {is_victory}, Base hand moving: {is_base_hand_moving}")

    # Vérification du geste et de l'absence de mouvement de la main
    if is_victory and not is_base_hand_moving and not self.recent_action:
        self.recent_action = True
        press_enter()
        return "Victory gesture detected - Enter pressed"
