import time
from src.tips import GetTips
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.logic import GestureDetection


def detect_Horns(self: "GestureDetection", tips: GetTips, prev_positions, hand_ratio):

    # 1.2 dans les conditions initiales donc multiplier les seuils initiaux (0.4,0.4,0.35,0.3,0.08) par 1.2

    is_dist_index = ((self.distance3D(tips.index, tips.base_hand)[1]) ** 2) ** 0.5 > (
        hand_ratio * 0.11
    )
    is_dist_auri = (
        (self.distance3D(tips.auriculaire, tips.base_hand)[1]) ** 2
    ) ** 0.5 > (hand_ratio * 0.11)

    is_dist_major = ((self.distance3D(tips.major, tips.base_hand)[1]) ** 2) ** 0.5 < (
        hand_ratio * 0.147
    )
    is_dist_annu = ((self.distance3D(tips.anular, tips.base_hand)[1]) ** 2) ** 0.5 < (
        hand_ratio * 0.13
    )
    is_dist_thumb = (self.distance3D(tips.thumb, tips.base_hand)[0] ** 2) ** 0.5 < (
        hand_ratio * 0.04
    )

    if (
        is_dist_index
        and is_dist_auri
        and is_dist_major
        and is_dist_annu
        and is_dist_thumb
    ):
        if not self.recent_action:
            print(
                "HORNSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs",
            )
            self.press_key("playpause")
            self.recent_action = True
            self.is_idle = False
            self.idle_time = time.time()
            return "PLAY/PAUSE"
