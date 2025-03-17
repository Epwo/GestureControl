import time
from src.tips import GetTips


def detect_Horns(self, tips: GetTips, prev_positions, hand_ratio):

    print(
        hand_ratio
    )  # 1.2 dans les conditions initiales donc multiplier les seuils initiaux (0.4,0.4,0.35,0.3,0.08) par 1.2

    is_dist_index = ((self.distance3D(tips.index, tips.base_hand)[1]) ** 2) ** 0.5 > (
        0.48 / hand_ratio
    )
    is_dist_auri = (
        (self.distance3D(tips.auriculaire, tips.base_hand)[1]) ** 2
    ) ** 0.5 > (0.48 / hand_ratio)
    is_dist_major = ((self.distance3D(tips.major, tips.base_hand)[1]) ** 2) ** 0.5 < (
        0.42 / hand_ratio
    )
    is_dist_annu = ((self.distance3D(tips.anular, tips.base_hand)[1]) ** 2) ** 0.5 < (
        0.36 / hand_ratio
    )
    is_dist_thumb = (self.distance3D(tips.thumb, tips.base_hand)[0] ** 2) ** 0.5 < (
        0.096 / hand_ratio
    )

    # print(
    #     "Distance index : ",
    #     (self.distance3D(tips.index, tips.base_hand)[1] ** 2) ** 0.5,
    # )
    # print(
    #     "Distance auri : ",
    #     (self.distance3D(tips.auriculaire, tips.base_hand)[1] ** 2) ** 0.5,
    # )
    # print(
    #     "Distance major : ",
    #     (self.distance3D(tips.major, tips.base_hand)[1] ** 2) ** 0.5,
    # )
    # print(
    #     "Distance annu : ",
    #     (self.distance3D(tips.anular, tips.base_hand)[1] ** 2) ** 0.5,
    # )

    # print(
    #     "Distance thumb : ",
    #     (self.distance3D(tips.thumb, tips.base_hand)[0] ** 2) ** 0.5,
    # )

    # print(
    #     is_dist_index,
    #     is_dist_auri,
    #     is_dist_major,
    #     is_dist_annu,
    #     is_dist_thumb,
    #     not is_base_hand_moving,
    # )
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
            print(self.recent_action, time.time())
            self.press_key("playpause")
            self.recent_action = True
            self.is_idle = False
            self.idle_time = time.time()
