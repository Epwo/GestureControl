import time
from src.tips import GetTips


def detect_Horns(self, tips: GetTips, prev_positions, hand_ratio):

    is_dist_index = ((self.distance3D(tips.index, tips.base_hand)[1]) ** 2) ** 0.5 > 0.4
    is_dist_auri = (
        (self.distance3D(tips.auriculaire, tips.base_hand)[1]) ** 2
    ) ** 0.5 > 0.4
    is_dist_major = (
        (self.distance3D(tips.major, tips.base_hand)[1]) ** 2
    ) ** 0.5 < 0.35
    is_dist_annu = ((self.distance3D(tips.anular, tips.base_hand)[1]) ** 2) ** 0.5 < 0.3
    is_dist_thumb = (self.distance3D(tips.thumb, tips.base_hand)[0] ** 2) ** 0.5 < 0.08

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

    dim3_dist_base = self.distance3D(tips.base_hand, prev_positions[-3]["base_hand"])

    is_base_hand_moving = False
    print(tips.index, prev_positions[-3]["index"])
    x_margin_moving = (self.width / 4700) * hand_ratio
    y_margin_moving = (self.height / 3350) * hand_ratio

    # Calculate the coordinates for the square
    if ((dim3_dist_base[0]) ** 2) ** 0.5 > x_margin_moving:  # x coords
        print("x axis")
        print(((dim3_dist_base[0]) ** 2) ** 0.5, ">", x_margin_moving)
        is_base_hand_moving = True

    if dim3_dist_base[1] > y_margin_moving:  # y coords
        print("y axis")
        is_base_hand_moving = True

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
        and not is_base_hand_moving
    ):
        if not self.recent_action:
            print(
                "HORNSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs",
            )
            print(self.recent_action, time.time())
            self.pause()
            self.recent_action = True
