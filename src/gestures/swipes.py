import time
from tips import GetTips


def detect_Swipe(self, tips: GetTips, prev_positions, hand_ratio):
    # print('dists index & previous', distance3D(thumb_tip, prev_positions[-3]["index_tip"]))5
    is_dist_index = (
        (self.distance3D(tips.index, prev_positions[-3]["index"])[1]) ** 2
    ) ** 0.5 > 0.12
    is_dist_major = (
        (self.distance3D(tips.major, prev_positions[-3]["major"])[1]) ** 2
    ) ** 0.5 > 0.12
    is_dist_annu = (
        (self.distance3D(tips.anular, prev_positions[-3]["anular"])[1]) ** 2
    ) ** 0.5 > 0.12
    dim3_dist_base = self.distance3D(tips.base_hand, prev_positions[-3]["base_hand"])
    is_base_hand_moving = False

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

    if is_dist_index and is_dist_major and is_dist_annu and not is_base_hand_moving:
        if not self.recent_action:
            print(
                "swipe",
                self.distance3D(tips.index, prev_positions[-3]["index"])[1],
            )
            print(self.recent_action, time.time())
            if self.distance3D(tips.index, prev_positions[-3]["index"])[1] > 0:
                self.scroll_mouse("down")
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("scroll down", time.time())
                return "swipe_vert down"
            else:
                self.scroll_mouse("up")
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("scroll up", time.time())
                return "swipe_vert up"
