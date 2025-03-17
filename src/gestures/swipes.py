import time
from src.tips import GetTips


def detect_Swipe(self, tips: GetTips, prev_positions, is_base_hand_moving):
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

    if is_dist_index and is_dist_major and is_dist_annu and not is_base_hand_moving:
        if not self.recent_action:
            print(
                "swipe",
                self.distance3D(tips.index, prev_positions[-3]["index"])[1],
            )
            print(self.recent_action, time.time())
            if self.distance3D(tips.index, prev_positions[-3]["index"])[1] > 0:
                self.scroll_mouse("down")
                # self.press_key("down")
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("scroll down", time.time())
                return "swipe_vert down"
            else:
                self.scroll_mouse("up")
                # self.press_key("up")
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("scroll up", time.time())
                return "swipe_vert up"
