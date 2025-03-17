import time
from src.tips import GetTips
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

def get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return interface.QueryInterface(IAudioEndpointVolume)

def change_volume(up=True):
    volume = get_volume_interface()
    current = volume.GetMasterVolumeLevelScalar()
    step = 0.1  # Ajuste le volume de 10%
    new_volume = min(1.0, max(0.0, current + step if up else current - step))
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"Nouveau volume : {new_volume:.0%}")
def volume_up_and_down(self, tips: GetTips, prev_positions, hand_ratio):
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
    is_dist_thumb = (
                           (self.distance3D(tips.anular, prev_positions[-3]["thumb"])[1]) ** 2
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
    thumb_x_movement = self.distance3D(tips.thumb, prev_positions[-3]["thumb"])[0]
    anular_x_movement = self.distance3D(tips.anular, prev_positions[-3]["anular"])[0]
    major_x_movement = self.distance3D(tips.anular, prev_positions[-3]["major"])[0]
    index_x_movement = self.distance3D(tips.anular, prev_positions[-3]["index"])[0]
    print("mouvement ", tips.thumb)
    if (thumb_x_movement + 0.07) < index_x_movement:
        print("c'est bon")
    if is_dist_thumb:
        if not self.recent_action and (thumb_x_movement + 0.02) < anular_x_movement and thumb_x_movement < index_x_movement and thumb_x_movement < major_x_movement :
            print("hih", self.recent_action, time.time())
            if self.distance3D(tips.thumb, prev_positions[-3]["thumb"])[1] > 0 and -0.2 < thumb_x_movement < 0.2 and -0.2 < index_x_movement < 0.2 and -0.2 < major_x_movement < 0.2 and -0.2 < anular_x_movement < 0.2:
                change_volume(False)
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("baisse de volume", time.time())
                print("baisse de volume", time.time())
                print("baisse de volume", time.time())
                return "swipe_vert down"
            elif self.distance3D(tips.thumb, prev_positions[-3]["thumb"])[1] < 0 and -0.2 < thumb_x_movement < 0.2 and -0.2 < index_x_movement < 0.2 and -0.2 < major_x_movement < 0.2 and -0.2 < anular_x_movement < 0.2:
                change_volume(True)
                self.recent_action = True
                self.is_idle = False
                self.idle_time = time.time()
                print("augmentation de volume", time.time())
                print("augmentation de volume", time.time())
                print("augmentation de volume", time.time())
                return "swipe_vert up"