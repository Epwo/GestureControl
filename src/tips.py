class GetTips:
    def __init__(self, landmarks, fingers_name_nb):
        self.thumb = landmarks[fingers_name_nb["thumb"]]
        self.index = landmarks[fingers_name_nb["index"]]
        self.major = landmarks[fingers_name_nb["major"]]
        self.anular = landmarks[fingers_name_nb["anular"]]
        self.auriculaire = landmarks[fingers_name_nb["auriculaire"]]
        self.base_hand = landmarks[fingers_name_nb["base_hand"]]
