from landmark.drawing_landmarks import load_detector, get_2d_lines


class Landmark_Streamer:
    def __init__(self, reply_factor=1):
        # reply_factor controls at how many frames will be skipped to reply
        self.detector = load_detector()
        self.reply_factor = int(reply_factor)
        assert self.reply_factor >= 1, "Comeon"
        self.counter = 0
        pass
    def run_frame(self, np_rgb_image, ts_ms):
        self.counter = (1+self.counter)%self.reply_factor
        if self.counter == 0:
            lines_2d = get_2d_lines(self.detector, np_rgb_image, ts_ms)
            if lines_2d:
                return {'timestamp': ts_ms,
                        'message': 'draw line landmarks',
                        'landmark_type': 'Blazepose/Mediapipe',
                        'landmarks': lines_2d}
            return None
            
    
    
