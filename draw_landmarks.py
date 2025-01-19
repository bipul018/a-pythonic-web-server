import cv2
import mediapipe as mpipe
import numpy

mp_pose = mpipe.solutions.pose
mp_drawing = mpipe.solutions.drawing_utils

BaseOptions = mpipe.tasks.BaseOptions
PoseLandmarker = mpipe.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mpipe.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mpipe.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

def load_detector():
    return PoseLandmarker.create_from_options(options)
    
def unnormalize(pt, wid, hei):
    x = (1-pt[0]) * wid
    y = pt[1] * hei
    # x = (x + 1) * 0.5 * wid
    # y = (y + 1) * 0.5 * hei
    if (x >= 0) and (y >= 0) and (x < wid) and (y < hei):
        return (int(x) , int(y))
    return None


# ts is the time in milliseconds
def run_on_image(detector, rgb_image, ts:int, also_draw=True):
    if len(rgb_image.shape) == 4:
        rgb_image = rgb_image[0]
    rgb_image = numpy.ascontiguousarray(rgb_image)
    rgb_image = rgb_image.copy()
    bgr_image = mpipe.Image(image_format=mpipe.ImageFormat.SRGB, data=cv2.flip(rgb_image,1))
    results = detector.detect_for_video(bgr_image, ts)
    if results.pose_landmarks:
        pts = []
        for pt in results.pose_landmarks[0]:
            pts.append([pt.x, pt.y, pt.z])
        # loop over all the landmark indices pairs and draw lines
        if also_draw:
            for pair in mpipe.solutions.pose.POSE_CONNECTIONS:
                pt1 = unnormalize(pts[pair[0]], rgb_image.shape[1], rgb_image.shape[0])
                pt2 = unnormalize(pts[pair[1]], rgb_image.shape[1], rgb_image.shape[0])
                if (pt1 is not None) and (pt2 is not None):
                    cv2.line(rgb_image, pt1, pt2, (0, 0, 255), thickness=1)
        else:
            rgb_images = None
        pts = []
        for pt in results.pose_world_landmarks[0]:
            pts.append([pt.x, pt.y, pt.z])
        pts = numpy.array(pts)
        return rgb_image, pts
    raise Exception(f"Couldnot detect any landmarks in frame at time {ts/1000}")



    
