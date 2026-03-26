import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(self):

        model_path = "hand_landmarker.task"

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )

        self.detector = vision.HandLandmarker.create_from_options(options)


    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        landmarks = None

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            h, w, _ = frame.shape

            for point in landmarks:

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

        return frame, landmarks
