from utils.math_utils import distance


class GestureRecognizer:

    def recognize(self, landmarks):

        # no hand detected
        if landmarks is None:
            return None

        # ignore noisy detections
        if len(landmarks) < 10:
            return None

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        d = distance(thumb_tip, index_tip)

        if d < 0.04:
            return "PINCH"

        return None
