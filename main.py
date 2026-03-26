import cv2
import pyautogui
import time
import joblib
from collections import deque, Counter

from detection.hand_detector import HandDetector


cap = cv2.VideoCapture(0)

detector = HandDetector()

model = joblib.load("gesture_model.pkl")

screen_w, screen_h = pyautogui.size()

# smoothing cursor
smoothening = 8
prev_x, prev_y = 0, 0

# gesture smoothing (IMPORTANT)
gesture_history = deque(maxlen=10)

# cooldown
last_time = 0
cooldown = 0.3

dragging = False


def preprocess(landmarks):
    data = []
    for lm in landmarks:
        data.append(lm.x)
        data.append(lm.y)
    return data


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame, landmarks = detector.detect(frame)

    if landmarks:

        # cursor movement
        index = landmarks[8]

        mouse_x = int(index.x * screen_w)
        mouse_y = int(index.y * screen_h)

        curr_x = prev_x + (mouse_x - prev_x) / smoothening
        curr_y = prev_y + (mouse_y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

        # ML prediction
        data = preprocess(landmarks)
        gesture = model.predict([data])[0]

        # add to history
        gesture_history.append(gesture)

        # get most common gesture
        most_common = Counter(gesture_history).most_common(1)[0][0]

        current_time = time.time()

        if current_time - last_time > cooldown:

            print("Gesture:", most_common)

            # mapping
            if most_common == 3:   # pinch → click
                pyautogui.click()

            if most_common == 1 and not dragging:  # fist → drag
                pyautogui.mouseDown()
                dragging = True

            if most_common != 1 and dragging:
                pyautogui.mouseUp()
                dragging = False

            last_time = current_time

    cv2.imshow("Stable ML Gesture System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
