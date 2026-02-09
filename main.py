import time

import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture("test-3.mp4")

cv2.namedWindow("DETECTION VIEW", cv2.WINDOW_NORMAL)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    is_locked = False
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            is_locked = True
            break

    center_color = (0, 0, 255) if is_locked else (90, 90, 90)

    overlay = frame.copy()
    cv2.line(overlay, (0, 0), (w, h), (90, 90, 90), 1)
    cv2.line(overlay, (w, 0), (0, h), (90, 90, 90), 1)

    alpha = 0.8
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.circle(frame, (cx, cy), 10, center_color, 2)

    d = 25
    cv2.line(frame, (cx - d, cy), (cx + d, cy), center_color, 2)
    cv2.line(frame, (cx, cy - d), (cx, cy + d), center_color, 2)

    if is_locked:
        cv2.putText(
            frame,
            "LOCK",
            (cx - 40, cy - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            center_color,
            2,
        )

    margin = 25
    l = 50
    color = (230, 230, 230)
    t = 2

    cv2.line(frame, (margin, margin), (margin + l, margin), color, t)
    cv2.line(frame, (margin, margin), (margin, margin + l), color, t)

    cv2.line(frame, (w - margin, margin), (w - margin - l, margin), color, t)
    cv2.line(frame, (w - margin, margin), (w - margin, margin + l), color, t)

    cv2.line(frame, (margin, h - margin), (margin + l, h - margin), color, t)
    cv2.line(frame, (margin, h - margin), (margin, h - margin - l), color, t)

    cv2.line(frame, (w - margin, h - margin), (w - margin - l, h - margin), color, t)
    cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - l), color, t)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{name} {conf:.2f}"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("DETECTION VIEW", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
