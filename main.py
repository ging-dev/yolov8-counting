import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator

cap = cv2.VideoCapture('./test.mp4')

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
             cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


yolo = YOLO(model='./yolov8n.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result: Results = yolo.track(frame, classes=[0], verbose=False).pop()

    assert result.boxes is not None

    cv2.putText(frame, f'People {len(result.boxes)}',
                (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

    annotator = Annotator(frame)
    for p in result.boxes:
        annotator.box_label(p.xyxy[0], yolo.names[int(p.cls)])

    cv2.imshow('Tracking', annotator.result())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
