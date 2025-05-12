from ultralytics import YOLO
import cv2

# 클래스 매핑 (YOLOv8 COCO 기준)
TARGET_CLASSES = {
    0: 'person',
    2: 'forklift',       # car로 대체 (fine-tune 필요)
    39: 'machine',       # screwdriver → 기계 대체
    67: 'conveyor_belt'  # cell phone → 벨트 대체
}

def run_detection(image, model):
    results = model(image)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id in TARGET_CLASSES:
            name = TARGET_CLASSES[cls_id]
            x, y = box.xywh[0][:2].tolist()
            detections.append((name, (x, y)))

    return detections
